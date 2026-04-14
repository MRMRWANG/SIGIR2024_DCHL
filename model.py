# coding=utf-8
"""
@author: Yantong Lai
@paper: [24 SIGIR] Disentangled Contrastive Hypergraph Learning for Next POI Recommendation
"""

import torch.nn as nn
import torch
import torch.nn.functional as F


class RegionResidualCalibration(nn.Module):
    """
    Residual calibration on top of pred_base: final = pred_base + alpha * region_score.
    Only constructed when use_region_calibration is enabled.
    """

    def __init__(self, args, num_regions, device):
        super(RegionResidualCalibration, self).__init__()
        self.emb_dim = args.emb_dim
        self.device = device
        self.recent_k = int(getattr(args, "region_recent_k", 10))
        self.sim_type = getattr(args, "region_sim_type", "dot")
        self.region_reg_temp = float(getattr(args, "region_reg_temperature", 0.1))
        self.use_dynamic_alpha_gate = int(getattr(args, "use_dynamic_alpha_gate", 0))
        self.use_region_rerank_only = int(getattr(args, "use_region_rerank_only", 0))
        self.region_rerank_topm = int(getattr(args, "region_rerank_topm", 50))

        # Optional: short-term region-aware user similarity residual (off by default)
        self.use_user_sim_residual = int(getattr(args, "use_user_sim_residual", 0))
        self.user_sim_topm = int(getattr(args, "user_sim_topm", 50))
        self.user_sim_type = getattr(args, "user_sim_type", "cosine")
        self.user_sim_recent_k = int(getattr(args, "user_sim_recent_k", self.recent_k))
        init_beta = float(getattr(args, "user_sim_beta", 0.02))
        self.beta = nn.Parameter(torch.tensor(init_beta, device=device))

        # Optional: confidence-aware adaptive gate gamma (off by default)
        self.use_confidence_gate = int(getattr(args, "use_confidence_gate", 0))
        self.conf_gate_topk = int(getattr(args, "conf_gate_topk", 20))
        if self.use_confidence_gate:
            # features: gap(top1-top5), entropy(topk), session_len, region_stability
            self.conf_gate = nn.Sequential(
                nn.Linear(4, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )
            for m in self.conf_gate:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)

        self.region_embedding = nn.Embedding(num_regions, self.emb_dim)
        nn.init.xavier_uniform_(self.region_embedding.weight)

        init_alpha = float(getattr(args, "region_calib_alpha", 0.05))
        self.alpha = nn.Parameter(torch.tensor(init_alpha, device=device))

        if self.sim_type == "mlp":
            self.user_proj = nn.Linear(self.emb_dim, self.emb_dim)
            self.region_proj = nn.Linear(self.emb_dim, self.emb_dim)
            nn.init.xavier_uniform_(self.user_proj.weight)
            nn.init.xavier_uniform_(self.region_proj.weight)

        if self.use_dynamic_alpha_gate:
            gate_in = 2 * self.emb_dim
            self.alpha_gate = nn.Sequential(
                nn.Linear(gate_in, self.emb_dim),
                nn.ReLU(),
                nn.Linear(self.emb_dim, 1),
            )
            for m in self.alpha_gate:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)

    def _recent_poi_region_pref(self, batch, poi_region_id):
        """Aggregate region embeddings from the last K valid POIs in user_seq. [BS, D]"""
        user_seq = batch["user_seq"]
        lens = batch["user_seq_len"]
        bsz = user_seq.size(0)
        prefs = []
        k = self.recent_k
        for b in range(bsz):
            L = int(lens[b].item())
            if L <= 0:
                prefs.append(torch.zeros(self.emb_dim, device=self.device))
                continue
            start = max(0, L - k)
            idx = user_seq[b, start:L].long()
            rid = poi_region_id[idx]
            emb = self.region_embedding(rid).mean(dim=0)
            prefs.append(emb)
        return torch.stack(prefs, dim=0)

    def _recent_poi_fusion_agg(self, batch, fusion_pois_embs):
        """Mean of last K POI fusion embeddings (for dynamic alpha). [BS, D]"""
        user_seq = batch["user_seq"]
        lens = batch["user_seq_len"]
        bsz = user_seq.size(0)
        k = self.recent_k
        out = []
        for b in range(bsz):
            L = int(lens[b].item())
            if L <= 0:
                out.append(torch.zeros(self.emb_dim, device=self.device))
                continue
            start = max(0, L - k)
            idx = user_seq[b, start:L].long().clamp(max=fusion_pois_embs.size(0) - 1)
            vec = fusion_pois_embs[idx].mean(dim=0)
            out.append(vec)
        return torch.stack(out, dim=0)

    def region_similarity_scores(self, user_pref, poi_region_id):
        """user_pref [BS, D], poi_region_id [L] -> scores [BS, L]"""
        r_emb = self.region_embedding(poi_region_id)
        if self.sim_type == "cosine":
            u = F.normalize(user_pref, p=2, dim=1)
            r = F.normalize(r_emb, p=2, dim=1)
            return u @ r.T
        if self.sim_type == "mlp":
            u = self.user_proj(user_pref)
            r = self.region_proj(r_emb)
            return u @ r.T
        return user_pref @ r_emb.T

    def _all_users_recent_region_pref(self, dataset, poi_region_id, recent_k):
        """
        Build short-term region preference for all users from dataset.pad_all_train_sessions.
        Returns [U, D]. This is used only when user-sim residual is enabled.
        """
        pad_seqs = dataset.pad_all_train_sessions  # [U, MAX_LEN]
        padding_idx = dataset.padding_idx
        num_users = pad_seqs.size(0)
        prefs = []
        for u in range(num_users):
            seq = pad_seqs[u]
            valid = seq != padding_idx
            L = int(valid.sum().item())
            if L <= 0:
                prefs.append(torch.zeros(self.emb_dim, device=self.device))
                continue
            start = max(0, L - recent_k)
            idx = seq[start:L].long()
            rid = poi_region_id[idx]
            prefs.append(self.region_embedding(rid).mean(dim=0))
        return torch.stack(prefs, dim=0)

    def _user_similarity_scores(self, batch_user_pref, all_user_pref):
        """batch_user_pref [BS,D], all_user_pref [U,D] -> sim [BS,U]"""
        if self.user_sim_type == "dot":
            return batch_user_pref @ all_user_pref.T
        # default cosine
        bu = F.normalize(batch_user_pref, p=2, dim=1)
        au = F.normalize(all_user_pref, p=2, dim=1)
        return bu @ au.T

    def _user_sim_score(self, batch, dataset, poi_region_id):
        """
        Compute user_sim_score [BS, L] by aggregating top-m similar users'
        short-term region preference, then scoring candidate regions.
        """
        batch_user_pref = self._recent_poi_region_pref(batch, poi_region_id)  # [BS,D]
        all_user_pref = self._all_users_recent_region_pref(dataset, poi_region_id, self.user_sim_recent_k)  # [U,D]

        sim = self._user_similarity_scores(batch_user_pref, all_user_pref)  # [BS,U]
        # exclude self
        user_idx = batch["user_idx"].long().view(-1)
        sim.scatter_(1, user_idx.unsqueeze(1), float("-inf"))

        topm = min(self.user_sim_topm, sim.size(1))
        if topm <= 0:
            agg = torch.zeros_like(batch_user_pref)
        else:
            topv, topi = torch.topk(sim, k=topm, dim=1)
            w = F.softmax(topv, dim=1)  # [BS,topm]
            neigh = all_user_pref[topi]  # [BS,topm,D]
            agg = torch.sum(neigh * w.unsqueeze(-1), dim=1)  # [BS,D]

        r_emb = self.region_embedding(poi_region_id)  # [L,D]
        return agg @ r_emb.T  # [BS,L]

    def _region_stability(self, batch, poi_region_id):
        """Fraction of unique regions in last K visits. Lower => more stable. [BS]"""
        user_seq = batch["user_seq"]
        lens = batch["user_seq_len"]
        bsz = user_seq.size(0)
        k = max(1, self.recent_k)
        out = []
        for b in range(bsz):
            L = int(lens[b].item())
            if L <= 0:
                out.append(torch.tensor(1.0, device=self.device))
                continue
            start = max(0, L - k)
            idx = user_seq[b, start:L].long()
            rid = poi_region_id[idx]
            uniq = int(torch.unique(rid).numel())
            out.append(torch.tensor(uniq / max(1, rid.numel()), device=self.device))
        return torch.stack(out, dim=0)

    def _confidence_gamma(self, pred_base, batch, poi_region_id):
        """
        Compute gamma in [0,1] from pred_base confidence and short-term signals.
        Features (per user):
          gap: top1 - top5
          entropy: softmax entropy over top-k
          session_len: normalized
          region_stability: unique_ratio(lastK)
        """
        topk = min(max(5, self.conf_gate_topk), pred_base.size(1))
        vals, _ = torch.topk(pred_base, k=topk, dim=1)  # [BS,topk]
        gap = (vals[:, 0] - vals[:, 4]).unsqueeze(1)  # [BS,1]
        p = F.softmax(vals, dim=1)
        entropy = (-torch.sum(p * torch.log(p + 1e-8), dim=1)).unsqueeze(1)  # [BS,1]
        sess_len = batch["user_seq_len"].float()
        sess_len = (sess_len / (sess_len.max().clamp_min(1.0))).unsqueeze(1)  # [BS,1]
        reg_stab = self._region_stability(batch, poi_region_id).unsqueeze(1)  # [BS,1]
        feats = torch.cat([gap, entropy, sess_len, reg_stab], dim=1)  # [BS,4]
        return torch.sigmoid(self.conf_gate(feats))  # [BS,1]

    def forward(self, pred_base, batch, dataset, fusion_batch_users_embs, fusion_pois_embs):
        poi_region_id = dataset.poi_region_id
        user_pref = self._recent_poi_region_pref(batch, poi_region_id)
        region_scores = self.region_similarity_scores(user_pref, poi_region_id)

        if self.use_dynamic_alpha_gate:
            recent_fused = self._recent_poi_fusion_agg(batch, fusion_pois_embs)
            gate_in = torch.cat([fusion_batch_users_embs, recent_fused], dim=-1)
            gate = torch.sigmoid(self.alpha_gate(gate_in))
            alpha_eff = self.alpha * gate
            calibrated_scores = alpha_eff * region_scores
        else:
            calibrated_scores = self.alpha * region_scores

        # optional user-sim residual
        user_sim_scores = torch.zeros_like(pred_base)
        if self.use_user_sim_residual:
            user_sim_scores = self._user_sim_score(batch, dataset, poi_region_id)
            calibrated_scores = calibrated_scores + self.beta * user_sim_scores

        # optional confidence-aware gate gamma on total calibration residual
        if self.use_confidence_gate:
            gamma = self._confidence_gamma(pred_base, batch, poi_region_id)  # [BS,1]
            calibrated_scores = gamma * calibrated_scores

        if self.use_region_rerank_only:
            # Re-rank only top-M candidates from pred_base; keep others unchanged.
            topm = min(self.region_rerank_topm, pred_base.size(1))
            if topm <= 0:
                pred = pred_base
            else:
                _, topm_idx = torch.topk(pred_base, k=topm, dim=1)
                rerank_delta = torch.zeros_like(pred_base)
                rerank_delta.scatter_(1, topm_idx, calibrated_scores.gather(1, topm_idx))
                pred = pred_base + rerank_delta
        else:
            pred = pred_base + calibrated_scores
        return pred, user_pref

    def contrastive_region_loss(self, user_pref, label, poi_region_id):
        """Optional regularization on region calibration path only."""
        pos_rid = poi_region_id[label.long()]
        pos_emb = self.region_embedding(pos_rid)
        u = F.normalize(user_pref, p=2, dim=1)
        p = F.normalize(pos_emb, p=2, dim=1)
        pos_logits = torch.sum(u * p, dim=1) / self.region_reg_temp
        neg_rid = torch.randint(
            0, self.region_embedding.num_embeddings, (label.size(0),), device=self.device, dtype=torch.long
        )
        neg_emb = self.region_embedding(neg_rid)
        n = F.normalize(neg_emb, p=2, dim=1)
        neg_logits = torch.sum(u * n, dim=1) / self.region_reg_temp
        loss = -torch.log(torch.sigmoid(pos_logits - neg_logits) + 1e-8)
        return loss.mean()


class MultiViewHyperConvLayer(nn.Module):
    """
    Multi-view Hypergraph Convolutional Layer
    """

    def __init__(self, emb_dim, device):
        super(MultiViewHyperConvLayer, self).__init__()

        # self.fc_seq = nn.Linear(2 * emb_dim, emb_dim, bias=True, device=device)
        self.fc_fusion = nn.Linear(2 * emb_dim, emb_dim, device=device)
        self.dropout = nn.Dropout(0.3)
        self.emb_dim = emb_dim
        self.device = device

    def forward(self, pois_embs, pad_all_train_sessions, HG_up, HG_pu):
        # pois_embs = [L, d]
        # H_pu = [L, U]
        # H_up = [U, L]
        # pad_all_train_session = [U, MAX_SESS_LEN]

        # 1. node -> hyperedge message
        # 1) poi node aggregation
        msg_poi_agg = torch.sparse.mm(HG_up, pois_embs)  # [U, d]

        # 2. propagation: hyperedge -> node
        # propag_pois_embs = torch.sparse.mm(HG_poi_session, msg_emb)    # [L, d]
        propag_pois_embs = torch.sparse.mm(HG_pu, msg_poi_agg)  # [L, d]
        # propag_pois_embs = self.dropout(propag_pois_embs)

        return propag_pois_embs


class DirectedHyperConvLayer(nn.Module):
    """Directed hypergraph convolutional layer"""

    def __init__(self):
        super(DirectedHyperConvLayer, self).__init__()

    def forward(self, pois_embs, HG_poi_src, HG_poi_tar):
        msg_tar = torch.sparse.mm(HG_poi_tar, pois_embs)
        msg_src = torch.sparse.mm(HG_poi_src, msg_tar)

        return msg_src


class MultiViewHyperConvNetwork(nn.Module):
    """
    Multi-view Hypergraph Convolutional Network
    """

    def __init__(self, num_layers, emb_dim, dropout, device):
        super(MultiViewHyperConvNetwork, self).__init__()

        self.num_layers = num_layers
        self.device = device
        self.mv_hconv_layer = MultiViewHyperConvLayer(emb_dim, device)
        self.dropout = dropout

    def forward(self, pois_embs, pad_all_train_sessions, HG_up, HG_pu):
        final_pois_embs = [pois_embs]
        for layer_idx in range(self.num_layers):
            pois_embs = self.mv_hconv_layer(pois_embs, pad_all_train_sessions, HG_up, HG_pu)  # [L, d]
            # add residual connection to alleviate over-smoothing issue
            pois_embs = pois_embs + final_pois_embs[-1]
            pois_embs = F.dropout(pois_embs, self.dropout)
            final_pois_embs.append(pois_embs)
        final_pois_embs = torch.mean(torch.stack(final_pois_embs), dim=0)  # [L, d]

        return final_pois_embs


class DirectedHyperConvNetwork(nn.Module):
    def __init__(self, num_layers, device, dropout=0.3):
        super(DirectedHyperConvNetwork, self).__init__()

        self.num_layers = num_layers
        self.device = device
        self.dropout = dropout
        self.di_hconv_layer = DirectedHyperConvLayer()

    def forward(self, pois_embs, HG_poi_src, HG_poi_tar):
        final_pois_embs = [pois_embs]
        for layer_idx in range(self.num_layers):
            pois_embs = self.di_hconv_layer(pois_embs, HG_poi_src, HG_poi_tar)
            # add residual connection
            pois_embs = pois_embs + final_pois_embs[-1]
            pois_embs = F.dropout(pois_embs, self.dropout)
            final_pois_embs.append(pois_embs)
        final_pois_embs = torch.mean(torch.stack(final_pois_embs), dim=0)  # [L, d]

        return final_pois_embs


class GeoConvNetwork(nn.Module):
    def __init__(self, num_layers, dropout):
        super(GeoConvNetwork, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, pois_embs, geo_graph):
        final_pois_embs = [pois_embs]
        for _ in range(self.num_layers):
            # pois_embs = geo_graph @ pois_embs
            pois_embs = torch.sparse.mm(geo_graph, pois_embs)
            pois_embs = pois_embs + final_pois_embs[-1]
            # pois_embs = F.dropout(pois_embs, self.dropout)
            final_pois_embs.append(pois_embs)
        output_pois_embs = torch.mean(torch.stack(final_pois_embs), dim=0)  # [L, d]

        return output_pois_embs


class DCHL(nn.Module):
    def __init__(self, num_users, num_pois, args, device):
        super(DCHL, self).__init__()

        # definition
        self.num_users = num_users
        self.num_pois = num_pois
        self.args = args
        self.device = device
        self.emb_dim = args.emb_dim
        self.ssl_temp = args.temperature

        # embedding
        self.user_embedding = nn.Embedding(num_users, self.emb_dim)
        self.poi_embedding = nn.Embedding(num_pois + 1, self.emb_dim, padding_idx=num_pois)

        # embedding init
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.poi_embedding.weight)

        # network
        self.mv_hconv_network = MultiViewHyperConvNetwork(args.num_mv_layers, args.emb_dim, 0, device)
        self.geo_conv_network = GeoConvNetwork(args.num_geo_layers, args.dropout)
        self.di_hconv_network = DirectedHyperConvNetwork(args.num_di_layers, device, args.dropout)

        # gate for adaptive fusion with pois embeddings
        self.hyper_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())
        self.gcn_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())
        self.trans_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())

        # gate for adaptive fusion with users embeddings
        self.user_hyper_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())
        self.user_gcn_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())

        # temporal-augmentation
        self.pos_embeddings = nn.Embedding(1500, self.emb_dim, padding_idx=0)
        self.w_1 = nn.Linear(2 * self.emb_dim, self.emb_dim)
        self.w_2 = nn.Parameter(torch.Tensor(self.emb_dim, 1))
        self.glu1 = nn.Linear(self.emb_dim, self.emb_dim)
        self.glu2 = nn.Linear(self.emb_dim, self.emb_dim, bias=False)

        # gating before disentangled learning
        self.w_gate_geo = nn.Parameter(torch.FloatTensor(args.emb_dim, args.emb_dim))
        self.b_gate_geo = nn.Parameter(torch.FloatTensor(1, args.emb_dim))
        self.w_gate_seq = nn.Parameter(torch.FloatTensor(args.emb_dim, args.emb_dim))
        self.b_gate_seq = nn.Parameter(torch.FloatTensor(1, args.emb_dim))
        self.w_gate_col = nn.Parameter(torch.FloatTensor(args.emb_dim, args.emb_dim))
        self.b_gate_col = nn.Parameter(torch.FloatTensor(1, args.emb_dim))
        nn.init.xavier_normal_(self.w_gate_geo.data)
        nn.init.xavier_normal_(self.b_gate_geo.data)
        nn.init.xavier_normal_(self.w_gate_seq.data)
        nn.init.xavier_normal_(self.b_gate_seq.data)
        nn.init.xavier_normal_(self.w_gate_col.data)
        nn.init.xavier_normal_(self.b_gate_col.data)

        # dropout
        self.dropout = nn.Dropout(args.dropout)

        # Optional: prediction-layer region residual (off by default; preserves original DCHL)
        self.region_calib = None
        if int(getattr(args, "use_region_calibration", 0)):
            num_regions = int(getattr(args, "region_lat_bins", 16)) * int(getattr(args, "region_lon_bins", 16))
            self.region_calib = RegionResidualCalibration(args, num_regions, device)

    @staticmethod
    def row_shuffle(embedding):
        corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]

        return corrupted_embedding

    def cal_loss_infonce(self, emb1, emb2):
        pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / self.ssl_temp)
        neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / self.ssl_temp), axis=1)
        loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))
        loss /= pos_score.shape[0]

        return loss

    def cal_loss_cl_pois(self, hg_pois_embs, geo_pois_embs, trans_pois_embs):
        # projection
        # proj_hg_pois_embs = self.proj_hg(hg_pois_embs)
        # proj_geo_pois_embs = self.proj_geo(geo_pois_embs)
        # proj_trans_pois_embs = self.proj_trans(trans_pois_embs)

        # normalization
        norm_hg_pois_embs = F.normalize(hg_pois_embs, p=2, dim=1)
        norm_geo_pois_embs = F.normalize(geo_pois_embs, p=2, dim=1)
        norm_trans_pois_embs = F.normalize(trans_pois_embs, p=2, dim=1)

        # calculate loss
        loss_cl_pois = 0.0
        loss_cl_pois += self.cal_loss_infonce(norm_hg_pois_embs, norm_geo_pois_embs)
        loss_cl_pois += self.cal_loss_infonce(norm_hg_pois_embs, norm_trans_pois_embs)
        loss_cl_pois += self.cal_loss_infonce(norm_geo_pois_embs, norm_trans_pois_embs)

        return loss_cl_pois

    def cal_loss_cl_users(self, hg_batch_users_embs, geo_batch_users_embs, trans_batch_users_embs):
        # normalization
        norm_hg_batch_users_embs = F.normalize(hg_batch_users_embs, p=2, dim=1)
        norm_geo_batch_users_embs = F.normalize(geo_batch_users_embs, p=2, dim=1)
        norm_trans_batch_users_embs = F.normalize(trans_batch_users_embs, p=2, dim=1)

        # calculate loss
        loss_cl_users = 0.0
        loss_cl_users += self.cal_loss_infonce(norm_hg_batch_users_embs, norm_geo_batch_users_embs)
        loss_cl_users += self.cal_loss_infonce(norm_hg_batch_users_embs, norm_trans_batch_users_embs)
        loss_cl_users += self.cal_loss_infonce(norm_geo_batch_users_embs, norm_trans_batch_users_embs)

        return loss_cl_users

    def forward(self, dataset, batch):

        # self-gating input
        geo_gate_pois_embs = torch.multiply(self.poi_embedding.weight[:-1],
                                            torch.sigmoid(torch.matmul(self.poi_embedding.weight[:-1],
                                                                       self.w_gate_geo) + self.b_gate_geo))
        seq_gate_pois_embs = torch.multiply(self.poi_embedding.weight[:-1],
                                            torch.sigmoid(torch.matmul(self.poi_embedding.weight[:-1],
                                                                       self.w_gate_seq) + self.b_gate_seq))
        col_gate_pois_embs = torch.multiply(self.poi_embedding.weight[:-1],
                                            torch.sigmoid(torch.matmul(self.poi_embedding.weight[:-1],
                                                                       self.w_gate_col) + self.b_gate_col))

        # multi-view hypergraph convolutional network
        hg_pois_embs = self.mv_hconv_network(col_gate_pois_embs, dataset.pad_all_train_sessions, dataset.HG_up, dataset.HG_pu)
        # hypergraph structure aware users embeddings
        hg_structural_users_embs = torch.sparse.mm(dataset.HG_up, hg_pois_embs)  # [U, d]
        hg_batch_users_embs = hg_structural_users_embs[batch["user_idx"]]  # [BS, d]

        # poi-poi geographical graph convolutional network
        geo_pois_embs = self.geo_conv_network(geo_gate_pois_embs, dataset.poi_geo_graph)  # [L, d]
        # geo-aware user embeddings
        geo_structural_users_embs = torch.sparse.mm(dataset.HG_up, geo_pois_embs)
        geo_batch_users_embs = geo_structural_users_embs[batch["user_idx"]]  # [BS, d]

        # poi-poi directed hypergraph
        trans_pois_embs = self.di_hconv_network(seq_gate_pois_embs, dataset.HG_poi_src, dataset.HG_poi_tar)
        # transition-aware user embeddings
        trans_structural_users_embs = torch.sparse.mm(dataset.HG_up, trans_pois_embs)
        trans_batch_users_embs = trans_structural_users_embs[batch["user_idx"]]  # [BS, d]

        # cross view contrastive learning
        loss_cl_poi = self.cal_loss_cl_pois(hg_pois_embs, geo_pois_embs, trans_pois_embs)
        loss_cl_user = self.cal_loss_cl_users(hg_batch_users_embs, geo_batch_users_embs, trans_batch_users_embs)

        # normalization
        norm_hg_pois_embs = F.normalize(hg_pois_embs, p=2, dim=1)
        norm_geo_pois_embs = F.normalize(geo_pois_embs, p=2, dim=1)
        norm_trans_pois_embs = F.normalize(trans_pois_embs, p=2, dim=1)

        norm_hg_batch_users_embs = F.normalize(hg_batch_users_embs, p=2, dim=1)
        norm_geo_batch_users_embs = F.normalize(geo_batch_users_embs, p=2, dim=1)
        norm_trans_batch_users_embs = F.normalize(trans_batch_users_embs, p=2, dim=1)

        # adaptive fusion for user embeddings
        hyper_coef = self.hyper_gate(norm_hg_batch_users_embs)
        geo_coef = self.gcn_gate(norm_geo_batch_users_embs)
        trans_coef = self.trans_gate(norm_trans_batch_users_embs)

        # final fusion for user and poi embeddings
        fusion_batch_users_embs = hyper_coef * norm_hg_batch_users_embs + geo_coef * norm_geo_batch_users_embs + trans_coef * norm_trans_batch_users_embs
        fusion_pois_embs = norm_hg_pois_embs + norm_geo_pois_embs + norm_trans_pois_embs

        # prediction (pred_base); optional region residual calibration on top
        prediction = fusion_batch_users_embs @ fusion_pois_embs.T
        loss_region = torch.tensor(0.0, device=self.device)
        if self.region_calib is not None:
            prediction, user_pref = self.region_calib(
                prediction, batch, dataset, fusion_batch_users_embs, fusion_pois_embs
            )
            if float(getattr(self.args, "lambda_region_reg", 0.0)) > 0.0:
                loss_region = self.region_calib.contrastive_region_loss(
                    user_pref, batch["label"], dataset.poi_region_id
                )

        return prediction, loss_cl_user, loss_cl_poi, loss_region



