# coding=utf-8
"""
@author: Yantong Lai
@paper: [24 SIGIR] Disentangled Contrastive Hypergraph Learning for Next POI Recommendation
"""

import torch.nn as nn
import torch
import torch.nn.functional as F


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

    def __init__(self, alpha_t=0.1):
        super(DirectedHyperConvLayer, self).__init__()
        self.alpha_t = alpha_t
        self.last_debug_stats = {}

    def _refine_sparse_adj(self, pois_embs, sparse_adj, stats_prefix):
        """
        Refine existing sparse edges only (no densification).
        """
        sparse_adj = sparse_adj.coalesce()
        edge_index = sparse_adj.indices()      # [2, E]
        base_val = sparse_adj.values()         # [E]

        src_idx = edge_index[0]
        tar_idx = edge_index[1]
        src_emb = pois_embs[src_idx]           # [E, d]
        tar_emb = pois_embs[tar_idx]           # [E, d]

        sim = F.cosine_similarity(src_emb, tar_emb, dim=1)       # [-1, 1], [E]
        sim_norm = ((sim + 1.0) * 0.5).clamp(0.0, 1.0)           # [0, 1], [E]
        refined_val = base_val * (1.0 + self.alpha_t * sim_norm) # residual refinement

        # debug stats for inspection
        eps = 1e-8
        self.last_debug_stats[f"sim_{stats_prefix}_mean"] = sim.mean().detach().item()
        self.last_debug_stats[f"sim_{stats_prefix}_var"] = sim.var(unbiased=False).detach().item()
        self.last_debug_stats[f"refined_val_{stats_prefix}_over_base_mean"] = (
            (refined_val.mean() / (base_val.mean() + eps)).detach().item()
        )

        refined_adj = torch.sparse_coo_tensor(
            edge_index,
            refined_val,
            sparse_adj.shape,
            device=refined_val.device,
            dtype=refined_val.dtype,
        ).coalesce()

        return refined_adj

    def forward(self, pois_embs, HG_poi_src, HG_poi_tar):
        # refine edge weights on existing sparse edges only
        refined_HG_poi_src = self._refine_sparse_adj(pois_embs, HG_poi_src, "src")
        refined_HG_poi_tar = self._refine_sparse_adj(pois_embs, HG_poi_tar, "tar")

        msg_tar = torch.sparse.mm(refined_HG_poi_tar, pois_embs)
        msg_src = torch.sparse.mm(refined_HG_poi_src, msg_tar)

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
    def __init__(self, num_layers, device, dropout=0.3, alpha_t=0.1):
        super(DirectedHyperConvNetwork, self).__init__()

        self.num_layers = num_layers
        self.device = device
        self.dropout = dropout
        self.di_hconv_layer = DirectedHyperConvLayer(alpha_t=alpha_t)
        self.last_debug_stats = {}

    def forward(self, pois_embs, HG_poi_src, HG_poi_tar):
        final_pois_embs = [pois_embs]
        for layer_idx in range(self.num_layers):
            pois_embs = self.di_hconv_layer(pois_embs, HG_poi_src, HG_poi_tar)
            self.last_debug_stats = dict(self.di_hconv_layer.last_debug_stats)
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

        # temporal branch edge refinement strength
        self.alpha_t = 0.1

        # network
        self.mv_hconv_network = MultiViewHyperConvNetwork(args.num_mv_layers, args.emb_dim, 0, device)
        self.geo_conv_network = GeoConvNetwork(args.num_geo_layers, args.dropout)
        self.di_hconv_network = DirectedHyperConvNetwork(
            args.num_di_layers,
            device,
            args.dropout,
            alpha_t=self.alpha_t,
        )

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

        # BTGR-v1 parameters are defined at the end of __init__ to avoid
        # perturbing the original DCHL/fine-branch initialization order.
        self.num_prototypes = 64
        self.prototype_tau = 1.0
        self.lambda_proto = 0.1
        self.prototype_embeddings = nn.Parameter(torch.empty(self.num_prototypes, self.emb_dim))

        # RNG isolation: init BTGR params without changing subsequent/global RNG state.
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        nn.init.xavier_uniform_(self.prototype_embeddings)
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)

        self.btgr_debug_stats = {}

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
        trans_pois_embs_fine = self.di_hconv_network(seq_gate_pois_embs, dataset.HG_poi_src, dataset.HG_poi_tar)

        if self.lambda_proto == 0.0:
            # Control baseline mode: fully bypass BTGR coarse branch.
            trans_pois_embs = trans_pois_embs_fine
            with torch.no_grad():
                self.btgr_debug_stats = {
                    "max_abs_diff": (trans_pois_embs - trans_pois_embs_fine).abs().max().detach().item()
                }
        else:
            # BTGR-v1 coarse/prototype-level transition branch
            # Normalize Z and prototypes before assignment for stable cosine-like matching.
            # Z_norm: [L, d], P_norm: [K, d], assign_logits: [L, K], A: [L, K]
            z_norm = F.normalize(seq_gate_pois_embs, p=2, dim=1)
            p_norm = F.normalize(self.prototype_embeddings, p=2, dim=1)
            assign_logits = torch.matmul(z_norm, p_norm.T) / self.prototype_tau
            assignment = F.softmax(assign_logits, dim=1)

            # debug: assignment entropy (per POI), prototype mass statistics
            eps = 1e-12
            assign_entropy = -(assignment * (assignment + eps).log()).sum(dim=1)  # [L]
            prototype_mass = assignment.sum(dim=0)  # [K]

            # Dual-graph coarse propagation aligned with T branch (tar -> src), no densification on [L, L].
            trans_to_proto_tar = torch.sparse.mm(dataset.HG_poi_tar, assignment)         # [L, K]
            trans_to_proto_src = torch.sparse.mm(dataset.HG_poi_src, assignment)         # [L, K]
            g_proto_tar = torch.matmul(assignment.T, trans_to_proto_tar)                 # [K, K]
            g_proto_src = torch.matmul(assignment.T, trans_to_proto_src)                 # [K, K]

            # add self-loop and row-normalize each prototype graph
            eye_k = torch.eye(self.num_prototypes, device=g_proto_tar.device, dtype=g_proto_tar.dtype)
            g_proto_tar = g_proto_tar + eye_k
            g_proto_src = g_proto_src + eye_k
            g_proto_tar = g_proto_tar / (g_proto_tar.sum(dim=1, keepdim=True) + 1e-8)
            g_proto_src = g_proto_src / (g_proto_src.sum(dim=1, keepdim=True) + 1e-8)

            # Z_proto = (A^T @ Z) / mass  -> prototype-quality normalized weighted mean
            proto_embs_sum = torch.matmul(assignment.T, seq_gate_pois_embs)          # [K, d]
            proto_embs = proto_embs_sum / (prototype_mass.unsqueeze(1) + 1e-8)       # [K, d]
            # H_proto_mid = G_proto_tar @ Z_proto; H_proto = G_proto_src @ H_proto_mid
            proto_mid = torch.matmul(g_proto_tar, proto_embs)                        # [K, d]
            proto_propag = torch.matmul(g_proto_src, proto_mid)                      # [K, d]
            # H_coarse = A @ H_proto
            trans_pois_embs_coarse = torch.matmul(assignment, proto_propag)          # [L, d]

            # H_T_final = H_fine + lambda_proto * H_coarse
            trans_pois_embs = trans_pois_embs_fine + self.lambda_proto * trans_pois_embs_coarse

            # debug stats for BTGR-v1 (kept as attributes for external inspection)
            with torch.no_grad():
                self.btgr_debug_stats = {
                    "assign_entropy_mean": assign_entropy.mean().detach().item(),
                    "prototype_mass_mean": prototype_mass.mean().detach().item(),
                    "prototype_mass_std": prototype_mass.std(unbiased=False).detach().item(),
                    "prototype_mass_min": prototype_mass.min().detach().item(),
                    "prototype_mass_max": prototype_mass.max().detach().item(),
                    "G_proto_tar_mean": g_proto_tar.mean().detach().item(),
                    "G_proto_tar_var": g_proto_tar.var(unbiased=False).detach().item(),
                    "G_proto_src_mean": g_proto_src.mean().detach().item(),
                    "G_proto_src_var": g_proto_src.var(unbiased=False).detach().item(),
                    "H_coarse_norm_mean": trans_pois_embs_coarse.norm(dim=1).mean().detach().item(),
                    "H_fine_norm_mean": trans_pois_embs_fine.norm(dim=1).mean().detach().item(),
                }
                self.btgr_debug_stats["H_coarse_norm_over_H_fine_norm"] = (
                    self.btgr_debug_stats["H_coarse_norm_mean"] / (self.btgr_debug_stats["H_fine_norm_mean"] + 1e-8)
                )
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

        # prediction
        prediction = fusion_batch_users_embs @ fusion_pois_embs.T

        return prediction, loss_cl_user, loss_cl_poi



