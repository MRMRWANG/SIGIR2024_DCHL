# coding=utf-8
"""
@author: Yantong Lai
@paper: [24 SIGIR] Disentangled Contrastive Hypergraph Learning for Next POI Recommendation
"""

import torch.nn as nn
import torch
import torch.nn.functional as F


class RSEAViewAggregator(nn.Module):
    def __init__(self, emb_dim, num_views=3, evidence_dim=None):
        super(RSEAViewAggregator, self).__init__()

        self.num_views = num_views
        self.evidence_dim = evidence_dim if evidence_dim is not None else emb_dim
        self.evidence_head = nn.Linear(emb_dim, self.evidence_dim)
        self.raw_eta = nn.Parameter(torch.zeros(1))
        self.register_buffer("I_view", torch.eye(num_views))
        self.W_delta = nn.Parameter(torch.zeros(num_views, num_views))

    def forward(self, view_embeddings, return_reliability=False):
        # view_embeddings: [N, V, D]
        evidence = F.softplus(self.evidence_head(view_embeddings))   # [N, V, K]
        alpha = evidence + 1.0
        S = alpha.sum(dim=-1, keepdim=True)
        uncertainty = float(self.evidence_dim) / (S + 1e-8)
        reliability = 1.0 - uncertainty

        eta = 0.1 * torch.tanh(self.raw_eta)
        modulated_views = view_embeddings * (1.0 + eta * reliability)

        W_view = self.I_view + 0.1 * torch.tanh(self.W_delta)
        mixed_views = torch.einsum('vw,nwd->nvd', W_view, modulated_views)
        fused_embeddings = mixed_views.sum(dim=1)

        if return_reliability:
            return fused_embeddings, reliability

        return fused_embeddings


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

        # dual-side RSEA-inspired reliable view fusion
        self.user_rsea_aggregator = RSEAViewAggregator(args.emb_dim)
        self.poi_rsea_aggregator = RSEAViewAggregator(args.emb_dim)
        self.raw_alpha_user = nn.Parameter(torch.zeros(1))
        self.raw_alpha_poi = nn.Parameter(torch.zeros(1))

    @staticmethod
    def row_shuffle(embedding):
        corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]

        return corrupted_embedding

    def cal_loss_infonce(self, emb1, emb2, sample_weight=None):
        pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / self.ssl_temp)
        neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / self.ssl_temp), axis=1)
        loss = -torch.log(pos_score / (neg_score + 1e-8) + 1e-8)

        if sample_weight is None:
            return loss.mean()

        weight = sample_weight.reshape(-1).to(device=loss.device, dtype=loss.dtype)
        loss = torch.sum(weight * loss) / (torch.sum(weight) + 1e-8)

        return loss

    def cal_loss_cl_pois(self, hg_pois_embs, geo_pois_embs, trans_pois_embs, pair_weights=None):
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
        if pair_weights is None:
            loss_cl_pois += self.cal_loss_infonce(norm_hg_pois_embs, norm_geo_pois_embs)
            loss_cl_pois += self.cal_loss_infonce(norm_hg_pois_embs, norm_trans_pois_embs)
            loss_cl_pois += self.cal_loss_infonce(norm_geo_pois_embs, norm_trans_pois_embs)
        else:
            loss_cl_pois += self.cal_loss_infonce(norm_hg_pois_embs, norm_geo_pois_embs, pair_weights["hg_geo"])
            loss_cl_pois += self.cal_loss_infonce(norm_hg_pois_embs, norm_trans_pois_embs, pair_weights["hg_trans"])
            loss_cl_pois += self.cal_loss_infonce(norm_geo_pois_embs, norm_trans_pois_embs, pair_weights["geo_trans"])

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
        user_views = torch.stack([norm_hg_batch_users_embs, norm_geo_batch_users_embs, norm_trans_batch_users_embs], dim=1)
        poi_views = torch.stack([norm_hg_pois_embs, norm_geo_pois_embs, norm_trans_pois_embs], dim=1)

        fusion_batch_users_embs = hyper_coef * norm_hg_batch_users_embs + geo_coef * norm_geo_batch_users_embs + trans_coef * norm_trans_batch_users_embs
        fusion_pois_embs = norm_hg_pois_embs + norm_geo_pois_embs + norm_trans_pois_embs

        user_rsea_embs = self.user_rsea_aggregator(user_views)
        poi_rsea_embs, poi_reliability = self.poi_rsea_aggregator(poi_views, return_reliability=True)

        alpha_user = 0.1 * torch.tanh(self.raw_alpha_user)
        alpha_poi = 0.1 * torch.tanh(self.raw_alpha_poi)

        fusion_batch_users_embs = fusion_batch_users_embs + alpha_user * (user_rsea_embs - fusion_batch_users_embs)
        fusion_pois_embs = fusion_pois_embs + alpha_poi * (poi_rsea_embs - fusion_pois_embs)

        # cross view contrastive learning
        r_hg = poi_reliability[:, 0, 0]
        r_geo = poi_reliability[:, 1, 0]
        r_trans = poi_reliability[:, 2, 0]
        pair_weights = {
            "hg_geo": (0.5 * (r_hg + r_geo)).detach(),
            "hg_trans": (0.5 * (r_hg + r_trans)).detach(),
            "geo_trans": (0.5 * (r_geo + r_trans)).detach(),
        }
        loss_cl_poi = self.cal_loss_cl_pois(hg_pois_embs, geo_pois_embs, trans_pois_embs, pair_weights=pair_weights)
        loss_cl_user = self.cal_loss_cl_users(hg_batch_users_embs, geo_batch_users_embs, trans_batch_users_embs)

        # prediction
        prediction = fusion_batch_users_embs @ fusion_pois_embs.T

        return prediction, loss_cl_user, loss_cl_poi



