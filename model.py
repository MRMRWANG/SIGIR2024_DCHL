# coding=utf-8
"""
@author: Yantong Lai
@paper: [24 SIGIR] Disentangled Contrastive Hypergraph Learning for Next POI Recommendation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiViewHyperConvLayer(nn.Module):
    def __init__(self, emb_dim, device):
        super(MultiViewHyperConvLayer, self).__init__()
        self.fc_fusion = nn.Linear(2 * emb_dim, emb_dim, device=device)
        self.dropout = nn.Dropout(0.3)
        self.emb_dim = emb_dim
        self.device = device

    def forward(self, pois_embs, pad_all_train_sessions, HG_up, HG_pu):
        msg_poi_agg = torch.sparse.mm(HG_up, pois_embs)  # [U, d]
        propag_pois_embs = torch.sparse.mm(HG_pu, msg_poi_agg)  # [L, d]
        return propag_pois_embs


class DirectedHyperConvLayer(nn.Module):
    def __init__(self):
        super(DirectedHyperConvLayer, self).__init__()

    def forward(self, pois_embs, HG_poi_src, HG_poi_tar):
        msg_tar = torch.sparse.mm(HG_poi_tar, pois_embs)
        msg_src = torch.sparse.mm(HG_poi_src, msg_tar)
        return msg_src


class MultiViewHyperConvNetwork(nn.Module):
    def __init__(self, num_layers, emb_dim, dropout, device):
        super(MultiViewHyperConvNetwork, self).__init__()
        self.num_layers = num_layers
        self.device = device
        self.mv_hconv_layer = MultiViewHyperConvLayer(emb_dim, device)
        self.dropout = dropout

    def forward(self, pois_embs, pad_all_train_sessions, HG_up, HG_pu):
        final_pois_embs = [pois_embs]
        for _ in range(self.num_layers):
            pois_embs = self.mv_hconv_layer(pois_embs, pad_all_train_sessions, HG_up, HG_pu)
            pois_embs = pois_embs + final_pois_embs[-1]
            pois_embs = F.dropout(pois_embs, p=self.dropout, training=self.training)
            final_pois_embs.append(pois_embs)
        final_pois_embs = torch.mean(torch.stack(final_pois_embs), dim=0)
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
        for _ in range(self.num_layers):
            pois_embs = self.di_hconv_layer(pois_embs, HG_poi_src, HG_poi_tar)
            pois_embs = pois_embs + final_pois_embs[-1]
            pois_embs = F.dropout(pois_embs, p=self.dropout, training=self.training)
            final_pois_embs.append(pois_embs)
        final_pois_embs = torch.mean(torch.stack(final_pois_embs), dim=0)
        return final_pois_embs


class GeoConvNetwork(nn.Module):
    def __init__(self, num_layers, dropout):
        super(GeoConvNetwork, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, pois_embs, geo_graph):
        final_pois_embs = [pois_embs]
        for _ in range(self.num_layers):
            pois_embs = torch.sparse.mm(geo_graph, pois_embs)
            pois_embs = pois_embs + final_pois_embs[-1]
            final_pois_embs.append(pois_embs)
        output_pois_embs = torch.mean(torch.stack(final_pois_embs), dim=0)
        return output_pois_embs


class DCHL(nn.Module):
    def __init__(self, num_users, num_pois, args, device):
        super(DCHL, self).__init__()
        self.num_users = num_users
        self.num_pois = num_pois
        self.args = args
        self.device = device
        self.emb_dim = args.emb_dim
        self.ssl_temp = args.temperature

        self.user_embedding = nn.Embedding(num_users, self.emb_dim)
        self.poi_embedding = nn.Embedding(num_pois + 1, self.emb_dim, padding_idx=num_pois)

        # 真实 region 从 1 开始，0 留给 padding
        self.num_regions = args.region_bins * args.region_bins
        self.region_embedding = nn.Embedding(self.num_regions + 1, self.emb_dim, padding_idx=0)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.poi_embedding.weight)
        nn.init.xavier_uniform_(self.region_embedding.weight)

        self.mv_hconv_network = MultiViewHyperConvNetwork(
            args.num_mv_layers, args.emb_dim, 0, device
        )
        self.geo_conv_network = GeoConvNetwork(args.num_geo_layers, args.dropout)
        self.di_hconv_network = DirectedHyperConvNetwork(
            args.num_di_layers, device, args.dropout
        )

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

        # 新增：区域约束相似邻居增强
        self.user_res_beta = args.user_res_beta
        self.mix_beta = args.mix_beta

        self.neighbor_query = nn.Linear(self.emb_dim * 2, self.emb_dim)
        self.neighbor_key = nn.Linear(self.emb_dim, self.emb_dim)
        self.neighbor_value = nn.Linear(self.emb_dim, self.emb_dim)

        # 改成 softmax gate，初始尽量接近原版平均融合
        self.region_view_gate = nn.Sequential(
            nn.Linear(self.emb_dim * 4, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, 3)
        )
        nn.init.zeros_(self.region_view_gate[2].weight)
        nn.init.zeros_(self.region_view_gate[2].bias)

        self.dropout = nn.Dropout(args.dropout)

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
        norm_hg_pois_embs = F.normalize(hg_pois_embs, p=2, dim=1)
        norm_geo_pois_embs = F.normalize(geo_pois_embs, p=2, dim=1)
        norm_trans_pois_embs = F.normalize(trans_pois_embs, p=2, dim=1)

        loss_cl_pois = 0.0
        loss_cl_pois += self.cal_loss_infonce(norm_hg_pois_embs, norm_geo_pois_embs)
        loss_cl_pois += self.cal_loss_infonce(norm_hg_pois_embs, norm_trans_pois_embs)
        loss_cl_pois += self.cal_loss_infonce(norm_geo_pois_embs, norm_trans_pois_embs)
        return loss_cl_pois

    def cal_loss_cl_users(self, hg_batch_users_embs, geo_batch_users_embs, trans_batch_users_embs):
        norm_hg_batch_users_embs = F.normalize(hg_batch_users_embs, p=2, dim=1)
        norm_geo_batch_users_embs = F.normalize(geo_batch_users_embs, p=2, dim=1)
        norm_trans_batch_users_embs = F.normalize(trans_batch_users_embs, p=2, dim=1)

        loss_cl_users = 0.0
        loss_cl_users += self.cal_loss_infonce(norm_hg_batch_users_embs, norm_geo_batch_users_embs)
        loss_cl_users += self.cal_loss_infonce(norm_hg_batch_users_embs, norm_trans_batch_users_embs)
        loss_cl_users += self.cal_loss_infonce(norm_geo_batch_users_embs, norm_trans_batch_users_embs)
        return loss_cl_users

    def forward(self, dataset, batch):
        user_idx = batch["user_idx"].long().to(self.device)
        current_region = batch["current_region"].long().to(self.device)
        neighbor_users = batch["neighbor_users"].long().to(self.device)     # [B, K]
        neighbor_mask = batch["neighbor_mask"].float().to(self.device)      # [B, K]

        geo_gate_pois_embs = torch.multiply(
            self.poi_embedding.weight[:-1],
            torch.sigmoid(torch.matmul(self.poi_embedding.weight[:-1], self.w_gate_geo) + self.b_gate_geo)
        )
        seq_gate_pois_embs = torch.multiply(
            self.poi_embedding.weight[:-1],
            torch.sigmoid(torch.matmul(self.poi_embedding.weight[:-1], self.w_gate_seq) + self.b_gate_seq)
        )
        col_gate_pois_embs = torch.multiply(
            self.poi_embedding.weight[:-1],
            torch.sigmoid(torch.matmul(self.poi_embedding.weight[:-1], self.w_gate_col) + self.b_gate_col)
        )

        hg_pois_embs = self.mv_hconv_network(
            col_gate_pois_embs, dataset.pad_all_train_sessions, dataset.HG_up, dataset.HG_pu
        )
        geo_pois_embs = self.geo_conv_network(geo_gate_pois_embs, dataset.poi_geo_graph)
        trans_pois_embs = self.di_hconv_network(
            seq_gate_pois_embs, dataset.HG_poi_src, dataset.HG_poi_tar
        )

        hg_structural_users_embs = torch.sparse.mm(dataset.HG_up, hg_pois_embs)
        geo_structural_users_embs = torch.sparse.mm(dataset.HG_up, geo_pois_embs)
        trans_structural_users_embs = torch.sparse.mm(dataset.HG_up, trans_pois_embs)

        hg_batch_users_embs = hg_structural_users_embs[user_idx]
        geo_batch_users_embs = geo_structural_users_embs[user_idx]
        trans_batch_users_embs = trans_structural_users_embs[user_idx]

        loss_cl_poi = self.cal_loss_cl_pois(hg_pois_embs, geo_pois_embs, trans_pois_embs)
        loss_cl_user = self.cal_loss_cl_users(
            hg_batch_users_embs, geo_batch_users_embs, trans_batch_users_embs
        )

        norm_hg_pois_embs = F.normalize(hg_pois_embs, p=2, dim=1)
        norm_geo_pois_embs = F.normalize(geo_pois_embs, p=2, dim=1)
        norm_trans_pois_embs = F.normalize(trans_pois_embs, p=2, dim=1)

        norm_all_hg_users_embs = F.normalize(hg_structural_users_embs, p=2, dim=1)
        norm_hg_batch_users_embs = norm_all_hg_users_embs[user_idx]
        norm_geo_batch_users_embs = F.normalize(geo_batch_users_embs, p=2, dim=1)
        norm_trans_batch_users_embs = F.normalize(trans_batch_users_embs, p=2, dim=1)

        # 当前区域 embedding
        region_emb = self.region_embedding(current_region)
        region_emb = F.normalize(region_emb, p=2, dim=1)

        # ===== 新增：区域约束用户邻居增强，只增强 HG 分支 =====
        safe_neighbor_users = neighbor_users.clone()
        safe_neighbor_users[safe_neighbor_users < 0] = 0

        neighbor_hg_embs = norm_all_hg_users_embs[safe_neighbor_users]   # [B, K, d]

        query = self.neighbor_query(
            torch.cat([norm_hg_batch_users_embs, region_emb], dim=-1)
        ).unsqueeze(1)  # [B, 1, d]

        keys = self.neighbor_key(neighbor_hg_embs)        # [B, K, d]
        values = self.neighbor_value(neighbor_hg_embs)    # [B, K, d]

        attn_scores = torch.sum(query * keys, dim=-1) / math.sqrt(self.emb_dim)  # [B, K]
        attn_scores = attn_scores.masked_fill(neighbor_mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)

        neighbor_ctx = torch.sum(attn_weights.unsqueeze(-1) * values, dim=1)  # [B, d]
        has_neighbor = (neighbor_mask.sum(dim=1, keepdim=True) > 0).float()
        neighbor_ctx = has_neighbor * neighbor_ctx
        neighbor_ctx = F.normalize(neighbor_ctx, p=2, dim=1)

        hg_user_final = F.normalize(
            norm_hg_batch_users_embs + self.user_res_beta * neighbor_ctx,
            p=2,
            dim=1
        )

        # ===== 预测分数 =====
        pred_hg = hg_user_final @ norm_hg_pois_embs.T
        pred_geo = norm_geo_batch_users_embs @ norm_geo_pois_embs.T
        pred_trans = norm_trans_batch_users_embs @ norm_trans_pois_embs.T

        # ===== 区域感知 softmax 残差融合 =====
        gate_input = torch.cat([
            hg_user_final,
            norm_geo_batch_users_embs,
            norm_trans_batch_users_embs,
            region_emb
        ], dim=-1)

        mix = torch.softmax(self.region_view_gate(gate_input), dim=-1)

        pred_base = (pred_hg + pred_geo + pred_trans) / 3.0
        pred_region = (
            mix[:, 0:1] * pred_hg +
            mix[:, 1:2] * pred_geo +
            mix[:, 2:3] * pred_trans
        )

        prediction = pred_base + self.mix_beta * (pred_region - pred_base)

        return prediction, loss_cl_poi, loss_cl_user