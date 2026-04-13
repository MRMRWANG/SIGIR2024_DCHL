# coding=utf-8
"""
@author: Yantong Lai
@paper: [24 SIGIR] Disentangled Contrastive Hypergraph Learning for Next POI Recommendation
"""

import math
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import torch
from utils import *
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from region_utils import build_poi2region


def binary_cosine(list_a, list_b):
    set_a, set_b = set(list_a), set(list_b)
    if len(set_a) == 0 or len(set_b) == 0:
        return 0.0
    inter = len(set_a & set_b)
    return inter / math.sqrt(len(set_a) * len(set_b) + 1e-8)


class POIDataset(Dataset):
    def __init__(self, data_filename, pois_coos_filename, num_users, num_pois, padding_idx, args, device):

        # get all sessions and labels
        self.data = load_list_with_pkl(data_filename)  # data = [sessions_dict, labels_dict]
        self.sessions_dict = self.data[0]  # poiID starts from 0
        self.labels_dict = self.data[1]
        self.pois_coos_dict = load_dict_from_pkl(pois_coos_filename)

        # 这里不要再二次 +1，region_utils.py 已经处理好了
        self.poi2region, self.num_regions = build_poi2region(
            self.pois_coos_dict,
            num_bins=args.region_bins
        )
        self.num_regions = int(self.num_regions)

        # definition
        self.num_users = num_users
        self.num_pois = num_pois
        self.padding_idx = padding_idx
        self.distance_threshold = args.distance_threshold
        self.keep_rate = args.keep_rate
        self.device = device
        self.user_topk = args.user_topk

        # get user's trajectory, reversed trajectory and its length
        self.users_trajs_dict, self.users_trajs_lens_dict = get_user_complete_traj(self.sessions_dict)
        self.users_rev_trajs_dict = get_user_reverse_traj(self.users_trajs_dict)
        self.max_user_seq_len = max(self.users_trajs_lens_dict.values()) if len(self.users_trajs_lens_dict) > 0 else 1

        # calculate poi-poi haversine distance and generate geographical adjacency matrix
        self.poi_geo_adj = gen_poi_geo_adj(num_pois, self.pois_coos_dict, self.distance_threshold)   # csr_matrix
        self.poi_geo_graph_matrix = normalized_adj(adj=self.poi_geo_adj, is_symmetric=False)
        self.poi_geo_graph = transform_csr_matrix_to_tensor(self.poi_geo_graph_matrix).to(device)

        # generate poi-session incidence matrix, its degree and hypergraph
        self.H_pu = gen_sparse_H_user(self.sessions_dict, num_pois, self.num_users)    # [L, U]
        self.H_pu = csr_matrix_drop_edge(self.H_pu, args.keep_rate)
        self.Deg_H_pu = get_hyper_deg(self.H_pu)    # [L, L]
        self.HG_pu = self.Deg_H_pu * self.H_pu    # [L, U]
        self.HG_pu = transform_csr_matrix_to_tensor(self.HG_pu).to(device)

        # generate session-poi incidence matrix, its degree and hypergraph
        self.H_up = self.H_pu.T    # [U, L]
        self.Deg_H_up = get_hyper_deg(self.H_up)    # [U, U]
        self.HG_up = self.Deg_H_up * self.H_up    # [U, L]
        self.HG_up = transform_csr_matrix_to_tensor(self.HG_up).to(device)

        # get all sessions for intra-sequential relation learning
        self.all_train_sessions = get_all_users_seqs(self.users_trajs_dict)
        self.pad_all_train_sessions = pad_sequence(self.all_train_sessions, batch_first=True, padding_value=padding_idx)
        self.pad_all_train_sessions = self.pad_all_train_sessions.to(device)    # [U, MAX_SEQ_LEN]
        self.max_session_len = self.pad_all_train_sessions.size(1)

        # generate directed poi-poi hypergraph
        self.H_poi_src = gen_sparse_directed_H_poi(self.users_trajs_dict, num_pois)    # [L, L]
        self.H_poi_src = csr_matrix_drop_edge(self.H_poi_src, args.keep_rate_poi)
        self.Deg_H_poi_src = get_hyper_deg(self.H_poi_src)    # [L, L]
        self.HG_poi_src = self.Deg_H_poi_src * self.H_poi_src    # [L, L]
        self.HG_poi_src = transform_csr_matrix_to_tensor(self.HG_poi_src).to(device)

        # generate targeted poi hypergraph
        self.H_poi_tar = self.H_poi_src.T    # [L, L]
        self.Deg_H_poi_tar = get_hyper_deg(self.H_poi_tar)    # [L, L]
        self.HG_poi_tar = self.Deg_H_poi_tar * self.H_poi_tar    # [L, L]
        self.HG_poi_tar = transform_csr_matrix_to_tensor(self.HG_poi_tar).to(device)

        self.build_region_transition_graph()
        self.build_poi_region_lookup()

        # 新增：构建 区域约束的 TopK 相似用户
        self.build_user_region_neighbors(topk=self.user_topk)

    def build_region_transition_graph(self):
        region_trans = np.zeros((self.num_regions + 1, self.num_regions + 1), dtype=np.float32)
        for _, traj in self.users_trajs_dict.items():
            if len(traj) < 2:
                continue
            for idx in range(len(traj) - 1):
                src_r = int(self.poi2region[int(traj[idx])])
                tar_r = int(self.poi2region[int(traj[idx + 1])])
                region_trans[src_r, tar_r] += 1.0
        region_trans += np.eye(self.num_regions + 1, dtype=np.float32)
        region_trans = sp.csr_matrix(region_trans)
        region_norm = normalized_adj(region_trans, is_symmetric=False)
        self.region_graph = transform_csr_matrix_to_tensor(region_norm).to(self.device)

    def build_poi_region_lookup(self):
        poi_region = np.zeros(self.num_pois, dtype=np.int64)
        for poi in range(self.num_pois):
            poi_region[poi] = int(self.poi2region.get(poi, 0))
        self.poi_region_ids = torch.tensor(poi_region, dtype=torch.long, device=self.device)

    def build_user_region_histories(self):
        self.user_region_pois = [defaultdict(list) for _ in range(self.num_users)]
        self.region2users = defaultdict(set)

        for u in range(self.num_users):
            traj = self.users_trajs_dict[u]
            for p in traj:
                r = int(self.poi2region[int(p)])
                self.user_region_pois[u][r].append(int(p))
                self.region2users[r].add(u)

    def build_user_region_neighbors(self, topk=10):
        self.build_user_region_histories()

        # region 已经是 1 ~ num_regions，0 留给 padding
        self.user_region_neighbors = torch.full(
            (self.num_users, self.num_regions + 1, topk),
            -1,
            dtype=torch.long
        )
        self.user_region_neighbor_mask = torch.zeros(
            (self.num_users, self.num_regions + 1, topk),
            dtype=torch.float
        )

        for r, users_in_r in self.region2users.items():
            users_in_r = list(users_in_r)
            for u in users_in_r:
                pois_u = self.user_region_pois[u][r]
                sims = []
                for v in users_in_r:
                    if v == u:
                        continue
                    pois_v = self.user_region_pois[v][r]
                    sim = binary_cosine(pois_u, pois_v)
                    if sim > 0:
                        sims.append((v, sim))

                sims.sort(key=lambda x: x[1], reverse=True)
                sims = sims[:topk]

                for i, (v, _) in enumerate(sims):
                    self.user_region_neighbors[u, r, i] = v
                    self.user_region_neighbor_mask[u, r, i] = 1.0

    def __len__(self):
        return self.num_users

    def __getitem__(self, user_idx):
        user_seq = self.users_trajs_dict[user_idx]
        user_seq_len = self.users_trajs_lens_dict[user_idx]
        user_seq_mask = [1] * user_seq_len
        user_rev_seq = self.users_rev_trajs_dict[user_idx]
        label = self.labels_dict[user_idx]

        # 当前轨迹最后一个 POI 所在区域
        current_region = int(self.poi2region[int(user_seq[-1])])

        neighbor_users = self.user_region_neighbors[user_idx, current_region]
        neighbor_mask = self.user_region_neighbor_mask[user_idx, current_region]
        last_poi = int(user_seq[-1])

        sample = {
            "user_idx": torch.tensor(user_idx, dtype=torch.long).to(self.device),
            "user_seq": torch.tensor(user_seq, dtype=torch.long).to(self.device),
            "user_rev_seq": torch.tensor(user_rev_seq, dtype=torch.long).to(self.device),
            "user_seq_len": torch.tensor(user_seq_len, dtype=torch.long).to(self.device),
            "user_seq_mask": torch.tensor(user_seq_mask, dtype=torch.long).to(self.device),
            "label": torch.tensor(label, dtype=torch.long).to(self.device),
            "current_region": torch.tensor(current_region, dtype=torch.long).to(self.device),
            "last_poi": torch.tensor(last_poi, dtype=torch.long).to(self.device),
            "neighbor_users": neighbor_users.to(self.device),
            "neighbor_mask": neighbor_mask.to(self.device),
        }

        return sample


class POIPartialDataset(Dataset):
    def __init__(self, full_dataset, user_indices):
        self.data = [full_dataset[i] for i in user_indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class POISessionDataset(Dataset):
    def __init__(self, data_filename, label_filename, pois_coos_filename, num_pois, padding_idx, args, device):
        self.sessions_dict = load_dict_from_pkl(data_filename)
        self.labels_dict = load_dict_from_pkl(label_filename)
        self.pois_coos_dict = load_dict_from_pkl(pois_coos_filename)

        # 这里也不要再二次 +1
        self.poi2region, self.num_regions = build_poi2region(
            self.pois_coos_dict,
            num_bins=args.region_bins
        )
        self.num_regions = int(self.num_regions)

        self.users_trajs_dict = self.sessions_dict
        self.num_pois = num_pois
        self.num_sessions = len(self.sessions_dict)
        self.padding_idx = padding_idx
        self.distance_threshold = args.distance_threshold
        self.keep_rate = args.keep_rate
        self.device = device
        self.user_topk = args.user_topk
        self.max_user_seq_len = max([len(v) for v in self.sessions_dict.values()]) if len(self.sessions_dict) > 0 else 1

        self.poi_geo_adj = gen_poi_geo_adj(num_pois, self.pois_coos_dict, self.distance_threshold)
        self.poi_geo_graph_matrix = normalized_adj(adj=self.poi_geo_adj, is_symmetric=False)
        self.poi_geo_graph = transform_csr_matrix_to_tensor(self.poi_geo_graph_matrix).to(device)

        self.H_poi_src = gen_sparse_directed_H_poi(self.users_trajs_dict, num_pois)
        self.H_poi_src = csr_matrix_drop_edge(self.H_poi_src, args.keep_rate_poi)
        self.Deg_H_poi_src = get_hyper_deg(self.H_poi_src)
        self.HG_poi_src = self.Deg_H_poi_src * self.H_poi_src
        self.HG_poi_src = transform_csr_matrix_to_tensor(self.HG_poi_src).to(device)

        self.H_poi_tar = self.H_poi_src.T
        self.Deg_H_poi_tar = get_hyper_deg(self.H_poi_tar)
        self.HG_poi_tar = self.Deg_H_poi_tar * self.H_poi_tar
        self.HG_poi_tar = transform_csr_matrix_to_tensor(self.HG_poi_tar).to(device)

        self.H_poi_session = gen_sparse_H_pois_session(self.sessions_dict, num_pois, self.num_sessions)
        self.HG_col = gen_HG_from_sparse_H(self.H_poi_session)
        self.HG_col = transform_csr_matrix_to_tensor(self.HG_col).to(device)

        self.H_pu = self.H_poi_session
        self.H_up = self.H_pu.T
        self.Deg_H_up = get_hyper_deg(self.H_up)
        self.HG_up = self.Deg_H_up * self.H_up
        self.HG_up = transform_csr_matrix_to_tensor(self.HG_up).to(device)

        self.build_region_transition_graph()
        self.build_poi_region_lookup()

        self.build_user_region_neighbors(topk=self.user_topk)

    def build_region_transition_graph(self):
        region_trans = np.zeros((self.num_regions + 1, self.num_regions + 1), dtype=np.float32)
        for _, traj in self.users_trajs_dict.items():
            if len(traj) < 2:
                continue
            for idx in range(len(traj) - 1):
                src_r = int(self.poi2region[int(traj[idx])])
                tar_r = int(self.poi2region[int(traj[idx + 1])])
                region_trans[src_r, tar_r] += 1.0
        region_trans += np.eye(self.num_regions + 1, dtype=np.float32)
        region_trans = sp.csr_matrix(region_trans)
        region_norm = normalized_adj(region_trans, is_symmetric=False)
        self.region_graph = transform_csr_matrix_to_tensor(region_norm).to(self.device)

    def build_poi_region_lookup(self):
        poi_region = np.zeros(self.num_pois, dtype=np.int64)
        for poi in range(self.num_pois):
            poi_region[poi] = int(self.poi2region.get(poi, 0))
        self.poi_region_ids = torch.tensor(poi_region, dtype=torch.long, device=self.device)

    def build_user_region_histories(self):
        self.user_region_pois = [defaultdict(list) for _ in range(self.num_sessions)]
        self.region2users = defaultdict(set)

        for u in range(self.num_sessions):
            traj = self.users_trajs_dict[u]
            for p in traj:
                r = int(self.poi2region[int(p)])
                self.user_region_pois[u][r].append(int(p))
                self.region2users[r].add(u)

    def build_user_region_neighbors(self, topk=10):
        self.build_user_region_histories()

        self.user_region_neighbors = torch.full(
            (self.num_sessions, self.num_regions + 1, topk),
            -1,
            dtype=torch.long
        )
        self.user_region_neighbor_mask = torch.zeros(
            (self.num_sessions, self.num_regions + 1, topk),
            dtype=torch.float
        )

        for r, users_in_r in self.region2users.items():
            users_in_r = list(users_in_r)
            for u in users_in_r:
                pois_u = self.user_region_pois[u][r]
                sims = []
                for v in users_in_r:
                    if v == u:
                        continue
                    pois_v = self.user_region_pois[v][r]
                    sim = binary_cosine(pois_u, pois_v)
                    if sim > 0:
                        sims.append((v, sim))

                sims.sort(key=lambda x: x[1], reverse=True)
                sims = sims[:topk]

                for i, (v, _) in enumerate(sims):
                    self.user_region_neighbors[u, r, i] = v
                    self.user_region_neighbor_mask[u, r, i] = 1.0

    def __len__(self):
        return self.num_sessions

    def __getitem__(self, user_idx):
        user_seq = self.users_trajs_dict[user_idx]
        user_seq_len = len(user_seq)
        user_seq_mask = [1] * user_seq_len
        user_rev_seq = user_seq[::-1]
        label = self.labels_dict[user_idx]

        current_region = int(self.poi2region[int(user_seq[-1])])
        neighbor_users = self.user_region_neighbors[user_idx, current_region]
        neighbor_mask = self.user_region_neighbor_mask[user_idx, current_region]
        last_poi = int(user_seq[-1])

        sample = {
            "user_idx": torch.tensor(user_idx, dtype=torch.long).to(self.device),
            "user_seq": torch.tensor(user_seq, dtype=torch.long).to(self.device),
            "user_rev_seq": torch.tensor(user_rev_seq, dtype=torch.long).to(self.device),
            "user_seq_len": torch.tensor(user_seq_len, dtype=torch.long).to(self.device),
            "user_seq_mask": torch.tensor(user_seq_mask, dtype=torch.long).to(self.device),
            "label": torch.tensor(label, dtype=torch.long).to(self.device),
            "current_region": torch.tensor(current_region, dtype=torch.long).to(self.device),
            "last_poi": torch.tensor(last_poi, dtype=torch.long).to(self.device),
            "neighbor_users": neighbor_users.to(self.device),
            "neighbor_mask": neighbor_mask.to(self.device),
        }

        return sample


def collate_fn_4sq(batch, padding_value=3835):
    batch_user_idx = []
    batch_user_seq = []
    batch_user_rev_seq = []
    batch_user_seq_len = []
    batch_user_seq_mask = []
    batch_label = []
    batch_current_region = []
    batch_last_poi = []
    batch_neighbor_users = []
    batch_neighbor_mask = []

    for item in batch:
        batch_user_idx.append(item["user_idx"])
        batch_user_seq_len.append(item["user_seq_len"])
        batch_label.append(item["label"])
        batch_current_region.append(item["current_region"])
        batch_last_poi.append(item["last_poi"])
        batch_neighbor_users.append(item["neighbor_users"])
        batch_neighbor_mask.append(item["neighbor_mask"])
        batch_user_seq.append(item["user_seq"])
        batch_user_rev_seq.append(item["user_rev_seq"])
        batch_user_seq_mask.append(item["user_seq_mask"])

    pad_user_seq = pad_sequence(batch_user_seq, batch_first=True, padding_value=padding_value)
    pad_user_rev_seq = pad_sequence(batch_user_rev_seq, batch_first=True, padding_value=padding_value)
    pad_user_seq_mask = pad_sequence(batch_user_seq_mask, batch_first=True, padding_value=0)

    batch_user_idx = torch.stack(batch_user_idx)
    batch_user_seq_len = torch.stack(batch_user_seq_len)
    batch_label = torch.stack(batch_label)
    batch_current_region = torch.stack(batch_current_region).long()
    batch_last_poi = torch.stack(batch_last_poi).long()
    batch_neighbor_users = torch.stack(batch_neighbor_users).long()
    batch_neighbor_mask = torch.stack(batch_neighbor_mask).float()

    return {
        "user_idx": batch_user_idx,
        "user_seq": pad_user_seq,
        "user_rev_seq": pad_user_rev_seq,
        "user_seq_len": batch_user_seq_len,
        "user_seq_mask": pad_user_seq_mask,
        "label": batch_label,
        "current_region": batch_current_region,
        "last_poi": batch_last_poi,
        "neighbor_users": batch_neighbor_users,
        "neighbor_mask": batch_neighbor_mask,
    }