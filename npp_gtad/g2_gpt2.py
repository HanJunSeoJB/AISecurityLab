#!/usr/bin/env python
# coding: utf-8
"""
Usage
-----
python g2_gpt2.py \
  --train ./train/rx/train.csv \
  --test  ./test/rx/test.csv \
  --model ./best_model.pt \
  --win_size 96 \
  --epochs 100 \
  --patience 10
"""

import os
import math
import sys
import argparse
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.utils import softmax
from torch_geometric.nn import MessagePassing

from anomaly_predict import ad_predict
from merlion.evaluate.anomaly import ScoreType

# ========================
# 0. 익명 컬럼 매핑 (v001 → 실제 태그)
# ========================

ANON_TO_ORIG_TAG = {
    "v001": "bufss[1]",                 # Reactor THERMAL POWER (MWth)
    "v002": "enfms.NFSC_LIN[0]",        # REACTOR POWER (%)
    "v003": "txMPJI003.Output",         # GENERATOR POWER (MWe)
    "v004": "drcs.GPM_CM[1]",           # CEA POSITION(RG 1)
    "v005": "drcs.GPM_CM[2]",           # CEA POSITION(RG 2)
    "v006": "drcs.GPM_CM[3]",           # CEA POSITION(RG 3)
    "v007": "drcs.GPM_CM[4]",           # CEA POSITION(RG 4)
    "v008": "drcs.GPM_CM[5]",           # CEA POSITION(RG 5)
    "v009": "rrsTERR.Output6_r8",       # CEA MOTION DEMAND
    "v010": "rrsTAVG.Output6_r8",       # RCS Tavg
    "v011": "rrsTREF.Output6_r8",       # RCS Tref
    "v012": "txRCTE132A.Output",        # RCS T hot LOOP1
    "v013": "txRCTE133A.Output",        # RCS T hot LOOP2
    "v014": "txRCTE142A.Output",        # RCS T cold 1A
    "v015": "txRCTE142B.Output",        # RCS T cold 1B
    "v016": "txRCTE143A.Output",        # RCS T cold 2A
    "v017": "txRCTE143B.Output",        # RCS T cold 2B
    "v018": "txRCPT102A.Output",        # RCS PRESSURE
    "v019": "iccm.TMARRCS[0]",          # RCS SM TEMP CH-A
    "v020": "iccm.TMARRCS[1]",          # RCS SM TEMP CH-B
    "v021": "mtRCPP01A.RotorSpeed",     # RCP 1A Speed
    "v022": "mtRCPP01B.RotorSpeed",     # RCP 1B Speed
    "v023": "mtRCPP02A.RotorSpeed",     # RCP 2A Speed
    "v024": "mtRCPP02B.RotorSpeed",     # RCP 2B Speed
    "v025": "txRCLT110A.Output",        # PRESSURIZER LEVEL
    "v026": "aovRC100E.avpVpos",        # PZR Spray Valve Position
    "v027": "aovRC100F.avpVpos",        # PZR Spray Valve Position
    "v028": "bkRCHTRP1.state",          # Proportional Heater P1 On/Off
    "v029": "bkRCHTRP1.state",          # Proportional Heater P2 On/Off (원문 동일 표기)
    "v030": "cnRCHTRB1_.output",        # Backup Heater B1 On/Off
    "v031": "cnRCHTRB2_.output",        # Backup Heater B2 On/Off
    "v032": "bkRCHTRB3.state",          # Backup Heater B3 On/Off
    "v033": "bkRCHTRB4.state",          # Backup Heater B4 On/Off
    "v034": "bkRCHTRB5.state",          # Backup Heater B5 On/Off
    "v035": "bkRCHTRB6.state",          # Backup Heater B6 On/Off
    "v036": "txNGII100A.Output",        # PZR PROP HTR P1 CURRENT
    "v037": "txNGII100B.Output",        # PZR PROP HTR P2 CURRENT
    "v038": "txRCHTRB1_A.Output",       # PZR BACKUP HTR B1 CURRENT
    "v039": "txRCHTRB2_A.Output",       # PZR BACKUP HTR B2 CURRENT
    "v040": "txNGII100E.Output",        # PZR BACKUP HTR B3 CURRENT
    "v041": "txNGII100F.Output",        # PZR BACKUP HTR B4 CURRENT
    "v042": "txNGII100G.Output",        # PZR BACKUP HTR B5 CURRENT
    "v043": "txNGII100H.Output",        # PZR BACKUP HTR B6 CURRENT
    "v044": "swRCPS100N04.Limit_Low",   # PZR Pressure Lo Alarm Backup Heater ON Signal
    "v045": "LPP_PRETRIPA.SetPoint",    # PZR Pressure Lo Pretrip
    "v046": "_RLNGTB735.Output_r",      # PZR Pressure Lo trip
    "v047": "RCPY102.Output",           # PPCS-P100-PRV VAL1
    "v048": "crPPCS_PV.Output",         # PPCS-P100-PV VAL1
    "v049": "ctRCPIK100.anvSP_Real",    # PPCS-P100-SP VAL1
    "v050": "RCLY1113.Output",          # PRV of PZR pressure VAL1
    "v051": "ctRCLIK110.ixvInputReal",  # P-100 Process value
    "v052": "ctRCLIK110.anvSP_Real",    # P-100 Setpoint
    "v053": "ctRCHIK100.rmtSpReal",     # Spray Valve SP
    "v054": "txMSPT1013A.Output",       # SG 1 PRESSURE
    "v055": "txMSPT1023A.Output",       # SG 2 PRESSURE
    "v056": "txFWLT1113A.Output",       # SG 1 LEVEL(WR)
    "v057": "txFWLT1123A.Output",       # SG 2 LEVEL(WR)
    "v058": "txFWLT1114A.Output",       # SG 1 LEVEL(NR)
    "v059": "txFWLT1124A.Output",       # SG 2 LEVEL(NR)
    "v060": "crSBCS_SSFS1.output",      # SG 1 STEAM FLOW
    "v061": "crSBCS_SSFS2.output",      # SG 2 STEAM FLOW
    "v062": "txFWFT1112X.Output",       # SG 1 FEED FLOW
    "v063": "txFWFT1122X.Output",       # SG 2 FEED FLOW
    "v064": "txFWFT1113X.Output",       # SG1 Downcomer FW Flow (Ch. X)
    "v065": "txFWFT1113Y.Output",       # SG1 Downcomer FW Flow (Ch. Y)
    "v066": "txFWFT1123X.Output",       # SG2 Downcomer FW Flow (Ch. X)
    "v067": "txFWFT1123Y.Output",       # SG2 Downcomer FW Flow (Ch. Y)
    "v068": "TbnSpeed.Output",          # TURBINE SPEED
    "v069": "txTAPT135.Output",         # Turbine Inlet Pressure
    "v070": "txTAPT143.Output",         # Turbine Outlet Pressure
    "v071": "bkCDPP01.state",           # CONDENSATE PUMP A
    "v072": "bkCDPP02.state",           # CONDENSATE PUMP B
    "v073": "bkCDPP03.state",           # CONDENSATE PUMP C
    "v074": "bkCAPP01.state",           # CONDENSATE VACUUM PUMP 01
    "v075": "bkCAPP02.state",           # CONDENSATE VACUUM PUMP 02
    "v076": "bkCAPP03.state",           # CONDENSATE VACUUM PUMP 03
    "v077": "bkCAPP04.state",           # CONDENSATE VACUUM PUMP 04
    "v078": "CDPI485.output",           # Condenser Pressure
    "v079": "txCDLT021.Output",         # Condenser Level(right side)
    "v080": "txCDLT005.Output",         # Condenser Level(left side)
    "v081": "txCDTE037.Output",         # Condenser Temperature
    "v082": "txFTST3101X.Output",       # FWPT A SPEED
    "v083": "txFTST3102X.Output",       # FWPT B SPEED
    "v084": "txFTST3103X.Output",       # FWPT C SPEED
    "v085": "FWZI1113.output",          # FW 1 Downcomer Control Valve Position
    "v086": "FWZI1123.output",          # FW 2 Downcomer Control Valve Position
    "v087": "FWZI1112.output",          # FW 1 Economizer Control Valve Position
    "v088": "FWZI1122.output",          # FW 2 Economizer Control Valve Position
    "v089": "crFWCS1_FC1111_SP.output", # 9-541-J-FIK-1111-SP
    "v090": "ctFWFC1111.anvCmd100",     # 9-541-J-FIK-1111-OP
    "v091": "crFWCS1_FC1111_PV.output", # 9-541-J-FIK-1111-PV
    "v092": "crFWCS2_FC1121_SP.output", # 9-541-J-FIK-1121-SP
    "v093": "ctFWFC1121.anvCmd100",     # 9-541-J-FIK-1121-OP
    "v094": "crFWCS2_FC1121_PV.output", # 9-541-J-FIK-1121-PV
    "v095": "ctFWHIC1112.anvSP_Real",   # 9-541-J-HIK-1112-SP
    "v096": "ctFWHIC1112.anvCmd100",    # 9-541-J-HIK-1112-OP
    "v097": "ctFWHIC1112.ixvInputReal", # 9-541-J-HIK-1112-PV
    "v098": "ctFWHIC1113.anvSP_Real",   # 9-541-J-HIK-1113-SP
    "v099": "ctFWHIC1113.anvCmd100",    # 9-541-J-HIK-1113-OP
    "v100": "ctFWHIC1113.ixvInputReal", # 9-541-J-HIK-1113-PV
    "v101": "ctFWHIC1122.anvSP_Real",   # 9-541-J-HIK-1122-SP
    "v102": "ctFWHIC1122.anvCmd100",    # 9-541-J-HIK-1122-OP
    "v103": "ctFWHIC1122.ixvInputReal", # 9-541-J-HIK-1122-PV
    "v104": "ctFWHIC1123.anvSP_Real",   # 9-541-J-HIK-1123-SP
    "v105": "ctFWHIC1123.anvCmd100",    # 9-541-J-HIK-1123-OP
    "v106": "ctFWHIC1123.ixvInputReal", # 9-541-J-HIK-1123-PV
    "v107": "txCMPT351A.Output",        # CNMT PRESS(NR) CH-A
    "v108": "txCMPT351B.Output",        # CNMT PRESS(NR) CH-B
    "v109": "txCMPT352A.Output",        # CNMT PRESS(WR) CH-A
    "v110": "txCMPT352B.Output",        # CNMT PRESS(WR) CH-B
    "v111": "txCMLT027A.Output",        # CNMT WTR LVL CH-A
    "v112": "txCMLT028B.Output",        # CNMT WTR LVL CH-B
    "v113": "txCSPT071.Output",         # CSP01A DISH PR
    "v114": "txCSPT081.Output",         # CSP01B DISH PR
    "v115": "txCSFT338C.Output",        # CS HX 01A OUTLET FLOW
    "v116": "txCSFT348D.Output",        # CS HX 01B OUTLET FLOW
    "v117": "txSIPT308.Output",         # SIP02A DISCH PR
    "v118": "txSIPT306.Output",         # SIP02C DISCH PR
    "v119": "txSIPT309.Output",         # SIP02B DISCH PR
    "v120": "txSIPT307.Output",         # SIP02D DISCH PR
    "v121": "txSIFT341A.Output",        # SI FLOW - DVI 1A
    "v122": "txSIFT311D.Output",        # SI FLOW - DVI 1B
    "v123": "txSIFT331C.Output",        # SI FLOW - DVI 2A
    "v124": "txSIFT321B.Output",        # SI FLOW - DVI 2B
    "v125": "txCVFT212B.Output",        # CHARGING FLOW
    "v126": "txCVFT202.Output",         # LETDOWN FLOW
    "v127": "alEF_SIAS.output",         # EFSAS 신호 SIAS On/Off
}

def apply_anon_column_map(df):
    """
    CSV 컬럼이 v001, v002 ... 형식이면 ANON_TO_ORIG_TAG를 이용해
    실제 태그명으로 rename 해 준다.
    - 'time', 'attack' 같은 특수 컬럼은 그대로 둔다.
    - 이미 실제 태그명으로 되어 있으면 아무 변화 없음.
    """
    rename_map = {}
    for c in df.columns:
        if c in ANON_TO_ORIG_TAG:
            rename_map[c] = ANON_TO_ORIG_TAG[c]
    if rename_map:
        print("[INFO] 익명 컬럼을 실제 태그명으로 매핑:", rename_map)
        df = df.rename(columns=rename_map)
    return df


# ===============================
# 1. 모델/데이터 관련 클래스 정의
# ===============================

class EdgeAwareGATConv(MessagePassing):
    """
    Edge-Aware GAT 레이어:
    α_ij = attention(h_i, h_j, e_ij)
    """
    def __init__(self, in_channels, out_channels, edge_dim, heads=1, **kwargs):
        super(EdgeAwareGATConv, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.edge_dim = edge_dim

        self.lin_l = nn.Linear(in_channels, out_channels * heads)
        self.lin_edge = nn.Linear(edge_dim, out_channels * heads)
        self.att = nn.Parameter(torch.Tensor(1, heads, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index, edge_attr):
        # x: (N, Fin) or (B, N, Fin) → 여기서는 (N, W)를 사용
        x_transformed = self.lin_l(x)                # (N, heads*out)
        edge_attr_transformed = self.lin_edge(edge_attr)  # (E, heads*out)
        return self.propagate(edge_index, x=x_transformed,
                              edge_attr=edge_attr_transformed)

    def message(self, x_j, x_i, edge_attr, index, ptr, size_i):
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)
        edge_attr = edge_attr.view(-1, self.heads, self.out_channels)

        alpha = F.leaky_relu(x_i + x_j + edge_attr)
        alpha = (alpha * self.att).sum(dim=-1)   # (E, heads)
        alpha = softmax(alpha, index, ptr, size_i)  # (E, heads)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        # (N, heads, out) → (N, heads*out)
        return aggr_out.view(-1, self.out_channels * self.heads)


class CustomGATModel_EdgeAware(nn.Module):
    """
    Edge-Aware GAT + TCN 기반 이진 분류 모델
    """
    def __init__(self, num_features, window_size,
                 tcn_out_channels, gat_out_dim,
                 edge_feature_dim, heads=4, dropout=0.5):
        super(CustomGATModel_EdgeAware, self).__init__()

        # 1) 시간축 TCN
        self.tcn = nn.Conv1d(
            in_channels=num_features,
            out_channels=tcn_out_channels,
            kernel_size=3,
            padding=1
        )

        # 2) GAT (Edge-aware)
        self.gat1 = EdgeAwareGATConv(
            in_channels=window_size,
            out_channels=gat_out_dim,
            edge_dim=edge_feature_dim,
            heads=heads
        )
        self.elu = nn.ELU()

        # 3) Classification head
        gat_output_flat_dim = num_features * gat_out_dim * heads
        self.classification_head = nn.Sequential(
            nn.Linear(gat_output_flat_dim, gat_output_flat_dim // 4),
            nn.BatchNorm1d(gat_output_flat_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(gat_output_flat_dim // 4, 2),
        )

    def forward(self, x, edge_index, edge_attr):
        # x: (B, W, N), edge_index: (2, E), edge_attr: (B, E, D_edge)
        # 1) TCN
        x_t = x.permute(0, 2, 1)           # (B, N, W)
        x_tcn = self.tcn(x_t)              # (B, C_out=N, W)
        x_tcn = x_tcn.permute(0, 2, 1)     # (B, W, N)

        # 2) GAT (배치별 loop)
        x_g = x_tcn.permute(0, 2, 1)       # (B, N, W)
        batch_outs = []
        for i in range(x_g.shape[0]):
            x_sample = x_g[i]                  # (N, W)
            attr_sample = edge_attr[i]         # (E, D_edge)
            gat_out = self.gat1(x_sample, edge_index, edge_attr=attr_sample)
            gat_out = self.elu(gat_out)
            batch_outs.append(gat_out)
        x_gat = torch.stack(batch_outs, dim=0)  # (B, N, gat_out_dim*heads)

        # 3) 분류
        x_flat = x_gat.reshape(x_gat.size(0), -1)
        logits = self.classification_head(x_flat)
        return logits


# ======================
# 2. 데이터 유틸 함수들
# ======================

def normalize(df, tag_min, tag_max):
    ndf = df.copy()
    for c in df.columns:
        if tag_min[c] == tag_max[c]:
            ndf[c] = df[c] - tag_min[c]
        else:
            ndf[c] = (df[c] - tag_min[c]) / (tag_max[c] - tag_min[c])
    return ndf


def create_fixed_edge_index(all_columns, column_groups):
    """
    그룹 정보(column_groups)를 바탕으로 고정 edge_index 생성.
    같은 그룹 내 노드들끼리 fully-connected edge.
    """
    col_to_idx = {col: i for i, col in enumerate(all_columns)}
    edge_list = []

    for group in column_groups:
        valid_group_cols = [col for col in group if col in col_to_idx]
        for c1, c2 in combinations(valid_group_cols, 2):
            i1, i2 = col_to_idx[c1], col_to_idx[c2]
            edge_list.append([i1, i2])
            edge_list.append([i2, i1])

    if not edge_list:
        return torch.empty((2, 0), dtype=torch.long)

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    print(f"[edge_index] 총 {edge_index.shape[1]}개의 엣지 생성")
    return edge_index


class WeightDataset(Dataset):
    """
    sliding window + edge feature(pearson + last-step diff) Dataset
    df: 'attack' 컬럼 포함되어야 함
    """
    def __init__(self, df, edge_index, base_pearson_weights, win_size=96):
        self.labels = df['attack'].values
        self.features = df.drop('attack', axis=1).values
        self.edge_index = edge_index
        self.base_pearson_weights = base_pearson_weights
        self.win_size = win_size

    def __len__(self):
        return len(self.features) - self.win_size + 1

    def __getitem__(self, index):
        x_window = self.features[index:index + self.win_size]
        label = self.labels[index + self.win_size - 1]

        last_time_step = x_window[-1, :]
        diff = last_time_step[self.edge_index[0]] - last_time_step[self.edge_index[1]]
        dynamic_diff_weights = torch.from_numpy(np.abs(diff)).float()

        static_feat = self.base_pearson_weights.unsqueeze(1)  # (E, 1)
        dynamic_feat = dynamic_diff_weights.unsqueeze(1)      # (E, 1)
        final_edge_attr = torch.cat([static_feat, dynamic_feat], dim=1)  # (E, 2)

        return (
            torch.from_numpy(x_window).float(),   # (W, N)
            torch.tensor(label).long(),           # label
            self.edge_index,                      # (2, E)
            final_edge_attr                       # (E, 2)
        )


def custom_loss_fn(pred, label):
    assert pred.ndim == 2
    if label.ndim > 1:
        label = label.view(-1)
    assert pred.size(0) == label.size(0)
    return nn.CrossEntropyLoss()(pred, label.long())


# ==================
# 3. Trainer 클래스
# ==================

ad_config_default = {
    'threshold_determine': 'floating',
    'detect_nu': 0.002,
    'lr': 3e-4,
    'weight_decay': 1e-5,
}


class EarlyStopping:
    def __init__(self, path='best_model.pt', patience=5, delta=0.0):
        self.path = path
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_indicator = -np.inf
        self.best_model_state = None
        self.best_val_scores = None
        self.best_test_scores = None
        self.early_stop = False

    def __call__(self, indicator, model, val_scores, test_scores):
        is_best = indicator > self.best_indicator + self.delta
        if is_best:
            self.best_indicator = indicator
            self.best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            self.best_val_scores = val_scores
            self.best_test_scores = test_scores
            self.counter = 0
            torch.save(self.best_model_state, self.path)
            print(f"[EarlyStopping] 새 best 모델 저장: {self.path}, indicator={indicator:.4f}")
        else:
            self.counter += 1
            print(f"[EarlyStopping] counter {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


class GAT_Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader,
                 config, device='cuda', path='best_model.pt'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('lr', 3e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=3
        )
        self.path = path
        self.config = config
        print(f"[Trainer] device = {self.device}")

    def _train_epoch(self):
        self.model.train()
        total_loss = 0.0
        for x_batch, label_batch, edge_index_batch, attr_batch in self.train_loader:
            x_batch = x_batch.to(self.device)
            labels = label_batch.to(self.device).long()
            edge_index = edge_index_batch[0].to(self.device)
            attr_batch = attr_batch.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x_batch, edge_index, attr_batch)
            loss = custom_loss_fn(logits, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * labels.size(0)
        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def _evaluate_epoch(self, loader):
        self.model.eval()
        total_loss = 0.0
        all_labels, all_scores = [], []

        for x_batch, label_batch, edge_index_batch, attr_batch in loader:
            x_batch = x_batch.to(self.device)
            labels = label_batch.to(self.device).long()
            edge_index = edge_index_batch[0].to(self.device)
            attr_batch = attr_batch.to(self.device)

            logits = self.model(x_batch, edge_index, attr_batch)
            loss = custom_loss_fn(logits, labels)
            total_loss += loss.item() * labels.size(0)

            scores = torch.softmax(logits, dim=1)[:, 1]
            all_labels.append(labels.cpu().numpy())
            all_scores.append(scores.cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)
        labels = np.concatenate(all_labels).flatten()
        scores = np.concatenate(all_scores).flatten()
        return avg_loss, labels, scores

    def _print_metrics(self, prefix, affiliation, rpa_score, pa_score, pw_score):
        aff_p, aff_r = affiliation.get('precision', 0), affiliation.get('recall', 0)
        aff_f1 = 2 * (aff_p * aff_r) / (aff_p + aff_r) if (aff_p + aff_r) > 0 else 0.0
        print(f"[{prefix} Affiliation] P={aff_p:2.4f} R={aff_r:2.4f} F1={aff_f1:2.4f}")

        rpa_f1 = rpa_score.f1(ScoreType.RevisedPointAdjusted)
        rpa_p = rpa_score.precision(ScoreType.RevisedPointAdjusted)
        rpa_r = rpa_score.recall(ScoreType.RevisedPointAdjusted)
        print(f"[{prefix} RPA]         F1={rpa_f1:2.4f} P={rpa_p:2.4f} R={rpa_r:2.4f}")

        pa_f1 = pa_score.f1(ScoreType.PointAdjusted)
        pa_p = pa_score.precision(ScoreType.PointAdjusted)
        pa_r = pa_score.recall(ScoreType.PointAdjusted)
        print(f"[{prefix} PA]          F1={pa_f1:2.4f} P={pa_p:2.4f} R={pa_r:2.4f}")

        pw_f1 = pw_score.f1(ScoreType.Pointwise)
        pw_p = pw_score.precision(ScoreType.Pointwise)
        pw_r = pw_score.recall(ScoreType.Pointwise)
        print(f"[{prefix} Pointwise]   F1={pw_f1:2.4f} P={pw_p:2.4f} R={pw_r:2.4f}")

    def _report_final_metrics(self, val_labels, val_scores, test_labels, test_scores):
        print("\n[Validation] best model 결과")
        val_aff, val_rpa, val_pa, val_pw, _ = ad_predict(
            val_labels, val_scores,
            self.config['threshold_determine'],
            self.config['detect_nu']
        )
        self._print_metrics("Val", val_aff, val_rpa, val_pa, val_pw)

        print("\n[Test] best model 결과")
        test_aff, test_rpa, test_pa, test_pw, _ = ad_predict(
            test_labels, test_scores,
            self.config['threshold_determine'],
            self.config['detect_nu']
        )
        self._print_metrics("Test", test_aff, test_rpa, test_pa, test_pw)

        return {
            "validation": {"affiliation": val_aff, "rpa": val_rpa,
                           "pa": val_pa, "pw": val_pw},
            "test": {"affiliation": test_aff, "rpa": test_rpa,
                     "pa": test_pa, "pw": test_pw},
        }

    def train(self, num_epochs=20, patience=5):
        early_stopping = EarlyStopping(path=self.path, patience=patience)
        train_hist, val_hist = [], []

        for epoch in range(1, num_epochs + 1):
            train_loss = self._train_epoch()
            val_loss, val_labels, val_scores = self._evaluate_epoch(self.val_loader)
            _, _, test_scores = self._evaluate_epoch(self.test_loader)
            self.scheduler.step(val_loss)

            train_hist.append(train_loss)
            val_hist.append(val_loss)

            val_aff, val_rpa, _, _, _ = ad_predict(
                val_labels, val_scores,
                self.config['threshold_determine'],
                self.config['detect_nu']
            )
            val_f1 = val_rpa.f1(ScoreType.RevisedPointAdjusted)
            print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} val_RPA_F1={val_f1:.4f}")

            early_stopping(val_f1, self.model, val_scores, test_scores)
            if early_stopping.early_stop:
                print(f"[EarlyStopping] epoch {epoch}에서 학습 중단")
                break

        print("\n=== Training Done ===")
        print(f"Best validation RPA F1 = {early_stopping.best_indicator:.4f}")

        # best 모델 기준으로 최종 metric
        best_val_scores = early_stopping.best_val_scores
        best_test_scores = early_stopping.best_test_scores
        _, val_labels, _ = self._evaluate_epoch(self.val_loader)
        _, test_labels, _ = self._evaluate_epoch(self.test_loader)
        final_results = self._report_final_metrics(
            val_labels, best_val_scores,
            test_labels, best_test_scores
        )
        return train_hist, val_hist, final_results, test_labels, best_test_scores

    def test(self):
        print(f"\n[TEST] {self.path}에서 best 모델 로드")
        if not os.path.exists(self.path):
            print(f"모델 파일이 없음: {self.path}")
            return None, None, None

        state = torch.load(self.path, map_location=self.device)
        self.model.load_state_dict(state)

        _, val_labels, val_scores = self._evaluate_epoch(self.val_loader)
        _, test_labels, test_scores = self._evaluate_epoch(self.test_loader)
        final_results = self._report_final_metrics(
            val_labels, val_scores, test_labels, test_scores
        )
        return final_results, test_labels, test_scores


# ==================
# 4. main / argparse
# ==================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="훈련 CSV 경로 (index_col=0)")
    p.add_argument("--test", required=True, help="테스트 CSV 경로 (index_col=0)")
    p.add_argument("--model", required=True, help="모델 저장 경로 (.pt)")
    p.add_argument("--win_size", type=int, default=96, help="슬라이딩 윈도우 크기")
    p.add_argument("--epochs", type=int, default=100, help="최대 학습 epoch 수")
    p.add_argument("--patience", type=int, default=10, help="조기 종료 patience")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[ARGS] {args}")

    # 1) CSV 로드
    train_orig = pd.read_csv(args.train, index_col=0)
    test_orig = pd.read_csv(args.test, index_col=0)

    # 익명 컬럼(v001 ~ v127)을 실제 태그명으로 매핑
    train_orig = apply_anon_column_map(train_orig)
    test_orig = apply_anon_column_map(test_orig)

    
    train_orig = train_orig[:10000]
    test_orig = test_orig[:10000]

    # 'time' 컬럼 제외(있으면)
    not_valid_field = {"time"}
    valid_columns = [c for c in test_orig.columns if c not in not_valid_field]

    # 2) 정규화 범위 계산 (train+test)
    tag_min_train = train_orig[valid_columns].min()
    tag_max_train = train_orig[valid_columns].max()
    tag_min_test = test_orig[valid_columns].min()
    tag_max_test = test_orig[valid_columns].max()
    tag_min = np.minimum(tag_min_train, tag_min_test)
    tag_max = np.maximum(tag_max_train, tag_max_test)

    # 3) 정규화
    train = normalize(train_orig[valid_columns], tag_min, tag_max)
    test = normalize(test_orig[valid_columns], tag_min, tag_max)
    print(f"[Data] train shape={train.shape}, test shape={test.shape}")

    # 4) attack 라벨 생성 (앞 절반=0, 뒤 절반=1)
    labels_train = np.zeros(len(train), dtype=int)
    labels_test = np.zeros(len(test), dtype=int)
    labels_train[len(train)//2:] = 1
    labels_test[len(test)//2:] = 1

    train_labels = pd.DataFrame(labels_train, columns=["attack"], index=train.index)
    test_labels = pd.DataFrame(labels_test, columns=["attack"], index=test.index)
    train = pd.concat([train, train_labels], axis=1)
    test = pd.concat([test, test_labels], axis=1)

    # 5) edge_index / base_pearson_weights
    #    - 여기서는 "모든 feature를 한 그룹"으로 fully-connected edge 생성
    column_groups = [valid_columns]
    fixed_edge_index = create_fixed_edge_index(valid_columns, column_groups)

    base_adj = train[valid_columns].corr(method='pearson').abs().fillna(0).values
    base_pearson_weights = torch.from_numpy(
        base_adj[fixed_edge_index[0], fixed_edge_index[1]]
    ).float()

    # 6) Dataset / Dataloader
    win_size = args.win_size
    train_dataset_full = WeightDataset(
        df=train, edge_index=fixed_edge_index,
        base_pearson_weights=base_pearson_weights,
        win_size=win_size
    )
    test_dataset = WeightDataset(
        df=test, edge_index=fixed_edge_index,
        base_pearson_weights=base_pearson_weights,
        win_size=win_size
    )

    val_ratio = 0.2
    val_len = int(len(train_dataset_full) * val_ratio)
    train_len = len(train_dataset_full) - val_len
    train_dataset, val_dataset = random_split(train_dataset_full, [train_len, val_len])
    print(f"[Dataset] train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 7) 모델 생성
    num_features = len(valid_columns)
    window_size = win_size
    edge_feature_dim = 2

    model = CustomGATModel_EdgeAware(
        num_features=num_features,
        window_size=window_size,
        tcn_out_channels=num_features,
        gat_out_dim=16,
        edge_feature_dim=edge_feature_dim,
        heads=4
    )

    # 8) Trainer
    cfg = ad_config_default.copy()
    trainer = GAT_Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=cfg,
        path=args.model
    )

    # 9) 학습
    train_hist, val_hist, final_results, test_labels_arr, best_test_scores = trainer.train(
        num_epochs=args.epochs,
        patience=args.patience
    )

    # 10) test_score와 attack_label 시각화
    try:
        # numpy 배열로 정리
        scores = np.asarray(best_test_scores).flatten()
        labels = np.asarray(test_labels_arr).flatten()

        # 길이 맞추기 (혹시라도 다를 경우를 대비)
        min_len = min(len(scores), len(labels))
        scores = scores[:min_len]
        labels = labels[:min_len]

        x = np.arange(min_len)

        plt.figure(figsize=(12, 4))

        # 이상탐지 score 라인
        plt.plot(x, scores, label="Anomaly score")

        # attack label (1 구간을 붉은색으로 채우기)
        # labels 가 0/1 이므로, 1인 구간이 빨간색으로 채워짐
        plt.fill_between(x, 0, labels, color="red", alpha=0.3, label="Attack label (1)")

        plt.xlabel("Time index")
        plt.ylabel("Score / Label")
        plt.title(f"NPP-GTAD Anomaly Detection")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig("test_score_attack_label.png", dpi=150)
        plt.close()
        print("[Plot] test_score_attack_label.png 저장 완료")
    except Exception as e:
        print(f"[Plot] test score 그래프 생성 실패: {e}")

    print("\n[Done] 학습 완료 및 best 모델 저장:", args.model)

if __name__ == "__main__":
    main()
