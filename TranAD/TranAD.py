#!/usr/bin/env python
# coding: utf-8

import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import TransformerEncoder, TransformerDecoder
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    ndcg_score,   # ndcg 사용을 위해 추가
)
import matplotlib.pyplot as plt

from src.spot import SPOT
from src.pot import pot_eval

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


# ============================================================
# 1. 모델 정의 (TranAD)
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model).float()
            * (-math.log(10000.0) / d_model)
        )
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        """
        x: (seq_len, batch, d_model)
        """
        x = x + self.pe[pos:pos + x.size(0), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        # src: (seq_len, batch, d_model)
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        **kwargs
    ):
        # tgt: (seq_len, batch, d_model)
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class TranAD(nn.Module):
    def __init__(self, feats, window=96, lr=0.001, batch=128):
        super(TranAD, self).__init__()
        self.name = 'TranAD'
        self.lr = lr
        self.batch = batch
        self.n_feats = feats
        self.n_window = window
        self.n = self.n_feats * self.n_window

        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)

        encoder_layers = TransformerEncoderLayer(
            d_model=2 * feats,
            nhead=feats,
            dim_feedforward=16,
            dropout=0.1,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)

        decoder_layers1 = TransformerDecoderLayer(
            d_model=2 * feats,
            nhead=feats,
            dim_feedforward=16,
            dropout=0.1,
        )
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)

        decoder_layers2 = TransformerDecoderLayer(
            d_model=2 * feats,
            nhead=feats,
            dim_feedforward=16,
            dropout=0.1,
        )
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)

        self.fcn = nn.Sequential(
            nn.Linear(2 * feats, feats),
            nn.Sigmoid()
        )

    def encode(self, src, c, tgt):
        # src, c: (seq_len, batch, feats)
        src = torch.cat((src, c), dim=2)  # (seq_len, batch, 2*feats)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)  # (1, batch, 2*feats)
        return tgt, memory

    def forward(self, src, tgt):
        # Phase 1
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2


# ============================================================
# 2. 데이터 전처리 및 유틸 함수
# ============================================================

def dataframe_from_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path).rename(columns=lambda x: x.strip())


def normalize(df, tag_min, tag_max):
    ndf = df.copy()
    for c in df.columns:
        if tag_min[c] == tag_max[c]:
            ndf[c] = df[c] - tag_min[c]
        else:
            ndf[c] = (df[c] - tag_min[c]) / (tag_max[c] - tag_min[c])
    return ndf


def boundary_check(df):
    x = np.array(df, dtype=np.float32)
    return np.any(x > 1.0), np.any(x < 0), np.any(np.isnan(x))


def convert_to_windows(data: torch.Tensor, w_size: int = 96) -> torch.Tensor:
    """
    data: (T, F) tensor
    return: (T, w_size, F) tensor
    """
    windows = []
    for i, g in enumerate(data):
        if i >= w_size:
            w = data[i - w_size:i]
        else:
            w = torch.cat([data[0].repeat(w_size - i, 1), data[0:i]])
        windows.append(w)
    return torch.stack(windows)


# ============================================================
# 3. 학습 / 추론 함수 (backprop_modify)
# ============================================================

def backprop_modify(
    epoch,
    model,
    data,
    dataO,
    optimizer,
    scheduler,
    training: bool = True,
    eval_bs: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    """
    training=True  : 학습 (average training loss, lr 반환)
    training=False : 평가 (loss matrix, prediction matrix 반환)
    """
    feats = dataO.shape[1]
    l = nn.MSELoss(reduction='none')

    data_x = torch.tensor(data, dtype=dtype)
    dataset = TensorDataset(data_x, data_x)

    if device is None:
        device = next(model.parameters()).device

    if training:
        bs = getattr(model, 'batch', 256)
    else:
        bs = eval_bs or getattr(model, 'batch', 256)
        bs = max(1, min(bs, len(data)))

    dataloader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    n = epoch + 1
    l1s = []

    if training:
        model.train()
        for d, _ in dataloader:
            d = d.to(device, dtype=dtype, non_blocking=pin_memory)  # (B, W, F)
            local_bs = d.shape[0]

            window = d.permute(1, 0, 2)  # (W, B, F)
            elem = window[-1, :, :].view(1, local_bs, feats)

            z = model(window, elem)
            l1 = (
                l(z, elem) if not isinstance(z, tuple)
                else (1 / n) * l(z[0], elem) + (1 - 1 / n) * l(z[1], elem)
            )
            if isinstance(z, tuple):
                z = z[1]

            l1s.append(torch.mean(l1).item())
            loss = torch.mean(l1)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        tqdm.write(f'Epoch {epoch}, L1 = {np.mean(l1s):.6f}')
        return float(np.mean(l1s)), optimizer.param_groups[0]['lr']

    else:
        model.eval()
        all_losses = []
        all_preds = []
        with torch.no_grad():
            for d, _ in dataloader:
                d = d.to(device, dtype=dtype, non_blocking=pin_memory)
                local_bs = d.shape[0]

                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, local_bs, feats)

                z = model(window, elem)
                if isinstance(z, tuple):
                    z = z[1]

                loss_mat = l(z, elem)[0].cpu()  # (B, F)
                pred_mat = z[0].cpu()           # (B, F)

                all_losses.append(loss_mat)
                all_preds.append(pred_mat)

        loss_full = torch.cat(all_losses, dim=0).numpy()  # (N, F)
        pred_full = torch.cat(all_preds, dim=0).numpy()   # (N, F)
        return loss_full, pred_full


# ============================================================
# 4. 기타 유틸 (플롯, 모델 저장, hit/ndcg)
# ============================================================

def plot_accuracies(accuracy_list, name: str):
    os.makedirs(f'plots/{name}/', exist_ok=True)
    trainAcc = [i[0] for i in accuracy_list]
    lrs = [i[1] for i in accuracy_list]
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Average Training Loss')
    plt.plot(range(len(trainAcc)), trainAcc,
             label='Average Training Loss',
             linewidth=1, linestyle='-', marker='.')
    plt.twinx()
    plt.plot(range(len(lrs)), lrs,
             label='Learning Rate',
             linewidth=1, linestyle='--', marker='.')
    plt.savefig(f'plots/{name}/training-graph.pdf')
    plt.close()


def save_model(model, optimizer, scheduler, epoch, accuracy_list, name: str):
    folder = f'./model/{name}/'
    os.makedirs(folder, exist_ok=True)
    file_path = f'{folder}/{name}.ckpt'
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'accuracy_list': accuracy_list,
        },
        file_path,
    )
    return file_path


def hit_att(ascore, labels, ps=[100, 150]):
    res = {}
    for p in ps:
        hit_score = []
        for i in range(ascore.shape[0]):
            a, l = ascore[i], labels[i]
            a_sorted = np.argsort(a).tolist()[::-1]
            l_idx = set(np.where(l == 1)[0])
            if l_idx:
                size = round(p * len(l_idx) / 100)
                a_p = set(a_sorted[:size])
                intersect = a_p.intersection(l_idx)
                hit = len(intersect) / len(l_idx)
                hit_score.append(hit)
        res[f'Hit@{p}%'] = np.mean(hit_score) if hit_score else 0.0
    return res


def ndcg(ascore, labels, ps=[100, 150]):
    res = {}
    for p in ps:
        ndcg_scores = []
        for i in range(ascore.shape[0]):
            a, l = ascore[i], labels[i]
            labs = list(np.where(l == 1)[0])
            if labs:
                k_p = round(p * len(labs) / 100)
                try:
                    hit = ndcg_score(
                        l.reshape(1, -1),
                        a.reshape(1, -1),
                        k=k_p
                    )
                except Exception:
                    continue
                ndcg_scores.append(hit)
        res[f'NDCG@{p}%'] = np.mean(ndcg_scores) if ndcg_scores else 0.0
    return res


# ============================================================
# 5. main: CLI에서 학습 + 테스트 실행
# ============================================================

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    # -----------------------------
    # 5.1 데이터 로드
    # -----------------------------
    train_raw = dataframe_from_csv(args.train)
    test_raw = dataframe_from_csv(args.test)

    # 필요하면 앞부분만 사용
    if args.max_len is not None and args.max_len > 0:
        train_raw = train_raw.iloc[:args.max_len]
        test_raw = test_raw.iloc[:args.max_len]

    # 익명 컬럼 매핑을 쓰려면 여기에서 적용
    # from g2_gpt2.py import ANON_TO_ORIG_TAG, apply_anon_column_map 처럼
    # 동일 블록을 복사해서 사용하면 됨.
    # ---------------------------------------------------
    # TODO: 필요하면 여기서 익명 컬럼 매핑 적용
    train_raw = apply_anon_column_map(train_raw)
    test_raw  = apply_anon_column_map(test_raw)
    # ---------------------------------------------------

    # time 컬럼 제외
    not_valid_field = {"time"}
    valid_cols = [c for c in test_raw.columns if c not in not_valid_field]

    train_feat = train_raw[valid_cols]
    test_feat = test_raw[valid_cols]

    # -----------------------------
    # 5.2 정규화 (train+test 공통 범위)
    # -----------------------------
    tag_min_train = train_feat.min()
    tag_max_train = train_feat.max()
    tag_min_test = test_feat.min()
    tag_max_test = test_feat.max()

    tag_min = np.minimum(tag_min_train, tag_min_test)
    tag_max = np.maximum(tag_max_train, tag_max_test)

    train_df = normalize(train_feat, tag_min, tag_max)
    test_df = normalize(test_feat, tag_min, tag_max)

    print("[INFO] Boundary check (train, test):")
    print("  train:", boundary_check(train_df))
    print("  test :", boundary_check(test_df))

    # -----------------------------
    # 5.3 텐서 & 윈도우 변환
    # -----------------------------
    w_size = args.win_size

    trainO = torch.from_numpy(train_df.to_numpy())
    trainD = convert_to_windows(trainO, w_size=w_size)

    testO = torch.from_numpy(test_df.to_numpy())
    testD = convert_to_windows(testO, w_size=w_size)

    print("[INFO] trainD / trainO shape:", trainD.shape, trainO.shape)
    print("[INFO] testD / testO shape :", testD.shape, testO.shape)

    # -----------------------------
    # 5.4 모델/옵티마이저 설정
    # -----------------------------
    model = TranAD(trainD.shape[-1], window=w_size).to(device).float()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model.lr,
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)

    # -----------------------------
    # 5.5 학습 (train mode일 때)
    # -----------------------------
    accuracy_list = []
    if args.mode in ("train", "train_test"):
        print("[INFO] Training...")
        for e in tqdm(range(args.epochs)):
            lossT, lr = backprop_modify(
                e,
                model,
                trainD,
                trainO,
                optimizer,
                scheduler,
                training=True,
                device=device,
            )
            accuracy_list.append((lossT, lr))

        plot_accuracies(accuracy_list, args.name)
        ckpt_path = save_model(
            model,
            optimizer,
            scheduler,
            args.epochs - 1,
            accuracy_list,
            args.name,
        )
        print(f"[INFO] Model saved to: {ckpt_path}")

    # -----------------------------
    # 5.6 테스트 (항상 수행 또는 mode=='test')
    # -----------------------------
    if args.mode in ("test", "train_test"):
        print("[INFO] Testing...")

        # 학습 데이터의 reconstruction error 분포
        lossT, _ = backprop_modify(
            0,
            model,
            trainD,
            trainO,
            optimizer,
            scheduler,
            training=False,
            device=device,
        )

        # 테스트 데이터의 reconstruction error
        loss, y_pred = backprop_modify(
            0,
            model,
            testD,
            testO,
            optimizer,
            scheduler,
            training=False,
            device=device,
        )

        print("[INFO] loss shape:", loss.shape, " y_pred shape:", y_pred.shape)

        # -------------------------
        # 5.6.1 라벨 생성 (간이)
        #    - 0 ~ anomaly_start-1 : 0
        #    - anomaly_start ~ 끝 : 1
        # -------------------------
        labels = np.zeros(test_df.shape[0], dtype=int)
        if 0 <= args.anomaly_start < len(labels):
            labels[args.anomaly_start:] = 1
        labels = np.tile(labels, (loss.shape[1], 1)).T  # (N, F)

        # -------------------------
        # 5.6.2 POT 기반 임계값/성능 평가
        # -------------------------
        lossTfinal = np.mean(lossT, axis=1)
        lossFinal = np.mean(loss, axis=1)
        labelsFinal = (np.sum(labels, axis=1) >= 1) + 0

        result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
        result.update(hit_att(loss, labels))
        result.update(ndcg(loss, labels))

        print("[RESULT]")
        for k, v in result.items():
            print(f"  {k}: {v}")

        # -------------------------
        # 5.6.3 스코어 플롯
        # -------------------------
        plt.figure()
        plt.plot(lossFinal, label='Anomaly Score')
        plt.title(f"{args.name} Scenario Anomaly Detection TranAD")
        plt.xlabel('Time index')
        plt.ylabel('Anomaly Score')
        plt.plot(labelsFinal * 0.1, color='red', label='Label')
        plt.legend()
        os.makedirs('plots', exist_ok=True)
        plt.savefig(f'plots/{args.name}_score_plot.png')
        plt.close()
        print(f"[INFO] Score plot saved to plots/{args.name}_score_plot.png")


# ============================================================
# 6. CLI entry
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TranAD anomaly detection (CLI 버전)"
    )
    parser.add_argument(
        "--train",
        required=True,
        help="학습용 CSV 경로",
    )
    parser.add_argument(
        "--test",
        required=True,
        help="테스트용 CSV 경로",
    )
    parser.add_argument(
        "--name",
        default="TranAD",
        help="실험 이름 / 모델/플롯 저장 폴더 이름",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="학습 epoch 수",
    )
    parser.add_argument(
        "--win_size",
        type=int,
        default=96,
        help="슬라이딩 윈도우 길이",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=12000,
        help="train/test 둘 다 앞에서부터 사용할 최대 길이 (0이면 전체)",
    )
    parser.add_argument(
        "--anomaly_start",
        type=int,
        default=6000,
        help="이 인덱스부터 라벨 1(이상)로 취급",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "test", "train_test"],
        default="train_test",
        help="train만 / test만 / train 후 test",
    )

    args = parser.parse_args()
    if args.max_len <= 0:
        args.max_len = None

    main(args)
