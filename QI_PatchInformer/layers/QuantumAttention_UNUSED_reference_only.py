# NOT USED IN TRAINING. The live implementation is `_QuantumInspiredAttention` inside PatchTST_backbone.py. This file is kept for reference only and is not imported anywhere.

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import numpy as np
from utils.masking import TriangularCausalMask

class QuantumInspiredAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(QuantumInspiredAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        # Input shapes:
        # queries: [B, L, H, E]
        # keys:    [B, S, H, E]
        # values:  [B, S, H, D]
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        assert E % 2 == 0, f"Query dimension E={E} must be even to split into amplitude and phase."
        
        d_k = E // 2
        
        # Split queries and keys into amplitude and phase components along the last dimension
        q_amp, q_phase = queries[..., :d_k], queries[..., d_k:]
        k_amp, k_phase = keys[..., :d_k], keys[..., d_k:]
        
        # 1. Normalize Amplitudes to be unit vectors (L2 normalization)
        eps = 1e-8
        q_amp_norm = q_amp / (torch.norm(q_amp, p=2, dim=-1, keepdim=True) + eps)
        k_amp_norm = k_amp / (torch.norm(k_amp, p=2, dim=-1, keepdim=True) + eps)
        
        # 2. Map Phases to [-pi, pi] using tanh
        q_theta = torch.pi * torch.tanh(q_phase)
        k_theta = torch.pi * torch.tanh(k_phase)
        
        # 3. Compute overlap probability |<psi_q | psi_k>|^2
        # Real[i, j] = \sum_d a_q_id * a_k_jd * cos(theta_q_id - theta_k_jd)
        # Imag[i, j] = \sum_d a_q_id * a_k_jd * sin(theta_q_id - theta_k_jd)
        # score[i, j] = Real[i, j]^2 + Imag[i, j]^2
        
        cos_q, sin_q = torch.cos(q_theta), torch.sin(q_theta)
        cos_k, sin_k = torch.cos(k_theta), torch.sin(k_theta)
        
        # Formulate real and imaginary parts of wavefunction states
        real_q = q_amp_norm * cos_q
        imag_q = q_amp_norm * sin_q
        
        real_k = k_amp_norm * cos_k
        imag_k = k_amp_norm * sin_k
        
        # Compute parts of transition probability overlap
        # real part of overlap: real_q * real_k + imag_q * imag_k
        term1 = torch.einsum("blhe,bshe->bhls", real_q, real_k)
        term2 = torch.einsum("blhe,bshe->bhls", imag_q, imag_k)
        
        # imaginary part of overlap: imag_q * real_k - real_q * imag_k
        term3 = torch.einsum("blhe,bshe->bhls", imag_q, real_k)
        term4 = torch.einsum("blhe,bshe->bhls", real_q, imag_k)
        
        real_overlap = term1 + term2
        imag_overlap = term3 - term4
        
        scores = real_overlap**2 + imag_overlap**2
        
        # Standard scale factor
        scale = self.scale or 1. / sqrt(d_k)
        scores = scores * scale
        
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
            
        A = self.dropout(torch.softmax(scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)
