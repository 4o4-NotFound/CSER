
# self supervised multimodal multi-task learning network (Left-11S step1: TFR-Sparse + generate)
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.subNets.Textmodel import Language_model

__all__ = ['CMCM']

class CMCM(nn.Module):
    def __init__(self, args):
        super(CMCM, self).__init__()
        # Text encoding (ChatGLM etc.)
        self.LLM = Language_model(args)

        # Modal input dims
        text_in, audio_in, video_in = args.feature_dims[:]
        text_len, audio_len, video_len = args.seq_lens[:]

        # Audio & Video encoders
        self.audio_LSTM = TVA_LSTM(audio_in, args.a_lstm_hidden_size,
                                   num_layers=args.a_lstm_layers, dropout=args.a_lstm_dropout)
        self.video_LSTM = TVA_LSTM(video_in, args.v_lstm_hidden_size,
                                   num_layers=args.v_lstm_layers, dropout=args.v_lstm_dropout)

        # Text-guided mixer (Lite: shared low-rank + dual-branch gating)
        self.text_guide_mixer = Text_guide_separate_mixer()

        # Multi-scale fusion to pseudo tokens (keeps your existing interface)
        fusion_input_size = 256
        self.mutli_scale_fusion = mutli_scale_fusion(
            input_size=fusion_input_size, output_size=text_in, pseudo_tokens=args.pseudo_tokens
        )

        # Step1: Parameter-free Text Feedback Refiner (sparse rect token)
        self.tfr_sparse = Text_Feedback_Refiner_Sparse(temp=1.0, keep_ratio=0.2, detach_warmup=200)

        self.tfr_sparse.use_position_debias = True
        # ===== Top-K 可视化自动保存 =====
        self.enable_topk_save = False
        self.final_topk_cases = []

    def _get_tokenizer(self):
        tokenizer = None
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            tokenizer = self.tokenizer
        elif hasattr(self, 'LLM'):
            if hasattr(self.LLM, 'tokenizer') and self.LLM.tokenizer is not None:
                tokenizer = self.LLM.tokenizer
            elif hasattr(self.LLM, 'model') and hasattr(self.LLM.model, 'tokenizer') and self.LLM.model.tokenizer is not None:
                tokenizer = self.LLM.model.tokenizer
        return tokenizer

    def _get_punct_id_set(self):
        tokenizer = self._get_tokenizer()
        punct_ids = set()
        if tokenizer is None:
            return punct_ids

        punct_chars = set(list("，。！？；：、“”‘’（）()【】《》〈〉—…·,.!?;:'\"-_/\\|@#$%^&*+=~`[]{}<>"))

        vocab = None
        if hasattr(tokenizer, "get_vocab"):
            try:
                vocab = tokenizer.get_vocab()
            except Exception:
                vocab = None
        if vocab is None:
            return punct_ids

        for tok, idx in vocab.items():
            if tok is None:
                continue
            s = str(tok).strip()
            # 兼容常见 tokenizer 的词前缀
            for prefix in ["Ġ", "▁", "##"]:
                if s.startswith(prefix):
                    s = s[len(prefix):]
            if len(s) == 0:
                continue
            pure = True
            for ch in s:
                if ch not in punct_chars:
                    pure = False
                    break
            if pure:
                punct_ids.add(int(idx))
        return punct_ids

    def _build_text_masks(self, text_ids):
        """
        根据 text_ids 自动构造真实 token mask：
        1) 优先去掉左侧连续 special token 前缀；
        2) 若 special token 无法识别，则回退为去掉左侧连续相同 token 前缀；
        3) 再过滤纯标点 token。
        """
        device = text_ids.device
        B, L = text_ids.shape

        tokenizer = self._get_tokenizer()

        special_ids = set()
        if tokenizer is not None:
            for attr in ['pad_token_id', 'eos_token_id', 'bos_token_id']:
                val = getattr(tokenizer, attr, None)
                if val is not None:
                    special_ids.add(int(val))

        valid_mask = torch.ones_like(text_ids, dtype=torch.bool, device=device)

        # 先尝试基于 special token 去掉左侧前缀
        if len(special_ids) > 0:
            for b in range(B):
                started = False
                for i in range(L):
                    tid = int(text_ids[b, i].item())
                    if (not started) and (tid in special_ids):
                        valid_mask[b, i] = False
                    else:
                        started = True

        # 若 special token 方案未生效，则回退到“左侧连续相同 token 前缀”
        for b in range(B):
            if bool(valid_mask[b].all()):
                lead_id = int(text_ids[b, 0].item())
                i = 0
                while i < L and int(text_ids[b, i].item()) == lead_id:
                    valid_mask[b, i] = False
                    i += 1
                # 防止整句都被误清空
                if int(valid_mask[b].sum().item()) == 0:
                    valid_mask[b] = True

        # 过滤标点 token
        punct_ids = self._get_punct_id_set()
        if len(punct_ids) > 0:
            punct_mask = torch.ones_like(text_ids, dtype=torch.bool, device=device)
            for pid in punct_ids:
                punct_mask &= (text_ids != pid)
            valid_mask &= punct_mask

        # 防止某一行全 False
        for b in range(B):
            if int(valid_mask[b].sum().item()) == 0:
                valid_mask[b] = torch.ones(L, dtype=torch.bool, device=device)

        return valid_mask


    # Training forward: returns loss for trainer
    def forward(self, labels, text, audio, video):
        audio, audio_len = audio
        video, video_len = video
        text, text_len = text
        # print("text_len[:20] =", text_len[:20])
        # NOTE: keep consistent with your pipeline (ids at [:,0,:])
        text_ids = text[:, 0, :].long()
        valid_mask = self._build_text_masks(text_ids)
        text_len = valid_mask.sum(dim=1)

        text = self.LLM.text_embedding(text_ids)  # (B, L, D)
        # Encode A/V
        video_h = self.video_LSTM(video, video_len)  # (B,256)
        audio_h = self.audio_LSTM(audio, audio_len)  # (B,256)

        # Text-guided gating
        audio_g, video_g = self.text_guide_mixer.get_guided_features(audio_h, video_h, text)
        fusion_h = self.text_guide_mixer(audio_g, video_g)  # (B,256)

        # Multi-scale fusion -> pseudo tokens
        fusion_h = self.mutli_scale_fusion(fusion_h)  # (B, P, D)

        # Parameter-free sparse rect token from text
        #rect = self.tfr_sparse(fusion_h, text, text_len)  # (B,1,D)
        rect = self.tfr_sparse(fusion_h, text, text_len, valid_mask=valid_mask)  # (B,1,D)

        # Pack and feed to LLM
        LLM_input = torch.cat([fusion_h, rect, text], dim=1)  # (B, P+1+L, D)
        LLM_output = self.LLM(LLM_input, labels)

        res = {
            'Loss': LLM_output.loss,
            'Feature_a': audio_h,
            'Feature_v': video_h,
            'Feature_f': fusion_h,
        }
        return res

    # Inference/validation: MUST exist for your runner (AMIO -> Model.generate)
    def generate(self, text, audio, video):
        audio, audio_len = audio
        video, video_len = video
        text, text_len = text

        # 保留原始 token ids
        text_ids = text[:, 0, :].long()  # (B, L)

        valid_mask = self._build_text_masks(text_ids)
        text_len = valid_mask.sum(dim=1)

        # text ids at [:,0,:] -> embeddings
        text = self.LLM.text_embedding(text_ids)  # (B, L, D)

        # Encode A/V
        audio_h = self.audio_LSTM(audio, audio_len)  # (B,256)
        video_h = self.video_LSTM(video, video_len)  # (B,256)

        # Text-guided gating + fusion
        audio_g, video_g = self.text_guide_mixer.get_guided_features(audio_h, video_h, text)
        fusion_h = self.text_guide_mixer(audio_g, video_g)  # (B,256)

        # Multi-scale fusion -> pseudo tokens
        fusion_h = self.mutli_scale_fusion(fusion_h)  # (B, P, D)

        # Parameter-free sparse rect token
        #rect = self.tfr_sparse(fusion_h, text, text_len)  # (B,1,D)

        rect = self.tfr_sparse(fusion_h, text, text_len, valid_mask=valid_mask)

        # ===== 只在最终 TEST 时缓存，不在这里保存图片 =====
        if self.enable_topk_save:
            debug = self.tfr_sparse.last_debug
            if debug is not None:
                alpha_before_batch = debug['alpha_before'].numpy()   # (B, L)
                alpha_after_batch = debug['alpha_after'].numpy()     # (B, L)
                mask_batch = debug['mask'].numpy()                   # (B, L)
                valid_mask_batch = debug.get('valid_mask', None)
                if valid_mask_batch is not None:
                    valid_mask_batch = valid_mask_batch.numpy()

                B = text_ids.size(0)
                for b in range(B):
                    selected_pos = np.where(mask_batch[b] > 0)[0].tolist()
                    if valid_mask_batch is not None:
                        selected_pos = [p for p in selected_pos if valid_mask_batch[b][p] > 0]

                    self.final_topk_cases.append({
                        'text_ids': text_ids[b].detach().cpu().tolist(),
                        'alpha_before': alpha_before_batch[b].tolist(),
                        'alpha_after': alpha_after_batch[b].tolist(),
                        'selected_pos': selected_pos,
                    })

        # LLM input and decoding
        LLM_input = torch.cat([fusion_h, rect, text], dim=1)  # (B, P+1+L, D)
        return self.LLM.generate(LLM_input)


class TVA_LSTM(nn.Module):
    def __init__(self, in_dim, hidden_size, num_layers, dropout):
        super(TVA_LSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=in_dim,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             batch_first=True,
                             bidirectional=True)
        self.projector = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.GELU(),
            nn.Linear(256, 256)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, x_lens):
        packed = pack_padded_sequence(x, x_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm1(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        out = torch.mean(out, dim=1)
        out = self.projector(out)
        out = self.drop(out)
        return out


class Text_guide_mixer(nn.Module):
    def __init__(self):
        super(Text_guide_mixer, self).__init__()
        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.text_mlp = nn.Linear(4096, 256)

    def forward(self, audio, video, text):
        text_GAP = self.GAP(text.permute(0, 2, 1)).squeeze()
        text_knowledge = self.text_mlp(text_GAP)

        audio_mixed = torch.mul(audio, text_knowledge)
        video_mixed = torch.mul(video, text_knowledge)

        fusion = audio_mixed + video_mixed
        return fusion


class Text_guide_separate_mixer(nn.Module):
    """
    TGSM_Lite：共享低秩 + 分路投影，无注意力/FFN，低参且稳定
    """
    def __init__(self, text_dim=2048, out_dim=256, rank=32, use_attn=False, d_k=64):
        super().__init__()
        self.GAP = nn.AdaptiveAvgPool1d(1)
        # 共享低秩基
        self.text_to_low = nn.Linear(text_dim, rank, bias=False)
        # 分路到 A/V 门控
        self.low_to_audio = nn.Linear(rank, out_dim, bias=False)
        self.low_to_video = nn.Linear(rank, out_dim, bias=False)

        # 可选极轻注意力（默认关闭）
        self.use_attn = use_attn
        if use_attn:
            self.q_proj = nn.Linear(out_dim, d_k, bias=False)
            self.k_proj = nn.Linear(out_dim, d_k, bias=False)
            self.v_proj = nn.Linear(out_dim, out_dim, bias=False)
            self.scale = d_k ** -0.5

    def get_guided_features(self, audio, video, text):
        # text: (B, T, 4096) -> GAP -> (B, 4096)
        z = self.GAP(text.permute(0, 2, 1)).squeeze(-1)
        low = self.text_to_low(z)                           # (B,rank)
        a_gate = torch.sigmoid(self.low_to_audio(low))      # (B,256)
        v_gate = torch.sigmoid(self.low_to_video(low))      # (B,256)

        def apply_gate(x, g):
            return x * (g.unsqueeze(1) if x.dim()==3 else g)

        audio_g = apply_gate(audio, a_gate)
        video_g = apply_gate(video, v_gate)
        return audio_g, video_g

    def forward(self, audio_guided, video_guided):
        # 默认无注意力，直接融合
        if not self.use_attn:
            return audio_guided + video_guided
        # 轻注意力（可选）
        Q = self.q_proj(audio_guided).unsqueeze(1)   # (B,1,d_k)
        K = self.k_proj(video_guided).unsqueeze(1)   # (B,1,d_k)
        V = self.v_proj(video_guided).unsqueeze(1)   # (B,1,out_dim)
        attn = torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)  # (B,1,1)
        out = (attn @ V).squeeze(1) + audio_guided
        return out


class Text_Feedback_Refiner_Sparse(nn.Module):
    """Parameter-free TFR: Top-k sparse rect token from text guided by fusion summary.
    Args:
        temp (float): initial softmax temperature.
        keep_ratio (float): fraction of tokens to keep in Top-k (0~1).
        detach_warmup (int): steps to detach fusion summary to avoid early collapse.
    """
    def __init__(self, temp: float = 1.0, keep_ratio: float = 1.0, detach_warmup: int = 200):
        super().__init__()
        self.register_buffer('step', torch.zeros(1, dtype=torch.long), persistent=False)
        self.temp0 = float(temp)
        self.keep_ratio = float(keep_ratio)
        self.detach_warmup = int(detach_warmup)

        # 新增：缓存最近一次前向传播的调试信息
        self.last_debug = None

        self.use_position_debias = False
        self.position_debias_start = 1.0
        self.position_debias_end = 0.5

    def _anneal_temp(self):
        # Linear anneal from 1.0 -> 0.4 in ~3000 steps
        s = self.step.item()
        t = max(0.4, self.temp0 * (1.0 - 0.0002 * s))
        return t

    def forward(self, fusion_tokens: torch.Tensor, text_tokens: torch.Tensor, text_len=None,
                valid_mask=None) -> torch.Tensor:
        # fusion_tokens: (B, P, D); text_tokens: (B, L, D)
        B, L, D = text_tokens.shape

        # fusion summary
        s = fusion_tokens.mean(dim=1)  # (B, D)
        if self.training and self.step.item() < self.detach_warmup:
            s = s.detach()

        # cosine sim between each text token and s
        s_n = F.normalize(s, dim=-1)  # (B, D)
        t_n = F.normalize(text_tokens, dim=-1)  # (B, L, D)
        sim = (t_n * s_n.unsqueeze(1)).sum(dim=-1)  # (B, L)

        # padding mask
        if valid_mask is not None:
            valid_mask = valid_mask.to(text_tokens.device).bool()
        elif text_len is not None:
            if not torch.is_tensor(text_len):
                text_len = torch.tensor(text_len, device=text_tokens.device)
            text_len = text_len.to(text_tokens.device).long().view(-1)
            valid_mask = (torch.arange(L, device=text_tokens.device).unsqueeze(0) < text_len.unsqueeze(1))  # (B, L)
        else:
            valid_mask = None

        # 在 softmax 前屏蔽 padding token
        if valid_mask is not None:
            sim = sim.masked_fill(~valid_mask, -1e9)

        # annealed softmax
        temp = self._anneal_temp()
        alpha_before = F.softmax(sim / temp, dim=1)  # (B, L)

        # 防止极端情况下出现数值问题
        if valid_mask is not None:
            alpha_before = alpha_before * valid_mask.float()
            alpha_before = alpha_before / (alpha_before.sum(dim=1, keepdim=True) + 1e-12)

        # ===== 新增：位置去偏分布（用于对照实验）=====
        alpha_for_topk = alpha_before
        alpha_debias = alpha_before

        if self.use_position_debias:
            pos_bias = torch.linspace(
                self.position_debias_start,
                self.position_debias_end,
                steps=L,
                device=text_tokens.device,
                dtype=alpha_before.dtype
            ).unsqueeze(0)  # (1, L)

            alpha_debias = alpha_before / (pos_bias + 1e-12)

            if valid_mask is not None:
                alpha_debias = alpha_debias * valid_mask.float()

            alpha_debias = alpha_debias / (alpha_debias.sum(dim=1, keepdim=True) + 1e-12)
            alpha_for_topk = alpha_debias

        # Top-k sparsification
        if valid_mask is None:
            k = max(1, int(L * self.keep_ratio))
            topv, topi = torch.topk(alpha_for_topk, k=k, dim=1)
            mask = torch.zeros_like(alpha_for_topk).scatter(1, topi, 1.0)
        else:
            valid_count = valid_mask.sum(dim=1)  # (B,)
            k_each = torch.clamp((valid_count.float() * self.keep_ratio).long(), min=1)  # (B,)
            max_k = int(k_each.max().item())

            topv, topi = torch.topk(alpha_for_topk, k=max_k, dim=1)
            rank_mask = (torch.arange(max_k, device=text_tokens.device).unsqueeze(0) < k_each.unsqueeze(1)).float()

            #print("valid_count[:20] =", valid_count[:20].detach().cpu().tolist())
            #print("k_each[:20] =", k_each[:20].detach().cpu().tolist())

            mask = torch.zeros_like(alpha_for_topk).scatter(1, topi, rank_mask)

        alpha_after = alpha_for_topk * mask
        alpha_after = alpha_after / (alpha_after.sum(dim=1, keepdim=True) + 1e-12)  # re-normalize

        # rect token
        rect = torch.bmm(alpha_after.unsqueeze(1), text_tokens)  # (B, 1, D)

        # 新增：缓存可视化信息
        self.last_debug = {
            'sim': sim.detach().cpu(),
            'alpha_before': alpha_before.detach().cpu(),
            'alpha_debias': alpha_debias.detach().cpu(),
            'alpha_after': alpha_after.detach().cpu(),
            'topi': topi.detach().cpu(),
            'mask': mask.detach().cpu(),
            'valid_mask': None if valid_mask is None else valid_mask.detach().cpu(),
            'valid_count': None if valid_mask is None else valid_mask.sum(dim=1).detach().cpu(),
            'k_each': None if valid_mask is None else torch.clamp((valid_mask.sum(dim=1).float() * self.keep_ratio).long(), min=1).detach().cpu(),
            'k': int(max_k) if valid_mask is not None else k,
            'temp': float(temp),
            'use_position_debias': bool(self.use_position_debias),
        }

        # increase step counter
        self.step += 1
        return rect


class mutli_scale_fusion(nn.Module):
    def __init__(self, input_size, output_size, pseudo_tokens=4):
        super(mutli_scale_fusion, self).__init__()
        multi_scale_hidden = 256
        self.scale1 = nn.Sequential(
            nn.Linear(input_size, output_size // 8),
            nn.GELU(),
            nn.Linear(output_size // 8, multi_scale_hidden)
        )
        self.scale2 = nn.Sequential(
            nn.Linear(input_size, output_size // 32),
            nn.GELU(),
            nn.Linear(output_size // 32, multi_scale_hidden)
        )
        self.scale3 = nn.Sequential(
            nn.Linear(input_size, output_size // 16),
            nn.GELU(),
            nn.Linear(output_size // 16, multi_scale_hidden)
        )

        self.integrating = Integrating(scales=3)

        self.multi_scale_projector = nn.Linear(multi_scale_hidden, output_size)
        # 低参 projector：把 (B,D,1) 线性到 (B,D,P)
        self.projector = nn.Linear(1, pseudo_tokens)

    def forward(self, x):
        # x : (B, 256)
        # compute different scale experts outputs
        scale1 = self.scale1(x)
        scale2 = self.scale2(x)
        scale3 = self.scale3(x)

        # Calculate the expert outputs
        multi_scale_stack = torch.stack([scale1, scale2, scale3], dim=2)  # (B, H, 3)
        multi_scale_integrating = self.integrating(multi_scale_stack)

        multi_scale = self.multi_scale_projector(multi_scale_integrating)  # (B, D)
        output = self.projector(multi_scale.unsqueeze(2))  # (B, D, P)
        return output.permute(0, 2, 1)  # (B, P, D)


class Integrating(nn.Module):
    def __init__(self, scales):
        super(Integrating, self).__init__()
        self.Integrating_layer = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(1, scales), stride=1),
        )

    def forward(self, x):
        # x: (B, H, S)
        x = x.unsqueeze(1)            # (B, 1, H, S)
        x = self.Integrating_layer(x) # (B, 1, H, 1)
        x = x.squeeze((1, 3))         # (B, H)
        return x
