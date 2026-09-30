import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th

from common import SiLU, TransformerEncoder
from utils import _extract_into_tensor, exponential_mapping
from step_sample import create_named_schedule_sampler, get_named_beta_schedule, space_timesteps


class DenoisedModel(nn.Module):
    def __init__(self, args):
        super(DenoisedModel, self).__init__()
        self.hidden_size = args.hidden_size

        if args.dif_decoder == 'mlp':
            self.decoder = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size * 4),
                SiLU(),
                nn.Linear(self.hidden_size * 4, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
            )
            self.decoder_is_seq = False
        else:
            self.decoder = TransformerEncoder(args, num_blocks=2, norm_first=False, hidden_size=self.hidden_size)
            self.decoder_is_seq = True

        self.time_embed = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            SiLU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size)
        )
        self.lambda_uncertainty = args.lambda_uncertainty

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        assert dim % 2 == 0
        half = dim // 2
        freqs = th.exp(
            -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32, device=timesteps.device) / half
        )
        args = timesteps.unsqueeze(-1).float() * freqs[None]
        emb = th.cat([th.cos(args), th.sin(args)], dim=-1)
        return emb

    def forward_cfg(self, c, x, t, mask_seq, mask_tgt, cfg_scale=1.0):
        cond = self.forward(c, x, t, mask_seq, mask_tgt, condition=True)
        uncond = self.forward(c, x, t, mask_seq, mask_tgt, condition=False)
        return uncond + cfg_scale * (cond - uncond)

    def forward(self, rep_item, x_t, t, mask_seq, mask_tgt, condition=True, return_align=False, align_layer_idx=-2):
        if not condition:
            rep_item = torch.zeros_like(rep_item)

        t = t.reshape(x_t.shape[0], -1)
        time_emb = self.time_embed(self.timestep_embedding(t, self.hidden_size)) 

        rep_diffu = rep_item + self.lambda_uncertainty * (x_t + time_emb)

        if self.decoder_is_seq:
            if return_align:
                rep_last, rep_align = self.decoder(rep_diffu, mask_seq, return_hidden=True, return_layer_idx=align_layer_idx)
                return rep_last, rep_align
            rep_diffu = self.decoder(rep_diffu, mask_seq)
            return rep_diffu
        else:
            rep_diffu = self.decoder(rep_diffu)
            if return_align:
                return rep_diffu, None
            return rep_diffu

class SEDRec(nn.Module):
    def __init__(self, args):
        super(SEDRec, self).__init__()
        self.args = args

        self.hidden_size = args.hidden_size
        self.cfg_scale = float(getattr(args, "cfg_scale", 1.0))
        self.geodesic = bool(getattr(args, "geodesic", False))

        # diffusion setup
        self.diffusion_steps = int(args.diffusion_steps)
        self.use_timesteps = space_timesteps(self.diffusion_steps, [self.diffusion_steps])

        betas = get_named_beta_schedule(args)
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        alphas = 1.0 - betas

        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        self.num_timesteps = int(self.betas.shape[0])
        self.rescale_timesteps = bool(getattr(args, "rescale_timesteps", True))
        self.schedule_sampler = create_named_schedule_sampler(
            getattr(args, "schedule_sampler_name", "uniform"),
            self.num_timesteps
        )

        # backbone
        self.net = DenoisedModel(args)
        self.ag_encoder = TransformerEncoder(args, num_blocks=2, norm_first=False)

        self.independent_diffusion = bool(getattr(args, "independent", True))
        self.sync_sem_last_t = int(getattr(args, "sync_sem_last_t", 1))  
        self.sem_cond_from_history_sem = int(getattr(args, "sem_cond_from_history_sem", 0))
        
        # semantic regularization
        self.beta_sem = float(getattr(args, "lambda_sem", 0.0))
        self.beta_align = float(getattr(args, "lambda_align", 0.0))
        self.align_layer_idx = int(getattr(args, "align_layer_idx", -2))  # -2 = last-1
        self.alpha_sem_train = float(getattr(args, "alpha_train", 0.0))
        self.alpha_sem_infer = float(getattr(args, "beta_infer", 0.0))

        # semantic embedding loading
        self.semantic_emb_file = getattr(args, "semantic_emb_file", "semantic_emb_sentence-t5-base.npy")
        self.freeze_semantic_emb = self._boollike(getattr(args, "freeze_semantic_emb", 1))
        self.freeze_semantic_proj = self._boollike(getattr(args, "freeze_semantic_proj", 1))

        sem_table, sem_dim = self._load_semantic_table(args)
        self.semantic_emb = nn.Embedding.from_pretrained(
            sem_table, freeze=self.freeze_semantic_emb, padding_idx=0
        )

        if sem_dim == self.hidden_size:
            self.semantic_proj = nn.Identity()
        else:
            self.semantic_proj = nn.Sequential(
                nn.Linear(sem_dim, self.hidden_size, bias=False),
                nn.LayerNorm(self.hidden_size),
            )

        if self.freeze_semantic_proj:
            for p in self.semantic_proj.parameters():
                p.requires_grad = False

    # ---------------- utilities ----------------

    @staticmethod
    def _boollike(x):
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, float)):
            return bool(int(x))
        if isinstance(x, str):
            return x.strip().lower() in ["1", "true", "yes", "y"]
        return bool(x)

    def _ones_token_mask(self, mask_seq):
        return torch.ones((mask_seq.shape[0], 1), device=mask_seq.device, dtype=mask_seq.dtype)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def _load_semantic_table(self, args):
        ds_dir = os.path.join("..", "datasets", "data", args.dataset)
        path = os.path.join(ds_dir, self.semantic_emb_file)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Semantic embedding file not found: {path}. "
                f"Place {self.semantic_emb_file} under {ds_dir} (same folder as dataset.pkl)."
            )

        arr = np.load(path)
        if arr.ndim != 2:
            raise ValueError(f"Semantic embedding must be 2-D, got {arr.shape} from {path}")
        arr = arr.astype(np.float32, copy=False)

        if arr.shape[0] == args.item_num:
            arr = np.concatenate([np.zeros((1, arr.shape[1]), dtype=np.float32), arr], axis=0)

        if arr.shape[0] < (args.item_num + 1):
            raise ValueError(
                f"Semantic embedding rows ({arr.shape[0]}) < expected (item_num+1={args.item_num+1}). "
                f"dataset={args.dataset}, file={path}."
            )

        return torch.from_numpy(arr), int(arr.shape[1])

    def _gather_last_nonzero(self, ids, mask):
        lengths = mask.sum(dim=1).long().clamp(min=1)
        idx = (lengths - 1).unsqueeze(1)
        return ids.gather(1, idx).squeeze(1)  

    def semantic_target(self, tag_ids, mask_tag):
        sem_id = self._gather_last_nonzero(tag_ids, mask_tag)
        sem = self.semantic_emb(sem_id).unsqueeze(1) 
        sem = self.semantic_proj(sem)                
        return sem

    def semantic_condition(self, rep_item, mask_seq, seq_ids=None):
        # option: use history semantic embeddings pooled
        if self.sem_cond_from_history_sem == 1 and seq_ids is not None:
            sem_h = self.semantic_proj(self.semantic_emb(seq_ids)) 
            m = mask_seq.float().unsqueeze(-1)
            denom = m.sum(dim=1, keepdim=True).clamp(min=1.0)
            return (sem_h * m).sum(dim=1, keepdim=True) / denom
        
        denom = mask_seq.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (rep_item * mask_seq.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom.unsqueeze(-1)
        return pooled

    @staticmethod
    def _cosine_loss(a, b, eps=1e-8):
        a = F.normalize(a, p=2, dim=-1, eps=eps)
        b = F.normalize(b, p=2, dim=-1, eps=eps)
        return (1.0 - (a * b).sum(dim=-1)).mean()

    # ---------------- diffusion core ----------------

    def q_sample(self, x_start, t, noise=None, mask=None):
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape

        if self.geodesic:
            x_start = F.normalize(x_start, p=2, dim=-1)

        x_t = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        if self.geodesic:
            x_t = exponential_mapping(x_start, x_t)

        if mask is None:
            return x_t

        mask = th.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)
        return th.where(mask == 0, x_start, x_t)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        return posterior_mean

    def p_mean_variance(self, rep_item, x_t, t, mask_seq, mask_tag):
        if self.cfg_scale == 1.0:
            x_0 = self.net(rep_item, x_t, self._scale_timesteps(t), mask_seq, mask_tag)
        else:
            x_0 = self.net.forward_cfg(rep_item, x_t, self._scale_timesteps(t), mask_seq, mask_tag, self.cfg_scale)

        model_log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))
        model_log_variance = _extract_into_tensor(model_log_variance, t, x_t.shape)
        model_mean = self.q_posterior_mean_variance(x_start=x_0, x_t=x_t, t=t)
        return model_mean, model_log_variance

    def p_sample(self, item_rep, noise_x_t, t, mask_seq, mask_tag):
        model_mean, model_log_variance = self.p_mean_variance(item_rep, noise_x_t, t, mask_seq, mask_tag)
        noise = th.randn_like(noise_x_t)
        nonzero_mask = (t != 0).float().unsqueeze(-1)
        sample_xt = model_mean + nonzero_mask * th.exp(0.5 * model_log_variance) * noise
        if self.geodesic:
            sample_xt = F.normalize(sample_xt, p=2, dim=-1)
        return sample_xt

    def independent_diffuse(self, tgt, mask, is_independent=False):
        
        if is_independent:
            t, _ = self.schedule_sampler.sample(tgt.shape[0] * tgt.shape[1], tgt.device)  
            t = t * mask.reshape(-1).long()
            x_t = self.q_sample(
                tgt.reshape(-1, tgt.shape[-1]),
                t,
                mask=mask.reshape(-1)
            ).reshape(*tgt.shape)
            t = t.reshape(tgt.shape[0], tgt.shape[1])
        else:
            t, _ = self.schedule_sampler.sample(tgt.shape[0], tgt.device)  
            x_t = self.q_sample(tgt, t, mask=mask)
        return x_t, t

    def shared_t_sem_last(self, x_start_ext, mask_ext):
    
        B, Lp1, _ = x_start_ext.shape
        t_base, _ = self.schedule_sampler.sample(B, x_start_ext.device)  
        t_ext = th.zeros((B, Lp1), dtype=th.long, device=x_start_ext.device)
        t_ext[:, 0] = t_base
        t_ext[:, -1] = t_base
        t_ext = t_ext * mask_ext.long()  
        x_t = self.q_sample(x_start_ext, t_ext, mask=mask_ext)
        return x_t, t_ext

    # ---------------- forward / inference ----------------

    def forward(self, item_rep, item_tag, mask_seq, mask_tag, tag_ids=None, seq_ids=None):
        # Encode history
        item_rep_enc = self.ag_encoder(item_rep, mask_seq)

        # semantic tokens
        sem_cond = self.semantic_condition(item_rep_enc, mask_seq, seq_ids=seq_ids)  # [B,1,D]
        if tag_ids is not None:
            sem_tgt = self.semantic_target(tag_ids, mask_tag)  # [B,1,D]
        else:
            sem_tgt = item_tag[:, -1:, :]

        # prepend semantic token
        item_rep_ext = torch.cat([sem_cond, item_rep_enc], dim=1)  # [B,L+1,D]
        item_tag_ext = torch.cat([sem_tgt, item_tag], dim=1)       # [B,L+1,D]

        mask_seq_ext = torch.cat([self._ones_token_mask(mask_seq), mask_seq], dim=1)  # [B,L+1]
        mask_tag_ext = torch.cat([self._ones_token_mask(mask_tag), mask_tag], dim=1)  # [B,L+1]

        # diffuse
        if self.sync_sem_last_t == 1:
            x_t_ext, t_ext = self.shared_t_sem_last(item_tag_ext, mask_tag_ext)
        else:
            x_t_ext, t_ext = self.independent_diffuse(item_tag_ext, mask_tag_ext, is_independent=self.independent_diffusion)

        if self.cfg_scale != 1.0:
            drop = (torch.rand([mask_seq_ext.shape[0], 1, 1], device=item_rep_ext.device) > 0.7)
            item_rep_ext = torch.where(drop, torch.zeros_like(item_rep_ext), item_rep_ext)

        if self.beta_align > 0:
            denoised_ext, h_align = self.net(
                item_rep_ext, x_t_ext, t_ext, mask_seq_ext, mask_tag_ext,
                return_align=True, align_layer_idx=self.align_layer_idx
            )
        else:
            denoised_ext = self.net(item_rep_ext, x_t_ext, t_ext, mask_seq_ext, mask_tag_ext)
            h_align = None

        denoised_sem = denoised_ext[:, :1, :]
        denoised_items = denoised_ext[:, 1:, :]

        if self.alpha_sem_train > 0:
            denoised_items[:, -1:, :] = denoised_items[:, -1:, :] + self.alpha_sem_train * denoised_sem

        item_losses = F.mse_loss(denoised_items, item_tag, reduction='none')  # [B,L,D]
        item_losses = item_losses * (mask_tag / mask_tag.sum(1, keepdim=True).clamp(min=1.0)).unsqueeze(-1)
        item_loss = item_losses.sum(1).mean()

        sem_loss = F.mse_loss(denoised_sem, sem_tgt, reduction='mean')

        # if self.beta_sem_cons > 0:
        #     sem_cons_loss = self._cosine_loss(denoised_sem, sem_cond.detach())
        # else:
        sem_cons_loss = denoised_sem.new_zeros(())

        if self.beta_align > 0 and (h_align is not None):
            h_sem = h_align[:, :1, :]
            h_last = h_align[:, -1:, :]
            align_sem = self._cosine_loss(h_sem, sem_tgt.detach())
            align_last = self._cosine_loss(h_last, sem_tgt.detach())
            align_loss = 0.5 * (align_sem + align_last)
        else:
            align_loss = denoised_sem.new_zeros(())

        total_loss = item_loss + self.beta_sem * sem_loss + self.beta_align * align_loss
        
        return denoised_items, total_loss

    def denoise_sample(self, seq, tgt, mask_seq, mask_tag, seq_ids=None):
        seq_enc = self.ag_encoder(seq, mask_seq)
        sem_cond = self.semantic_condition(seq_enc, mask_seq, seq_ids=seq_ids)  # [B,1,D]
        rep_item_ext = torch.cat([sem_cond, seq_enc], dim=1)  # [B,L+1,D]
        mask_seq_ext = torch.cat([self._ones_token_mask(mask_seq), mask_seq], dim=1)

        sem_noise = th.randn_like(sem_cond)
        last_noise = th.randn_like(tgt[:, -1:, :])

        B, L, D = tgt.shape
        zeros_mid = [0] * (L - 1)

        for i in range(self.num_timesteps - 1, -1, -1):
            t_ext = th.tensor([i] + zeros_mid + [i], device=seq.device).unsqueeze(0).repeat(B, 1)  # [B,L+1]
            noise_x_t_ext = torch.cat([sem_noise, tgt[:, :-1, :], last_noise], dim=1)  # [B,L+1,D]
            mask_tag_ext = torch.cat([self._ones_token_mask(mask_tag), mask_tag], dim=1)

            noise_x_t_ext = self.p_sample(rep_item_ext, noise_x_t_ext, t_ext, mask_seq_ext, mask_tag_ext)

            sem_noise = noise_x_t_ext[:, :1, :]
            last_noise = noise_x_t_ext[:, -1:, :]

            if self.alpha_sem_infer > 0:
                last_noise = last_noise + self.alpha_sem_infer * sem_noise

        return noise_x_t_ext[:, 1:, :]
