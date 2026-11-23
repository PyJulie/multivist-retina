"""Multi-visit transformer (MVT).

A pretrained RETFound ViT-Large encoder processes every image of a patient. Three interaction
modules (C1-C3), inserted after encoder blocks 18, 21 and 24, let each image attend to the
patient's other images: within-image spatial attention, same-eye cross-visit attention and
same-visit bilateral attention. A query-based readout then produces one prediction per patient.

Parameter names match the checkpoints used in the study, so trained weights load with
`strict=True`.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

TIME_SCALES_DAYS = (1, 7, 30, 90, 180, 365, 730, 1825)
UNKNOWN_EYE = 2  # laterality codes: 0 right, 1 left, 2 unknown
IMAGENET_MEAN = (.485, .456, .406)
IMAGENET_STD = (.229, .224, .225)


def time_features(days):
    """Sine and cosine of an interval in days at eight time scales (16 features)."""
    scales = days.new_tensor(TIME_SCALES_DAYS).float()
    x = days.float().unsqueeze(-1) / scales
    return torch.cat((torch.sin(x), torch.cos(x)), dim=-1)


def image_relation(eye, dt, kind):
    """Allowed (query image, key image) pairs for one patient.

    Images of the same visit date share the same interval to the index date (`dt`). Images
    with unknown laterality never take part in cross-image attention.
    """
    known = (eye[:, None] < UNKNOWN_EYE) & (eye[None, :] < UNKNOWN_EYE)
    same_eye = eye[:, None] == eye[None, :]
    same_date = dt[None, :] == dt[:, None]
    if kind == 'temporal':   # same eye, other visit dates
        return known & same_eye & ~same_date
    if kind == 'bilateral':  # fellow eye, same visit date
        return known & ~same_eye & same_date
    raise ValueError(kind)


class Block(nn.Module):
    """Pre-norm attention block: 12 heads, feed-forward expansion 4, dropout 0.1, and
    per-channel residual scales initialised at 0.1."""

    def __init__(self, d, heads, drop=.1):
        super().__init__()
        assert d % heads == 0
        self.heads, self.drop = heads, drop
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))
        self.scale1 = nn.Parameter(torch.full((d,), .1))
        self.scale2 = nn.Parameter(torch.full((d,), .1))

    def forward(self, x, allowed=None, bias=None):
        b, n, d = x.shape
        q, k, v = (self.qkv(self.norm1(x)).view(b, n, 3, self.heads, d // self.heads)
                   .permute(2, 0, 3, 1, 4).unbind(0))
        mask = allowed
        if bias is not None:
            mask = bias.to(q.dtype).unsqueeze(0)
            if allowed is not None:
                mask = mask.masked_fill(~allowed[None, None], float('-inf'))
        z = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=self.drop if self.training else 0.)
        z = z.transpose(1, 2).reshape(b, n, d)
        x = x + F.dropout(self.out(z), self.drop, self.training) * self.scale1
        return x + F.dropout(self.ff(self.norm2(x)), self.drop, self.training) * self.scale2


def block_call(module, *args):
    """Gradient checkpointing during training, a plain call otherwise."""
    if module.training and torch.is_grad_enabled():
        return checkpoint(module, *args, use_reentrant=False)
    return module(*args)


class CrossImageAttention(nn.Module):
    """Attention from each image's tokens to the tokens of its eligible partner images.

    `gap` is the learned per-head interval bias: an MLP of sine/cosine features of the signed
    interval between the two images plus a signed log-interval, added to the attention logits.
    """

    def __init__(self, d, heads, kind, drop=.1):
        super().__init__()
        self.kind = kind
        self.block = Block(d, heads, drop)
        self.gap = nn.Sequential(nn.Linear(17, 32), nn.GELU(), nn.Linear(32, heads))

    def forward(self, x, eye, dt):
        n, p, d = x.shape
        relation = image_relation(eye, dt, self.kind)
        active = relation.any(-1)
        # Images without an eligible partner attend to themselves only to keep the softmax
        # finite; their update is discarded below, so they pass through unchanged.
        allowed_img = relation | ((~active)[:, None] & torch.eye(n, device=x.device, dtype=torch.bool))
        allowed = allowed_img.repeat_interleave(p, 0).repeat_interleave(p, 1)
        gap = dt[None, :] - dt[:, None]
        features = torch.cat((time_features(gap),
                              torch.sign(gap)[..., None] * torch.log1p(gap.abs())[..., None] / 8.), -1)
        bias = self.gap(features).permute(2, 0, 1).repeat_interleave(p, 1).repeat_interleave(p, 2)
        z = block_call(self.block, x.reshape(1, n * p, d), allowed, bias).reshape(n, p, d)
        return x + (z - x) * active[:, None, None].to(x.dtype)


class CrossBlock(nn.Module):
    """Pre-norm cross-attention block (queries attend to image tokens) with a feed-forward layer."""

    def __init__(self, d, heads, drop=.1):
        super().__init__()
        self.heads, self.drop = heads, drop
        self.nq, self.nk, self.nff = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.kv, self.proj = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))

    def forward(self, q, x):
        b, n, d = q.shape
        m, h = x.shape[1], self.heads
        qq = self.q(self.nq(q)).view(b, n, h, d // h).transpose(1, 2)
        k, v = self.kv(self.nk(x)).view(b, m, 2, h, d // h).permute(2, 0, 3, 1, 4).unbind(0)
        z = F.scaled_dot_product_attention(qq, k, v, dropout_p=self.drop if self.training else 0.)
        q = q + F.dropout(self.proj(z.transpose(1, 2).reshape(b, n, d)), self.drop, self.training)
        return q + F.dropout(self.ff(self.nff(q)), self.drop, self.training)


class Readout(nn.Module):
    """Patient-level aggregation: Q learned queries, two cross-attention layers, mean over queries."""

    def __init__(self, d, heads, depth=2, queries=8, drop=.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, queries, d) * .02)
        self.blocks = nn.ModuleList([CrossBlock(d, heads, drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        x = x.reshape(1, -1, x.shape[-1])
        q = self.query
        for block in self.blocks:
            q = block_call(block, q, x)
        return self.norm(q.mean(1)).squeeze(0)


class TokenEmbedding(nn.Module):
    """Projection of the 17 compact tokens plus positional, laterality and time embeddings."""

    def __init__(self, d_in, d, grid=4):
        super().__init__()
        self.proj = nn.Linear(d_in, d)
        self.eye = nn.Embedding(3, d)
        self.time = nn.Sequential(nn.Linear(2 * len(TIME_SCALES_DAYS), d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.randn(1, grid * grid + 1, d) * .02)

    def forward(self, x, eye, dt):
        assert x.shape[1] == self.pos.shape[1]
        return self.proj(x) + self.pos + self.eye(eye)[:, None] + self.time(time_features(dt))[:, None]


def compact_tokens(x, grid=4):
    """Classification token plus patch tokens average-pooled to a grid x grid layout."""
    n, nt, d = x.shape
    side = math.isqrt(nt - 1)
    assert side * side == nt - 1
    patches = x[:, 1:].transpose(1, 2).reshape(n, d, side, side)
    patches = F.adaptive_avg_pool2d(patches, (grid, grid)).flatten(2).transpose(1, 2)
    return torch.cat((x[:, :1], patches), dim=1)


class InteractionModule(nn.Module):
    """One interaction module (C1, C2 or C3) inserted between encoder blocks.

    Spatial attention within each image, then same-eye cross-visit attention, then same-visit
    bilateral attention. The compact update is projected back to the encoder width, scaled by a
    learnable residual scale (initialised at 0.01), spatially upsampled to the patch grid and
    added to the encoder tokens.
    """

    def __init__(self, d_in=1024, d=768, heads=12, grid=4, drop=.1):
        super().__init__()
        self.grid = grid
        self.embedding = TokenEmbedding(d_in, d, grid)
        self.spatial = Block(d, heads, drop)
        self.temporal = CrossImageAttention(d, heads, 'temporal', drop)
        self.bilateral = CrossImageAttention(d, heads, 'bilateral', drop)
        self.back = nn.Linear(d, d_in)
        self.scale = nn.Parameter(torch.tensor(.01))

    def forward(self, tokens, eye, dt, groups):
        small = compact_tokens(tokens, self.grid)
        z = self.embedding(small, eye, dt)
        z = block_call(self.spatial, z)
        parts = []
        for a, b in groups:  # one patient at a time
            p = self.temporal(z[a:b], eye[a:b], dt[a:b])
            parts.append(self.bilateral(p, eye[a:b], dt[a:b]))
        z = torch.cat(parts)
        delta = self.back(z - self.embedding.proj(small)) * self.scale
        n, nt, d = tokens.shape
        side = math.isqrt(nt - 1)
        patch = delta[:, 1:].transpose(1, 2).reshape(n, d, self.grid, self.grid)
        patch = F.interpolate(patch, (side, side), mode='bilinear', align_corners=False).flatten(2).transpose(1, 2)
        return tokens + torch.cat((delta[:, :1], patch), 1).to(tokens.dtype)


class MVT(nn.Module):
    """RETFound encoder + interaction modules C1-C3 + patient-level readout + linear head.

    Batch format (see mvt/data.py): `x` uint8 images (N, 224, 224, 3) of all patients in the
    batch, `eye` (N,) laterality codes, `dt` (N,) days between acquisition and index date, and
    `groups`, a list of (start, stop) image ranges, one per patient. Returns one logit per patient.
    """

    def __init__(self, encoder, width=768, heads=12, grid=4, dropout=.1,
                 insert_after=(18, 21, 24), queries=8, image_chunk=16):
        super().__init__()
        assert encoder.num_prefix_tokens == 1 and max(insert_after) <= len(encoder.blocks)
        self.encoder = encoder
        self.encoder.set_grad_checkpointing(False)  # checkpointing is applied per block below
        self.insert_after, self.grid, self.image_chunk = tuple(insert_after), grid, image_chunk
        self.checkpointing = True
        d_in = encoder.embed_dim
        self.embedding = TokenEmbedding(d_in, width, grid)
        self.couplings = nn.ModuleList([InteractionModule(d_in, width, heads, grid, dropout)
                                        for _ in self.insert_after])
        self.read = Readout(width, heads, queries=queries, drop=dropout)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Dropout(dropout), nn.Linear(width, 1))

    def set_epoch(self, epoch, warmup_epochs=1):
        """Encoder frozen during the warm-up epoch, fine-tuned afterwards."""
        for p in self.encoder.parameters():
            p.requires_grad_(epoch > warmup_epochs)

    def _stem(self, pixels):
        x = pixels.permute(0, 3, 1, 2).float() / 255.
        mean = x.new_tensor(IMAGENET_MEAN)[None, :, None, None]
        std = x.new_tensor(IMAGENET_STD)[None, :, None, None]
        x = ((x - mean) / std).contiguous()
        x = self.encoder._pos_embed(self.encoder.patch_embed(x))
        return self.encoder.norm_pre(self.encoder.patch_drop(x))

    def _blocks(self, x, start, stop):
        out = []
        for chunk in x.split(self.image_chunk):
            for j in range(start, stop):
                block = self.encoder.blocks[j]
                if self.checkpointing and self.training and torch.is_grad_enabled():
                    chunk = checkpoint(block, chunk, use_reentrant=False)
                else:
                    chunk = block(chunk)
            out.append(chunk)
        return torch.cat(out)

    def features(self, batch):
        x = torch.cat([self._stem(p) for p in batch['x'].split(self.image_chunk)])
        start = 0
        for stop, module in zip(self.insert_after, self.couplings):
            x = self._blocks(x, start, stop)
            x = module(x, batch['eye'], batch['dt'], batch['groups'])
            start = stop
        x = self._blocks(x, start, len(self.encoder.blocks))
        return self.encoder.norm(x)

    def forward(self, batch):
        tokens = compact_tokens(self.features(batch), self.grid)
        tokens = self.embedding(tokens, batch['eye'], batch['dt'])
        logits = [self.head(self.read(tokens[a:b])).squeeze(-1) for a, b in batch['groups']]
        return torch.stack(logits).float()
