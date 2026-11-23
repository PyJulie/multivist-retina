"""RETFound-DINOv2 ViT-Large encoder for colour fundus photographs.

The weights are not distributed here; obtain the RETFound-DINOv2 checkpoint from the RETFound
authors and pass its path. With `weights=None` the architecture is built without pretrained
weights, which is enough to load a trained MVT checkpoint.
"""
from __future__ import annotations

import timm
import torch
from timm.layers import resample_abs_pos_embed

ARCH = 'vit_large_patch14_dinov2.lvd142m'
PATCH = 14


def _state_dict(path):
    sd = torch.load(path, map_location='cpu')
    # DINOv2-style checkpoints wrap the weights, e.g. {'teacher': {'backbone.blocks...': ...}}
    for key in ('teacher', 'student', 'model', 'state_dict'):
        if isinstance(sd, dict) and key in sd and len(sd) < 20 and hasattr(sd[key], 'items'):
            sd = sd[key]
            break
    if any(k.startswith('backbone.') for k in sd):
        sd = {k[len('backbone.'):]: v for k, v in sd.items() if k.startswith('backbone.')}
    if all(k.startswith('module.') for k in sd):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    return sd


def retfound_dinov2(weights=None, img_size=224, drop_path_rate=0.):
    """ViT-L/14 at 224 x 224 input. Positional embeddings of a checkpoint trained at another
    resolution (518 x 518 for DINOv2) are resampled to the 16 x 16 grid by bicubic interpolation."""
    model = timm.create_model(ARCH, pretrained=False, num_classes=0, img_size=img_size,
                              dynamic_img_size=False, drop_path_rate=drop_path_rate)
    if weights is None:
        return model
    sd = _state_dict(weights)
    current = model.state_dict()['pos_embed']
    if 'pos_embed' in sd and sd['pos_embed'].shape != current.shape:
        grid = (img_size // PATCH, img_size // PATCH)
        print(f'Interpolating pos_embed {tuple(sd["pos_embed"].shape)} -> {tuple(current.shape)}')
        sd['pos_embed'] = resample_abs_pos_embed(sd['pos_embed'], new_size=grid, interpolation='bicubic',
                                                 num_prefix_tokens=current.shape[1] - grid[0] * grid[1])
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'Loaded encoder weights: {len(missing)} missing and {len(unexpected)} unexpected keys')
    return model
