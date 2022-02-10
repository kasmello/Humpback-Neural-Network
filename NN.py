import torch
from vit_pytorch import ViT

v = ViT(
    image_size = 1167,
    patch_size = 389,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)
