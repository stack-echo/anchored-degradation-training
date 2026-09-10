import torch.nn as nn


class GraphEncoder(nn.Module):
    """Graph tower: encodes a precomputed structural representation.

    - fastrp: 128-dimensional 3-order FastRP embedding of the molecule--scaffold
      graph (computed offline by src/data_pipeline/generate_fastrp.py)
    - morgan: 2048-bit Morgan fingerprint (radius 2); LazyLinear infers the
      input dimension
    """

    def __init__(self, in_dim=128, embed_dim=512, encoder_type='fastrp'):
        super().__init__()
        self.encoder_type = encoder_type

        if encoder_type == 'fastrp':
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Linear(256, embed_dim),
            )
        elif encoder_type == 'morgan':
            self.encoder = nn.Sequential(
                nn.LazyLinear(512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Linear(512, embed_dim),
            )
        else:
            raise ValueError(f"Unsupported graph encoder: {encoder_type}")

    def forward(self, graph_input):
        return self.encoder(graph_input)
