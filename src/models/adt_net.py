import torch
import torch.nn as nn
import numpy as np

from .vision_encoder import VisionEncoder
from .graph_encoder import GraphEncoder


class ADTNet(nn.Module):
    """Two-tower image--graph retrieval model.

    The image tower receives (possibly degraded) molecular depictions; the
    graph tower always receives the clean structural embedding. Trained with
    symmetric InfoNCE and a learnable logit scale initialized as in CLIP
    (equivalent to an initial temperature of 0.07).

    Note: checkpoints saved by earlier internal versions of this model load
    unchanged -- the module tree (vision_encoder / graph_encoder /
    logit_scale) is identical.
    """

    def __init__(self, vision_backbone='resnet50', graph_type='fastrp',
                 graph_in_dim=128, embed_dim=512):
        super().__init__()
        self.vision_encoder = VisionEncoder(embed_dim=embed_dim,
                                            backbone_type=vision_backbone)
        self.graph_encoder = GraphEncoder(in_dim=graph_in_dim, embed_dim=embed_dim,
                                          encoder_type=graph_type)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image, graph_feat):
        img_embed = self.vision_encoder(image)
        graph_embed = self.graph_encoder(graph_feat)
        return img_embed, graph_embed, self.logit_scale.exp()
