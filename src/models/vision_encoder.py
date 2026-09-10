import os

import torch.nn as nn
import torch.nn.functional as F
import timm


class VisionEncoder(nn.Module):
    """Image tower: a timm backbone plus a non-linear projection head.

    Swin backbones expect 224x224 inputs, so 300x300 inputs are rescaled with
    bicubic interpolation to preserve bond geometry.
    """

    def __init__(self, embed_dim=512, backbone_type='resnet50'):
        super().__init__()
        self.backbone_type = backbone_type
        self.is_swin = 'swin' in backbone_type

        try:
            if backbone_type in ['resnet50', 'resnet101']:
                self.backbone = timm.create_model(backbone_type, pretrained=True,
                                                  num_classes=0)
                in_features = self.backbone.num_features

            elif backbone_type == 'convnext_base':
                ckpt = './pretrained_weights/convnext_base.safetensors'
                if os.path.exists(ckpt):
                    self.backbone = timm.create_model('convnext_base', pretrained=False,
                                                      checkpoint_path=ckpt)
                else:
                    self.backbone = timm.create_model('convnext_base', pretrained=True)
                self.backbone.reset_classifier(0)
                self.backbone.set_grad_checkpointing(enable=True)
                in_features = self.backbone.num_features

            elif backbone_type == 'swin_base':
                ckpt = './pretrained_weights/swin_base.safetensors'
                if os.path.exists(ckpt):
                    self.backbone = timm.create_model('swin_base_patch4_window7_224',
                                                      pretrained=False,
                                                      pretrained_strict=False,
                                                      checkpoint_path=ckpt)
                else:
                    self.backbone = timm.create_model('swin_base_patch4_window7_224',
                                                      pretrained=True)
                self.backbone.reset_classifier(0)
                self.backbone.set_grad_checkpointing(enable=True)
                in_features = self.backbone.num_features

            else:
                raise NotImplementedError(f"Backbone {backbone_type} is not implemented")

        except Exception as e:
            raise RuntimeError(f"Failed to load backbone {backbone_type}: {e}")

        self.projection = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, embed_dim),
        )

    def forward(self, x):
        if self.is_swin and x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
        features = self.backbone(x)
        return self.projection(features)
