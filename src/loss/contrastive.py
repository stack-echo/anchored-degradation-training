import torch
import torch.nn as nn
import torch.nn.functional as F


class HardNegativeInfoNCE(nn.Module):
    """Symmetric InfoNCE over image--graph pairs with optional
    Tanimoto-based hard-negative reweighting.

    Negatives whose Morgan-fingerprint Tanimoto similarity to the query
    exceeds `hard_neg_margin` are pushed away with an extra log-weight.
    With the default margin of 1.1 no pair qualifies (Tanimoto <= 1.0), so
    the objective reduces to plain symmetric InfoNCE; this default
    configuration is what the paper's models use. The temperature is
    overwritten every step from the model's learnable logit scale.
    """

    def __init__(self, temperature=0.07, hard_neg_margin=1.1, push_weight=1.0):
        super().__init__()
        self.temperature = temperature
        self.hard_neg_margin = hard_neg_margin
        self.push_weight = push_weight

    def tanimoto_matrix(self, fp_batch):
        """Pairwise Tanimoto similarity of a batch of binary fingerprints."""
        fp_batch = fp_batch.float()
        intersection = torch.mm(fp_batch, fp_batch.t())
        sum_fp = fp_batch.sum(dim=1).unsqueeze(1)
        union = sum_fp + sum_fp.t() - intersection
        return intersection / (union + 1e-8)

    def forward(self, img_features, graph_features, fp_batch):
        img_features = F.normalize(img_features, dim=-1)
        graph_features = F.normalize(graph_features, dim=-1)

        logits = torch.matmul(img_features, graph_features.t()) / self.temperature

        tanimoto_sim = self.tanimoto_matrix(fp_batch)
        batch_size = img_features.shape[0]
        mask_off_diag = ~torch.eye(batch_size, dtype=torch.bool, device=logits.device)
        hard_neg_mask = mask_off_diag & (tanimoto_sim > self.hard_neg_margin)

        # Apply the extra push as an additive log-weight on the logits
        weight_matrix = torch.ones_like(logits)
        weight_matrix[hard_neg_mask] = self.push_weight
        modified_logits = logits + torch.log(weight_matrix)

        labels = torch.arange(batch_size, dtype=torch.long, device=logits.device)
        loss_i2g = F.cross_entropy(modified_logits, labels)
        loss_g2i = F.cross_entropy(modified_logits.t(), labels)
        return (loss_i2g + loss_g2i) / 2.0
