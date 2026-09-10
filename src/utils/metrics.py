import torch


def calculate_retrieval_metrics(img_embeds, graph_embeds):
    """Image-to-graph (I2G) and graph-to-image (G2I) retrieval metrics.

    Inputs must already be L2-normalized. Returns R@{1,5,10} and MRR for
    both directions.
    """
    with torch.no_grad():
        sim_matrix = img_embeds @ graph_embeds.t()

        metrics = {}
        metrics.update(get_recall_at_k(sim_matrix, prefix="I2G_"))
        metrics.update(get_recall_at_k(sim_matrix.t(), prefix="G2I_"))
        return metrics


def get_recall_at_k(sim_matrix, prefix=""):
    """Top-k recall and MRR over the diagonal (paired) ground truth."""
    num_queries = sim_matrix.shape[0]
    _, indices = sim_matrix.topk(10, dim=1, largest=True, sorted=True)

    target = torch.arange(num_queries, device=sim_matrix.device).view(-1, 1)
    correct = (indices == target)

    metrics = {}
    for k in [1, 5, 10]:
        hits = correct[:, :k].sum().float()
        metrics[f"{prefix}R@{k}"] = (hits / num_queries).item() * 100.0

    ranks = correct.nonzero()[:, 1] + 1
    metrics[f"{prefix}MRR"] = (1.0 / ranks.float()).mean().item()
    return metrics
