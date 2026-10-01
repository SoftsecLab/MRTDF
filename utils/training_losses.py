import torch
import torch.nn.functional as F


def style_triplet_consistency_loss(initial_text, target_text, target_ids,
                                   margin_neutral, margin_target, margin_human, pool_size):
    batch = initial_text.size(0)
    if batch < 2 or pool_size < 1:
        raise ValueError('STC-Loss requires at least two samples and a positive pool size.')
    indices = torch.arange(batch, device=initial_text.device)
    positives, negatives = [], []
    for i in range(batch):
        candidates = torch.cat([indices[:i], indices[i + 1:]])
        pool = candidates[torch.randperm(len(candidates), device=indices.device)[:pool_size]]
        positives.append(pool[torch.randint(len(pool), (1,), device=indices.device)])
        negative = None
        for _ in range(10):
            pool = candidates[torch.randperm(len(candidates), device=indices.device)[:pool_size]]
            for candidate in pool:
                if target_ids[i] != target_ids[candidate.item()]:
                    negative = candidate.reshape(1)
                    break
            if negative is not None:
                break
        negatives.append(indices[i:i + 1] if negative is None else negative)
    positive = initial_text[torch.cat(positives)]
    negative = target_text[torch.cat(negatives)]
    neutral = F.relu(F.pairwise_distance(initial_text, positive) - margin_neutral)
    target = F.relu(margin_target - F.pairwise_distance(initial_text, target_text))
    human = F.relu(margin_human - F.pairwise_distance(target_text, negative))
    return (neutral + target + human).mean()


def variance_regularization_loss(z, eps=1e-6):
    return F.relu(0.5 - torch.var(z, dim=0, unbiased=z.size(0) > 1) + eps).mean()


def pullaway_loss(embeddings):
    if embeddings.size(0) <= 1:
        return embeddings.sum() * 0
    normalized = F.normalize(embeddings, p=2, dim=1)
    similarity = normalized @ normalized.T
    mask = ~torch.eye(len(embeddings), device=embeddings.device, dtype=torch.bool)
    return similarity[mask].square().mean()
