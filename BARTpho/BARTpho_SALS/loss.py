import torch
import torch.nn as nn
import torch.nn.functional as F

class SALSLoss(nn.Module):
    def __init__(self, sim_matrix, token_id_map, pad_idx, alpha=0.2, eps = 0.1):
        super().__init__()

        self.register_buffer("sim_matrix", sim_matrix.clone().detach().float())

        self.alpha = alpha
        self.pad_idx = pad_idx
        self.eps = eps

        max_id = max(token_id_map.keys())
        mapping = torch.full((max_id + 1,), -1, dtype=torch.long)
        for tok_id, sim_idx in token_id_map.items():
            mapping[tok_id] = sim_idx
        
        self.register_buffer("token_mapping", mapping)
        self.register_buffer("used_token_ids", torch.tensor(list(token_id_map.keys()), dtype=torch.long))

        # reverse mapping
        rev_map = torch.full((sim_matrix.size(0),), -1, dtype=torch.long)
        for tok_id, sim_idx in token_id_map.items():
            rev_map[sim_idx] = tok_id

        self.register_buffer("simidx2token", rev_map)

    def forward(self, logits, labels):
        device = logits.device
        B, T, V = logits.shape
        log_probs = F.log_softmax(logits, dim=-1).view(-1, V)
        labels_flat = labels.view(-1)

        valid_mask = (labels_flat != -100) & (labels_flat != self.pad_idx)
        safe_labels = labels_flat.clone()
        safe_labels[~valid_mask] = 0

        valid_indices = torch.where(valid_mask)[0]
        
        #safe_labels = labels_flat.clamp(0, self.token_mapping.size(0) - 1)

        valid_positions = (labels_flat >= 0) & (labels_flat < self.token_mapping.size(0))

        #one-hot
        one_hot = torch.zeros_like(log_probs)
        one_hot[valid_indices, safe_labels[valid_mask]] = 1.0

        #uniform
        uniform_dist = torch.zeros_like(log_probs)
        uniform_dist[:, self.used_token_ids] = 1.0 / len(self.used_token_ids)

        #similarity
        sim_dist = torch.zeros_like(log_probs)

        #map token
        sim_indices = torch.full_like(labels_flat, -1)

        valid_positions = (labels_flat >= 0) & (labels_flat < self.token_mapping.size(0))
        if valid_positions.any():
            valid_labels = labels_flat[valid_positions].long()
            valid_labels = torch.clamp(valid_labels, 0, self.token_mapping.size(0) - 1)

            sim_indices[valid_positions] = self.token_mapping[valid_labels]

        final_mask = valid_mask & (sim_indices != -1)

        if final_mask.any():
            sim_ids = sim_indices[final_mask]
            rows = torch.where(final_mask)[0]

            sim_vectors = torch.clamp(self.sim_matrix[sim_ids], min=0)
            sim_vectors = sim_vectors / (sim_vectors.sum(dim=-1, keepdim=True) + 1e-8)

            sim_dist[rows[:, None], self.used_token_ids[None, :]] = sim_vectors
                
        target_dist = (
            (1 - self.alpha - self.eps) * one_hot
            + self.alpha * sim_dist
            + self.eps * uniform_dist
        )

        target_dist = target_dist / (target_dist.sum(dim=-1, keepdim=True) + 1e-8)
                
        loss = -(target_dist * log_probs).sum(dim=-1)
        loss = loss[valid_mask].mean()

        return loss