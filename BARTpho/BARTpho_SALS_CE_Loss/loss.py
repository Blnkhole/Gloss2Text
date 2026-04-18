import torch
import torch.nn as nn
import torch.nn.functional as F

class SALSLoss(nn.Module):
    def __init__(self, sim_matrix, token_id_map, pad_idx, alpha=0.1):
        super().__init__()

        self.register_buffer("sim_matrix", sim_matrix.clone().detach().float())

        self.alpha = alpha
        self.pad_idx = pad_idx
        self.eps = 1e-8
        self.top_k = 3
        
        topk = torch.topk(sim_matrix, k=self.top_k + 1, dim=-1)
        topk_indices = topk.indices[:, 1:]

        self.register_buffer("topk_indices", topk_indices)

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
        
        #safe_labels = labels_flat.clamp(0, self.token_mapping.size(0) - 1)

        sim_indices = torch.full_like(labels_flat, -1)
        valid_positions = (labels_flat >= 0) & (labels_flat < self.token_mapping.size(0))

        if valid_positions.any():
            valid_labels = labels_flat[valid_positions].long()

            # clamp để tránh out-of-bound
            valid_labels = torch.clamp(valid_labels, 0, self.token_mapping.size(0) - 1)

            token_mapping = self.token_mapping.to(device)

            sim_indices[valid_positions] = token_mapping[valid_labels]

        #one-hot
        target_dist = torch.zeros_like(log_probs)

        safe_labels = labels_flat.clone()
        safe_labels[~valid_mask] = 0

        valid_indices = torch.where(valid_mask)[0]
        target_dist[valid_indices, safe_labels[valid_mask]] = 1.0 - self.alpha

        final_mask = valid_mask & (sim_indices != -1)

        """
        if final_mask.any():
            rows = torch.where(final_mask)[0].unsqueeze(1)
            cols = self.used_token_ids.unsqueeze(0)

            sim_vectors = self.sim_matrix[sim_indices[final_mask]]
            target_dist[rows, cols] += self.alpha * sim_vectors
        """

        #Use top-k sim
        if final_mask.any():

            valid_rows = torch.where(final_mask)[0]
            sim_ids = sim_indices[final_mask]  # [N]

            topk_ids = self.topk_indices[sim_ids]   # [N, k]
            topk_token_ids = self.simidx2token[topk_ids]

            gt = safe_labels[final_mask].unsqueeze(1)
            topk_token_ids[topk_token_ids == gt] = -1

            topk_sims = self.sim_matrix[sim_ids].gather(1, topk_ids)  # [N, k]

            temperature = 0.3
            topk_sims = torch.softmax(topk_sims / temperature, dim=-1)

            valid_topk_mask = topk_token_ids != -1

            target_dist[
              valid_rows.unsqueeze(1).expand_as(topk_token_ids)[valid_topk_mask],
              topk_token_ids[valid_topk_mask]
              ] += self.alpha * topk_sims[valid_topk_mask]
        
        target_dist = target_dist / (target_dist.sum(dim=-1, keepdim=True) + 1e-8)
        
        loss = -(target_dist * log_probs).sum(dim=-1)
        loss = loss[valid_mask].mean()

        return loss