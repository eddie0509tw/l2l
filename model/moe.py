import torch.nn.functional as F
import torch
from torch import nn

# @TODO: Implement the Mixture of experts layer
class Expert(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    
# @TODO: Implement the Sparsegen
class MoeLayer(nn.Module):
    def __init__(self, embed_size, n_exp, top_k):
        super().__init__()

        self.n_exp = n_exp
        self.top_k = top_k
        self.heads = nn.ModuleList([Expert() for _ in range(n_exp)])
        self.gate = nn.Linear(embed_size, n_exp)

    def forward(self, x):
        """
        x: Input tensor of shape (bz, hidden_size, embed_size)
        """
        # Compute gating scores and select top-k experts
        gate_out = self.gate(x)  # Shape: (bz, hidden_size, n_exp)
        top_k_values, top_k_indices = torch.topk(gate_out, k=self.top_k, dim=-1)  # Shape: (bz, hidden_size, top_k)
        probs = F.softmax(top_k_values, dim=-1)  # Shape: (bz, hidden_size, top_k)
        expert_outputs = torch.stack([expert(x) for expert in self.heads], dim=-1)  # Shape: (bz, hidden_size, embed_size, n_exp)
        # Select the experts indicated by top_k_indices
        selected_expert_outputs = expert_outputs.gather( # Shape: (bz, hidden_size, embed_size, top_k)
            -1, top_k_indices.unsqueeze(-2).expand(-1, -1, expert_outputs.shape[2], -1))  
        weighted_expert_outputs = selected_expert_outputs * probs.unsqueeze(-2)  
        # Summing across the expert dimension (-1)
        out = weighted_expert_outputs.sum(dim=-1)  # Shape: (bz, hidden_size, embed_size)

        return out