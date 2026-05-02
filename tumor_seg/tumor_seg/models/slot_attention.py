import torch
from torch import nn


class SlotAttention(nn.Module):
    """Minimal Slot Attention (Locatello et al. 2020).

    Returns:
        slots [B, K, D]
        attn  [B, N, K]   (post-softmax over slots, normalized over tokens)
    """

    def __init__(self, num_slots: int = 2, slot_dim: int = 256,
                 n_iters: int = 3, hidden_dim: int = 512, eps: float = 1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.n_iters = n_iters
        self.eps = eps
        self.scale = slot_dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, slot_dim) * 0.02)
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, num_slots, slot_dim))

        self.norm_input = nn.LayerNorm(slot_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_pre_ff = nn.LayerNorm(slot_dim)

        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_k = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(slot_dim, slot_dim, bias=False)

        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, slot_dim),
        )

    def forward(self, x: torch.Tensor):
        B, N, _ = x.shape
        K, D = self.num_slots, self.slot_dim

        mu = self.slots_mu.expand(B, K, D)
        if self.training:
            sigma = self.slots_log_sigma.exp().expand(B, K, D)
            slots = mu + sigma * torch.randn_like(mu)
        else:
            slots = mu.clone()

        x = self.norm_input(x)
        k = self.to_k(x)
        v = self.to_v(x)

        for _ in range(self.n_iters):
            slots_prev = slots
            slots_n = self.norm_slots(slots)
            q = self.to_q(slots_n)

            dots = torch.einsum("bnd,bkd->bnk", k, q) * self.scale
            attn = dots.softmax(dim=-1) + self.eps
            attn_norm = attn / attn.sum(dim=1, keepdim=True)
            updates = torch.einsum("bnd,bnk->bkd", v, attn_norm)

            slots = self.gru(updates.reshape(-1, D), slots_prev.reshape(-1, D)).reshape(B, K, D)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots, attn
