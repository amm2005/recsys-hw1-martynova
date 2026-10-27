import numpy as np
import torch


class ItemKNN:
    """Item-based kNN: cosine-сходство со shrinkage, top-k соседей, взвешенная сумма по истории."""

    def __init__(self, k=50, shrink=10.0, alpha=0.5, device="cuda"):
        self.k, self.shrink, self.alpha, self.device = k, shrink, alpha, device

    def fit(self, R):
        """R: scipy csr [n_users, n_items], бинарная матрица положительных взаимодействий."""
        R = torch.tensor(R.toarray(), dtype=torch.float32, device=self.device)
        cnt = R.sum(0)
        co = R.T @ R  # число общих пользователей
        norm = cnt[:, None] ** self.alpha * cnt[None, :] ** (1 - self.alpha)
        S = co / (norm + self.shrink)                             
        S.fill_diagonal_(0)
        vals, idx = S.topk(self.k, dim=1)  # k лучших соседей каждого объекта
        W = torch.zeros_like(S).scatter_(1, idx, vals)  # W[i, j]: сходство i с соседом j
        self.R, self.W_T = R, W.T.contiguous()
        return self

    @torch.no_grad()
    def score(self, users):
        """users: LongTensor глобальных id. Возвращает [len(users), n_items]."""
        return self.R[users] @ self.W_T  # score(u,i) = sum_{j in H_u} s_ij, j in N(i)