import numpy as np
import torch


class EASE:
    """EASE: линейная авто-энкодер модель item-item с замкнутым решением."""

    def __init__(self, lam=500.0, device="cuda"):
        self.lam, self.device = lam, device

    def fit(self, R):
        """R: scipy csr [n_users, n_items], бинарные положительные взаимодействия."""
        X = torch.tensor(R.toarray(), dtype=torch.float64, device=self.device)
        G = X.T @ X + self.lam * torch.eye(X.shape[1], dtype=torch.float64, device=self.device)
        P = torch.linalg.inv(G)
        B = -P / torch.diag(P)[None, :]  # B_ij = -P_ij / P_jj
        B.fill_diagonal_(0)            
        self.X, self.B = X.float(), B.float()
        return self

    @torch.no_grad()
    def score(self, users):
        return self.X[users] @ self.B