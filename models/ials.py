import numpy as np
import torch


class IALS:
    """Implicit ALS с батчевым решением нормальных уравнений на GPU."""

    def __init__(self, factors=64, reg=0.1, alpha=20.0, iters=15, seed=42, device="cuda", bs=2048):
        self.d, self.reg, self.alpha, self.iters = factors, reg, alpha, iters
        self.seed, self.device, self.bs = seed, device, bs

    def _solve(self, Y, C):
        """Обновляет факторы для строк C (csr: confidence-1 в данных) при фиксированных Y."""
        d, dev = self.d, self.device
        YtY = Y.T @ Y + self.reg * torch.eye(d, device=dev)
        X = torch.empty(C.shape[0], d, device=dev)
        for s in range(0, C.shape[0], self.bs):
            blk = C[s:s + self.bs].tocoo()
            r, c = torch.as_tensor(blk.row, device=dev).long(), torch.as_tensor(blk.col, device=dev).long()
            w = torch.as_tensor(blk.data, device=dev, dtype=torch.float32)       
            Yc = Y[c]                                                            
            n = C[s:s + self.bs].shape[0]
            A = YtY.expand(n, d, d).clone()
            A.index_add_(0, r, (w[:, None, None] * Yc[:, :, None] * Yc[:, None, :]))
            b = torch.zeros(n, d, device=dev).index_add_(0, r, (w + 1)[:, None] * Yc)   
            X[s:s + n] = torch.linalg.solve(A, b[:, :, None])[:, :, 0]
        return X

    def fit(self, R, weights=None):
        """R: csr бинарная матрица положительных взаимодействий [n_users, n_items].
        weights: опционально csr с теми же позициями (значения > 0) вместо 1 — используется в кастомном таргете."""
        g = torch.Generator(device=self.device).manual_seed(self.seed)
        n_u, n_i = R.shape
        self.U = torch.randn(n_u, self.d, device=self.device, generator=g) * 0.01
        self.V = torch.randn(n_i, self.d, device=self.device, generator=g) * 0.01
        P = R.tocsr().astype(np.float32)
        P.data = (weights.tocsr()[P.nonzero()].A1 if weights is not None else np.ones_like(P.data))
        C = P.copy(); C.data = self.alpha * C.data           
        CT = C.T.tocsr()
        for _ in range(self.iters):
            self.U = self._solve(self.V, C)
            self.V = self._solve(self.U, CT)
        return self

    @torch.no_grad()
    def score(self, users):
        return self.U[users] @ self.V.T