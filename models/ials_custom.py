import numpy as np
import scipy.sparse as sp
import torch


class IALSCustom:
    """iALS с кастомным целевым сигналом (graded preference + явные негативы).

    Для каждой наблюдаемой пары (u, i) из истории:
      b_u: сглаженное смещение пользователя (оценки без смещения объекта), r~ = r - b_u
      p_ui = sigmoid(gamma * (r~_ui - tau))              # целевая «степень предпочтения» в [0, 1]
      c_ui = 1 + alpha * w_t * |2 p_ui - 1|              # уверенность: сильное мнение (5 или 1-2) весит больше
      w_t  = max(2^(-age / half_life), w_min)            # затухание по возрасту взаимодействия
    Ненаблюдаемые пары: p = 0, c = 1.
    Потери: sum c_ui (p_ui - x_u^T y_i)^2 + reg (||X||^2 + ||Y||^2).
    """

    def __init__(self, factors=64, reg=0.1, alpha=20.0, iters=15, tau=3.5, gamma=2.0,
                 half_life=730.0, w_min=0.2, lam_b=5.0, debias=True,
                 seed=42, device="cuda", bs=1024):
        self.d, self.reg, self.alpha, self.iters = factors, reg, alpha, iters
        self.tau, self.gamma, self.half_life, self.w_min = tau, gamma, half_life, w_min
        self.lam_b, self.debias = lam_b, debias
        self.seed, self.device, self.bs = seed, device, bs

    # целевой сигнал
    def build_target(self, hist, n_users, n_items):
        u, i = hist.u.values, hist.i.values
        r = hist.rating.values.astype(np.float64)
        bu = np.zeros(n_users)
        if self.debias:
            mu, bi = r.mean(), np.zeros(n_items)
            cu, ci = np.bincount(u, minlength=n_users), np.bincount(i, minlength=n_items)
            for _ in range(5):  # чередующиеся сглаженные средние
                bi = np.bincount(i, r - mu - bu[u], minlength=n_items) / (ci + self.lam_b)
                bu = np.bincount(u, r - mu - bi[i], minlength=n_users) / (cu + self.lam_b)
        rt = r - bu[u]  # оценка без смещения пользователя
        p = 1.0 / (1.0 + np.exp(-self.gamma * (rt - self.tau)))
        age = (hist.timestamp.max() - hist.timestamp.values) / 86_400_000
        wt = np.maximum(0.5 ** (age / self.half_life), self.w_min) if self.half_life else np.ones(len(r))
        c1 = self.alpha * wt * np.abs(2 * p - 1)                         
        shape = (n_users, n_items)
        P = sp.csr_matrix((p.astype(np.float32), (u, i)), shape=shape)
        C = sp.csr_matrix((c1.astype(np.float32), (u, i)), shape=shape)  
        return P, C

    # ALS
    def _solve(self, Y, C, P):
        d, dev = self.d, self.device
        YtY = Y.T @ Y + self.reg * torch.eye(d, device=dev)
        X = torch.zeros(C.shape[0], d, device=dev)
        for s in range(0, C.shape[0], self.bs):
            cb, pb = C[s:s + self.bs].tocoo(), P[s:s + self.bs].tocoo()
            n = cb.shape[0]
            r = torch.as_tensor(cb.row, device=dev).long()
            c = torch.as_tensor(cb.col, device=dev).long()
            w = torch.as_tensor(cb.data, device=dev, dtype=torch.float32)      
            p = torch.as_tensor(pb.data, device=dev, dtype=torch.float32)      
            Yc = Y[c]
            A = YtY.expand(n, d, d).clone()
            A.index_add_(0, r, w[:, None, None] * Yc[:, :, None] * Yc[:, None, :])
            b = torch.zeros(n, d, device=dev).index_add_(0, r, ((w + 1) * p)[:, None] * Yc)   
            X[s:s + n] = torch.linalg.solve(A, b[:, :, None])[:, :, 0]
        return X

    def fit(self, hist, n_users, n_items):
        """hist: DataFrame [u, i, rating, timestamp] (train или train+val), все оценки 1-5."""
        g = torch.Generator(device=self.device).manual_seed(self.seed)
        self.U = torch.randn(n_users, self.d, device=self.device, generator=g) * 0.01
        self.V = torch.randn(n_items, self.d, device=self.device, generator=g) * 0.01
        P, C = self.build_target(hist, n_users, n_items)
        PT, CT = P.T.tocsr(), C.T.tocsr()
        for _ in range(self.iters):
            self.U = self._solve(self.V, C, P)
            self.V = self._solve(self.U, CT, PT)
        return self

    @torch.no_grad()
    def score(self, users):
        return self.U[users] @ self.V.T