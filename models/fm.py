import json
from collections import Counter
import numpy as np, pandas as pd, scipy.sparse as sp, torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def build_item_meta(items, n_store=100, n_cat=100, n_price=8, svd_dim=32, seed=42):
    """Метапризнаки объектов (строки в порядке items). Возвращает (Xm, n_attr).
    Первые n_attr столбцов (store, категории, цена) используются и в профиле пользователя."""
    store = items.store.astype(object)
    top = store.value_counts().head(n_store).index
    S = pd.get_dummies(store.where(store.isin(top), "other").fillna("other")).values.astype(np.float32)

    cats = [json.loads(c) for c in items.categories]
    vocab = {c: j for j, (c, _) in enumerate(Counter(c for l in cats for c in l).most_common(n_cat))}
    C = np.zeros((len(items), len(vocab)), np.float32)
    for r, l in enumerate(cats):
        for c in l:
            if c in vocab: C[r, vocab[c]] = 1

    p = items.price.values.astype(float)
    bins = np.digitize(p, np.nanquantile(p, np.linspace(0, 1, n_price + 1)[1:-1]))
    P = np.zeros((len(items), n_price + 1), np.float32)
    P[np.arange(len(items)), np.where(np.isnan(p), n_price, bins)] = 1      

    text = (items.title.astype(str) + " " + items.features.apply(lambda x: " ".join(json.loads(x)))
            + " " + items.description.apply(lambda x: " ".join(json.loads(x))))
    T = TfidfVectorizer(max_features=20000, min_df=3, stop_words="english").fit_transform(text)
    T = TruncatedSVD(svd_dim, random_state=seed).fit_transform(T).astype(np.float32)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9

    attr = np.hstack([S, C, P])
    return np.hstack([attr, T]), attr.shape[1]


class FM:
    """Факторизационная машина с метапризнаками.
    Признаки пары (u, i): объект - ID + store/категории/цена/текст (SVD); пользователь - профиль по метапризнакам
    купленных ранее объектов, log(число покупок), средняя оценка, log(дней с последней покупки).
    Все признаки пользователя считаются только по истории ДО момента рекомендации.
    Слагаемые, зависящие только от пользователя, не влияют на ранжирование, поэтому
    score(u,i) = b_i + s_u · s_i,  s = sum_f x_f v_f,  b_i = w_i + 0.5(||s_i||^2 - sum_f x_f^2 ||v_f||^2)."""

    def __init__(self, d=32, lr=1e-3, reg=1e-5, epochs=20, bs=4096, n_neg=4, rel=4,
                 seed=42, device="cuda"):
        self.d, self.lr, self.reg, self.epochs, self.bs = d, lr, reg, epochs, bs
        self.n_neg, self.rel, self.seed, self.device = n_neg, rel, seed, device

    # признаки пользователя
    def _user_feats(self, H, k, sumr, gap_ms):
        dev, out = self.device, []
        for s in range(0, H.shape[0], 8192):
            Hb = torch.tensor(H[s:s + 8192].toarray(), device=dev)
            kb = torch.tensor(k[s:s + 8192], device=dev, dtype=torch.float32)
            prof = (Hb @ self.A) / kb.clamp(min=1)[:, None]
            rb = torch.tensor(sumr[s:s + 8192], device=dev, dtype=torch.float32) / kb.clamp(min=1)
            gb = torch.tensor(gap_ms[s:s + 8192] / 86_400_000, device=dev, dtype=torch.float32)
            out.append(torch.cat([prof, torch.stack([torch.log1p(kb) / 5, rb / 5, torch.log1p(gb) / 8], 1)], 1))
        return torch.cat(out)

    def _train_samples(self, d):
        """Для каждой строки train: профиль и статистики по более ранним взаимодействиям того же пользователя."""
        u, i, r, t = d.u.values, d.i.values, d.rating.values.astype(float), d.timestamp.values.astype(float)
        order = np.argsort(u, kind="stable")  # внутри пользователя порядок по времени
        k, sumr, gap = np.zeros(len(d)), np.zeros(len(d)), np.zeros(len(d))
        rows, cols = [], []
        starts = np.r_[0, np.flatnonzero(np.diff(u[order])) + 1, len(d)]
        for a, b in zip(starts[:-1], starts[1:]):
            idx = order[a:b]; m = len(idx)
            rr, cc = np.tril_indices(m, -1)
            rows.append(idx[rr]); cols.append(i[idx][cc])
            k[idx] = np.arange(m); sumr[idx] = np.cumsum(r[idx]) - r[idx]
            gap[idx[1:]] = t[idx[1:]] - t[idx[:-1]]
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        H = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(d), self.n_items))
        return self._user_feats(H, k, sumr, gap)

    def _eval_feats(self, hist, now):
        g = hist.groupby("u").agg(k=("i", "size"), sumr=("rating", "sum"), last=("timestamp", "max"))
        k, sumr, gap = (np.zeros(self.n_users) for _ in range(3))
        k[g.index], sumr[g.index], gap[g.index] = g.k, g.sumr, now - g["last"]
        H = sp.csr_matrix((np.ones(len(hist), np.float32), (hist.u.values, hist.i.values)),
                          shape=(self.n_users, self.n_items))
        return self._user_feats(H, k, sumr, gap)

    # модель
    def _item_repr(self, idx):
        x = self.Xm[idx]
        s = x @ self.Vm + self.E[idx]
        q = (x ** 2) @ (self.Vm ** 2).sum(1) + (self.E[idx] ** 2).sum(1)
        b = x @ self.wm + self.wid[idx] + 0.5 * ((s ** 2).sum(1) - q)
        return s, b

    def fit(self, hist, Xm, n_attr, n_users, n_items):
        """hist: DataFrame [u, i, rating, timestamp] (train или train+val). Xm: матрица метапризнаков объектов."""
        dev = self.device
        torch.manual_seed(self.seed)
        self.n_users, self.n_items = n_users, n_items
        self.Xm = torch.tensor(Xm, device=dev)
        self.A = self.Xm[:, :n_attr].clone()
        d = hist.sort_values("timestamp", kind="stable").reset_index(drop=True)

        Xu = self._train_samples(d)
        pos = torch.tensor(np.flatnonzero(d.rating.values >= self.rel), device=dev)
        pos_i = torch.tensor(d.i.values, device=dev)

        Fu, Fm, init = Xu.shape[1], self.Xm.shape[1], lambda *s: (torch.randn(*s, device=dev) * 0.01).requires_grad_()
        self.Vu, self.Vm, self.E = init(Fu, self.d), init(Fm, self.d), init(n_items, self.d)
        self.wm = torch.zeros(Fm, device=dev, requires_grad=True)
        self.wid = torch.zeros(n_items, device=dev, requires_grad=True)
        opt = torch.optim.Adam([self.Vu, self.Vm, self.E, self.wm, self.wid], lr=self.lr, weight_decay=self.reg)

        self.losses = []
        for ep in range(self.epochs):
            perm = pos[torch.randperm(len(pos), device=dev)]
            tot = 0.0
            for s in range(0, len(perm), self.bs):
                b = perm[s:s + self.bs]
                su = Xu[b] @ self.Vu                                            
                sp_, bp = self._item_repr(pos_i[b])
                neg = torch.randint(0, n_items, (len(b), self.n_neg), device=dev)
                sn, bn = self._item_repr(neg.flatten())
                pos_sc = bp + (su * sp_).sum(1)
                neg_sc = bn.view(-1, self.n_neg) + (su[:, None, :] * sn.view(-1, self.n_neg, self.d)).sum(2)
                loss = -F.logsigmoid(pos_sc[:, None] - neg_sc).mean()            
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(b)
            self.losses.append(tot / len(pos))

        with torch.no_grad():
            self.S_i, self.b_i = self._item_repr(torch.arange(n_items, device=dev))
            self.Xu_eval = self._eval_feats(hist, now=hist.timestamp.max()) @ self.Vu  # s_u на момент рекомендации
        return self

    @torch.no_grad()
    def score(self, users):
        return self.Xu_eval[users] @ self.S_i.T + self.b_i