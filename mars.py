"""MARS — Multivariate Adaptive Regression Splines (Friedman, 1991), своя реализация на numpy.

Модель строит сумму кусочно-линейных «шарнирных» функций  max(0, x - t)  и  max(0, t - x)
(и их попарных произведений при max_degree=2):

  1) прямой проход — жадно добавляем пары шарниров (признак, узел t), которые сильнее всего
     уменьшают сумму квадратов остатков;
  2) обратный проход — по одному удаляем слабейшие функции и выбираем подмодель
     с минимальным критерием обобщённой перекрёстной проверки
         GCV = RSS/n / (1 - C(M)/n)^2,   C(M) = M + penalty·(M-1)/2;
  3) для бинарной цели (дефолт 0/1) поверх отобранного базиса оцениваем логистическую
     регрессию — получаем корректные вероятности в [0, 1] (схема «MARS-logit»,
     её обычно и используют в кредитном скоринге).

Готовую библиотеку py-earth не используем: она заброшена и не ставится на свежий Python.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression


class Hinge:
    __slots__ = ("terms",)

    def __init__(self, terms=()):
        self.terms = tuple(terms)          # кортеж (j, t, sign); sign=+1: max(0,x-t), -1: max(0,t-x)

    def eval(self, X):
        out = np.ones(X.shape[0])
        for j, t, s in self.terms:
            out *= np.maximum(0.0, s * (X[:, j] - t))
        return out

    def vars(self):
        return {j for j, _, _ in self.terms}

    def describe(self, names):
        if not self.terms:
            return "1"
        parts = []
        for j, t, s in self.terms:
            parts.append(f"max(0, {names[j]} − {t:.4g})" if s > 0 else f"max(0, {t:.4g} − {names[j]})")
        return " × ".join(parts)


class MARSClassifier:
    def __init__(self, max_terms=21, max_degree=1, penalty=None, n_knots=20, min_leaf=10,
                 logit_C=1.0, random_state=0):
        self.max_terms = max_terms
        self.max_degree = max_degree
        self.penalty = penalty if penalty is not None else (2.0 if max_degree == 1 else 3.0)
        self.n_knots = n_knots
        self.min_leaf = min_leaf
        self.logit_C = logit_C
        self.random_state = random_state

    # ------------------------------------------------------------------ прямой проход
    def _knots(self, x, parent_mask):
        v = x[parent_mask]
        if v.size < 2 * self.min_leaf:
            return np.array([])
        qs = np.linspace(0, 1, self.n_knots + 2)[1:-1]
        k = np.unique(np.quantile(v, qs))
        # оставляем узлы, по обе стороны которых не меньше min_leaf наблюдений
        lo = np.searchsorted(np.sort(v), k, side="left")
        ok = (lo >= self.min_leaf) & (v.size - lo >= self.min_leaf)
        return k[ok]

    TOL = 1e-7

    def _forward(self, X, y):
        """Прямой проход. Базис B поддерживается полного ранга: новая функция добавляется, только
        если она не выражается через уже выбранные (иначе, например, пара шарниров на втором узле
        того же признака линейно зависима: h⁺(t₂) − h⁻(t₂) = h⁺(t₁) − h⁻(t₁) + (t₁ − t₂)·1).
        Q — ортонормированный базис span(B), обновляется по одному столбцу (Грам–Шмидт ×2)."""
        n, p = X.shape
        basis = [Hinge()]
        B = np.ones((n, 1))
        Q = B / np.sqrt(n)
        r = y - Q @ (Q.T @ y)
        banned = set()
        while len(basis) < self.max_terms:
            best = (1e-12, None)
            for bi, b in enumerate(basis):
                if len(b.terms) >= self.max_degree:
                    continue
                bcol = B[:, bi]
                mask = bcol > 0
                for j in range(p):
                    if j in b.vars():
                        continue
                    knots = self._knots(X[:, j], mask)
                    knots = np.array([k for k in knots if (bi, j, k) not in banned])
                    if knots.size == 0:
                        continue
                    xj = X[:, j][:, None]
                    C1 = bcol[:, None] * np.maximum(0, xj - knots[None, :])
                    C2 = bcol[:, None] * np.maximum(0, knots[None, :] - xj)
                    R1 = C1 - Q @ (Q.T @ C1)
                    R2 = C2 - Q @ (Q.T @ C2)
                    n1 = (C1 * C1).sum(0); n2 = (C2 * C2).sum(0)
                    a11 = (R1 * R1).sum(0); a22 = (R2 * R2).sum(0); a12 = (R1 * R2).sum(0)
                    ok1 = a11 > self.TOL * (n1 + 1e-30); ok2 = a22 > self.TOL * (n2 + 1e-30)
                    b1 = R1.T @ r; b2 = R2.T @ r
                    det = a11 * a22 - a12 ** 2
                    with np.errstate(divide="ignore", invalid="ignore"):
                        g12 = (a22 * b1 ** 2 - 2 * a12 * b1 * b2 + a11 * b2 ** 2) / det
                        g1 = np.where(ok1, b1 ** 2 / a11, 0)
                        g2 = np.where(ok2, b2 ** 2 / a22, 0)
                    full = ok1 & ok2 & (det > self.TOL * (a11 * a22 + 1e-30))
                    gain = np.nan_to_num(np.where(full, g12, np.maximum(g1, g2)))
                    k = int(np.argmax(gain))
                    if gain[k] > best[0]:
                        best = (gain[k], (bi, j, knots[k]))
            if best[1] is None:
                break
            bi, j, t = best[1]
            parent = basis[bi]
            added = 0
            for s in (+1, -1):
                if len(basis) >= self.max_terms:
                    break
                h = Hinge(parent.terms + ((j, t, s),))
                col = h.eval(X)
                v = col - Q @ (Q.T @ col)
                v = v - Q @ (Q.T @ v)                      # повторная ортогонализация
                if np.linalg.norm(v) <= np.sqrt(self.TOL) * (np.linalg.norm(col) + 1e-30):
                    continue                               # линейно зависима — не добавляем
                basis.append(h)
                B = np.column_stack([B, col])
                Q = np.column_stack([Q, v / np.linalg.norm(v)])
                added += 1
            banned.add((bi, j, t))
            if not added:
                continue
            r = y - Q @ (Q.T @ y)
        self.forward_rank_ = int(np.linalg.matrix_rank(B))
        return basis, B

    # ------------------------------------------------------------------ обратный проход
    def _gcv(self, B, y):
        n, m = B.shape
        coef, *_ = np.linalg.lstsq(B, y, rcond=None)
        rss = float(((y - B @ coef) ** 2).sum())
        c = m + self.penalty * (m - 1) / 2
        if c >= n:
            return np.inf
        return rss / n / (1 - c / n) ** 2

    def _backward(self, basis, B, y):
        active = list(range(len(basis)))
        best_set, best_gcv = list(active), self._gcv(B, y)
        while len(active) > 1:
            cand = []
            for k in active[1:]:                     # свободный член не удаляем
                s = [a for a in active if a != k]
                cand.append((self._gcv(B[:, s], y), k))
            g, k = min(cand)
            active.remove(k)
            if g <= best_gcv:
                best_gcv, best_set = g, list(active)
        self.gcv_ = best_gcv
        return best_set

    # ------------------------------------------------------------------ API
    def fit(self, X, y, feature_names=None):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.feature_names_ = list(feature_names) if feature_names is not None else [f"x{j}" for j in range(X.shape[1])]
        basis, B = self._forward(X, y)
        keep = self._backward(basis, B, y)
        self.basis_ = [basis[k] for k in keep]
        Z = self._design(X)
        self.logit_ = LogisticRegression(C=self.logit_C, max_iter=5000)
        self.logit_.fit(Z, y.astype(int))
        self.importance_ = self._importance(X, y)
        return self

    def _design(self, X):
        cols = [h.eval(X) for h in self.basis_ if h.terms]
        Z = np.column_stack(cols) if cols else np.zeros((X.shape[0], 1))
        if not hasattr(self, "z_scale_"):
            self.z_scale_ = Z.std(0) + 1e-9
        return Z / self.z_scale_

    def predict_proba(self, X):
        Z = self._design(np.asarray(X, float))
        p = self.logit_.predict_proba(Z)[:, 1]
        return np.column_stack([1 - p, p])

    def _importance(self, X, y):
        """Важность признака: рост GCV при удалении всех базисных функций с этим признаком."""
        Bfull = np.column_stack([h.eval(X) for h in self.basis_])
        base = self._gcv(Bfull, y)
        imp = {}
        for j, name in enumerate(self.feature_names_):
            keep = [i for i, h in enumerate(self.basis_) if j not in h.vars()]
            if len(keep) == len(self.basis_):
                continue
            imp[name] = max(self._gcv(Bfull[:, keep], y) - base, 0.0)
        tot = sum(imp.values()) or 1.0
        return dict(sorted(((k, v / tot) for k, v in imp.items()), key=lambda kv: -kv[1]))

    def contributions(self, x):
        """Вклад каждой базисной функции в логит для одного наблюдения (для интерфейса)."""
        x = np.asarray(x, float).reshape(1, -1)
        hs = [h for h in self.basis_ if h.terms]
        coefs = self.logit_.coef_[0]
        out = []
        for h, c, s in zip(hs, coefs, self.z_scale_):
            out.append((h.describe(self.feature_names_), float(c * h.eval(x)[0] / s)))
        return float(self.logit_.intercept_[0]), out

    def summary(self):
        hs = [h for h in self.basis_ if h.terms]
        lines = [f"intercept: {self.logit_.intercept_[0]:+.3f}"]
        for h, c, s in zip(hs, self.logit_.coef_[0], self.z_scale_):
            lines.append(f"{c / s:+.4f} · {h.describe(self.feature_names_)}")
        return "\n".join(lines)
