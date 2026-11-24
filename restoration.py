"""LP reference and primal-dual certificate bank for restoration queries."""

import os

# Single-threaded reference and benchmark runs use the same numerical settings.
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

import json
from pathlib import Path

import highspy
import numpy as np
from scipy.sparse import csr_matrix


PROJECT_ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = PROJECT_ROOT / "protocol.json"
P = json.loads(PROTOCOL_PATH.read_text())


def profile(rng, i):
    """Generate a synthetic recovery profile with three or nine trajectories."""
    m = 1 if i % 2 == 0 else 3
    gs = np.sort(rng.choice(np.arange(3, 17), 3, False))
    cs = np.sort(rng.choice(np.arange(5, 22), m, False))
    g = np.repeat(gs, m)
    c = np.tile(cs, 3)
    s = len(g)
    return dict(id=i, grid=g.tolist(), cool=c.tolist(), path=rng.uniform(0.75, 1.05, s).tolist(), ups=float(rng.uniform(0.95, 1.2)), ride=int(rng.integers(6, 11)), remote=float(rng.uniform(0.5, 0.9)), rate=float(rng.uniform(0.12, 0.25)), message_delay=int(rng.choice([1, 2, 3])))

def model(c, w, lag):
    """Construct the finite-support LP with shared observation histories."""
    S = len(c['grid'])
    N = 20
    idx = np.empty((S, N), int)
    G = 0
    for k in range(N):
        groups = {}
        for s in range(S):
            key = tuple(((name, int(c[name][s])) for name in ['grid', 'cool'] if c[name][s] + c['message_delay'] <= k))
            groups.setdefault(key, []).append(s)
        for ids in groups.values():
            idx[ids, k] = G
            G += 1
    grid = np.arange(N)[None, :] >= np.array(c['grid'])[:, None]
    cool = (np.arange(N)[None, :] < c['ride']) | (np.arange(N)[None, :] >= np.array(c['cool'])[:, None])
    cap = np.where(grid, np.array(c['path'])[:, None], c['ups']) * cool
    n = 2 * G + 1
    A0 = []
    A1 = []
    b = []
    energy = []

    def row(items, rhs, mi=[]):
        a = np.zeros(n)
        d = np.zeros(n)
        for j, v in items:
            a[j] += v
        for j, v in mi:
            d[j] += v
        A0.append(a)
        A1.append(d)
        b.append(rhs)
    for s in range(S):
        for k in range(N):
            row([(idx[s, k], 1)] + [(G + idx[s, j], 1) for j in range(k)], 1.0)
            row([(idx[s, k], 1)], float(cap[s, k]), [(G + idx[s, k], 1)])
        energy.append(len(b))
        row([(idx[s, k], 1) for k in range(N) if not grid[s, k]], 0.0, [(G + idx[s, k], 1) for k in range(N) if not grid[s, k]])
        row([(idx[s, k], -1) for k in range(N)] + [(G + idx[s, j], -max(0, N - j - lag - 1)) for j in range(N)] + [(n - 1, -1)], -N)
        row([(G + idx[s, k], 1) for k in range(N)], c['remote'])
    gt = np.zeros(G, int)
    for s in range(S):
        gt[idx[s]] = np.arange(N)
    ub = np.r_[np.ones(G), [c['rate'] if k >= w else 0 for k in gt], N]
    obj = np.zeros(n)
    obj[-1] = 1
    return (np.array(A0), np.array(A1), np.array(b), ub, obj, np.array(energy, dtype=np.int32))

class Oracle:

    def __init__(self, m):
        self.A0, self.A1, self.b, self.ub, self.c, self.e = m
        self.k = None
        a = csr_matrix(self.A0)
        self.h = highspy.Highs()
        for k, v in [('output_flag', False), ('threads', 1), ('solver', 'simplex'), ('presolve', 'off'), ('primal_feasibility_tolerance', 1e-09), ('dual_feasibility_tolerance', 1e-09)]:
            self.h.setOptionValue(k, v)
        self.h.addVars(len(self.c), np.zeros(len(self.c)), self.ub)
        self.h.changeColsCost(len(self.c), np.arange(len(self.c), dtype=np.int32), self.c)
        self.h.addRows(len(self.b), np.full(len(self.b), -highspy.kHighsInf), self.b, a.nnz, a.indptr.astype(np.int32), a.indices.astype(np.int32), a.data)
        self.rr, self.cc = np.nonzero(self.A1)

    def solve(self, E, k, witness=False):
        if not np.isfinite(E + k) or E < 0 or k < 0:
            raise ValueError('Energy and migration cost must be finite and nonnegative')
        if self.k != k:
            for r, c in zip(self.rr, self.cc):
                self.h.changeCoeff(int(r), int(c), float(self.A0[r, c] + k * self.A1[r, c]))
            self.k = k
        self.h.changeRowsBounds(len(self.e), self.e, np.full(len(self.e), -highspy.kHighsInf), np.full(len(self.e), E))
        self.h.run()
        assert self.h.getModelStatus() == highspy.HighsModelStatus.kOptimal
        if not witness:
            return float(self.h.getObjectiveValue())
        s = self.h.getSolution()
        return (np.maximum(0, np.minimum(self.ub, np.array(s.col_value))), np.minimum(0, np.array(s.row_dual)))

class Bank:

    def __init__(self, m, o):
        self.m = m
        A0, A1, b, ub, c, e = m
        self.resource = b >= 0
        self.service = b < 0
        self.e = e
        self.ub = ub
        self.b = b
        self.c = c
        xx = []
        yy = []
        coords = []
        for k in P['bank_kappa']:
            for E in P['bank_energy']:
                x, y = o.solve(E, k, True)
                xx.append(x)
                yy.append(y)
                coords.append([E, k])
        self.coords = np.array(coords)
        X = np.array(xx)
        Y = np.array(yy)
        # Remove action coordinates blocked by zero allowances before scaling.
        mask = b == 0
        mask[e] = False
        X[:, np.any(A0[mask] + 4 * A1[mask] > 0, axis=0)] = 0
        self.X = X
        self.Y = Y
        self.p0 = X @ A0.T
        self.p1 = X @ A1.T
        self.d0 = c[None, :] - Y @ A0
        self.d1 = -Y @ A1
        self.yb = Y @ b
        self.ye = Y[:, e].sum(1)

    def bounds(self, E, k, method):
        if not np.isfinite(E + k) or not 2 <= E <= 10 or (not 0 <= k <= 4):
            return (-np.inf, np.inf)
        if method == 'nearest_certificate':
            ids = np.array([np.argmin(((self.coords - [E, k]) / [8, 4]) ** 2 @ np.ones(2))])
        else:
            ids = np.arange(len(self.coords))
        # Re-evaluate each stored dual witness against the queried matrix.
        lo = self.yb[ids] + self.ye[ids] * E + np.minimum(self.d0[ids] + k * self.d1[ids], 0) @ self.ub
        use = (self.p0[ids] + k * self.p1[ids])[:, self.resource]
        rhs = self.b.copy()
        rhs[self.e] = E
        rhs = rhs[self.resource]
        ratio = np.divide(rhs[None, :], use, out=np.full_like(use, np.inf), where=use > 0)
        scale = np.minimum(1, np.min(ratio, axis=1)) * (1 - 1e-10)
        # Remove the old epigraph and compute the loss of the repaired policy.
        servicework = self.p0[ids][:, self.service] + self.X[ids, -1, None]
        up = np.maximum(0, np.max(20 + scale[:, None] * servicework, axis=1)) + 1e-08
        return (float(np.max(lo)) - 1e-07, float(np.min(up)) + 1e-07)

    def monotone(self, E, k):
        low = max((v for v in P['bank_kappa'] if v <= k))
        high = min((v for v in P['bank_kappa'] if v >= k))
        ids = np.where(self.coords[:, 1] == low)[0]
        # Re-evaluate each stored dual witness against the queried matrix.
        lo = self.yb[ids] + self.ye[ids] * E + np.minimum(self.d0[ids] + low * self.d1[ids], 0) @ self.ub
        vals = []
        for ee in P['bank_energy']:
            j = np.where((self.coords[:, 0] == ee) & (self.coords[:, 1] == high))[0][0]
            use = self.p0[j] + high * self.p1[j]
            rhs = self.b.copy()
            rhs[self.e] = ee
            pos = self.resource & (use > 0)
            scale = min(1, float(np.min(rhs[pos] / use[pos]))) * (1 - 1e-10)
            vals.append(max(0, float(np.max(20 + scale * (self.p0[j, self.service] + self.X[j, -1])))) + 1e-08)
        return (float(np.max(lo)) - 1e-07, float(np.interp(E, P['bank_energy'], vals)) + 1e-07)
