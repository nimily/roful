import numpy as np
import numpy.linalg as npl
import numpy.random as npr


def sqrt_sym(x):
    u, d, v = npl.svd(x)

    return u @ np.diag(d ** 0.5) @ v


class DataSummary:
    xy: np.ndarray
    xx: np.ndarray

    _mean: np.ndarray
    _basis: np.ndarray
    _scale: np.ndarray
    _dirty: bool

    def __init__(self, dim, alpha):
        self.xy = np.zeros(dim, dtype=np.float)
        self.xx = np.eye(dim, dtype=np.float) * alpha

        self._mean = np.zeros(dim, dtype=np.float)
        self._basis = np.eye(dim, dtype=np.float)
        self._scale = np.ones(dim, dtype=np.float) * alpha
        self._dirty = False

    def _update_caches(self):
        svd = npl.svd(self.xx, hermitian=True)

        self._mean = npl.solve(self.xx, self.xy)
        self._basis = svd[0].T
        self._scale = svd[1]
        self._dirty = False

    def add_obs(self, x, y):
        self.xy += x * y
        self.xx += np.outer(x, x)

        self._dirty = True

    @property
    def d(self):
        return self.xy.shape[0]

    @property
    def mean(self) -> np.ndarray:
        if self._dirty:
            self._update_caches()

        return self._mean

    @property
    def basis(self) -> np.ndarray:
        if self._dirty:
            self._update_caches()

        return self._basis

    @property
    def scale(self) -> np.ndarray:
        if self._dirty:
            self._update_caches()

        return self._scale


class MetricAggregator:

    def __init__(self):
        self.m0 = []
        self.m1 = []
        self.m2 = []

    def confidence_band(self):
        m0 = np.array(self.m0)
        m1 = np.array(self.m1)
        m2 = np.array(self.m2)

        m0 = np.maximum(m0, 1)

        mean = m1 / m0
        var = (m2 - m1 ** 2 / m0) / (m0 - 1)
        sd = var ** 0.5
        se = (var / m0) ** 0.5

        return mean, sd, se

    def aggregate(self, xs, filter=lambda _: True):
        self._ensure_len(len(xs))

        for i, x in enumerate(xs):
            if filter(i):
                self.m0[i] += 1
                self.m1[i] += x
                self.m2[i] += x ** 2

    def _ensure_len(self, n):
        dn = n - len(self.m0)

        if dn > 0:
            self.m0 += [0] * dn
            self.m1 += [0] * dn
            self.m2 += [0] * dn


class StateFactory:

    def __init__(self, seed):
        self.seed = seed

    def __call__(self):
        state = npr.seed(self.seed)

        self.seed += 1

        return state
