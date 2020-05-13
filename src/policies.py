import abc

from typing import List, Callable

import numpy as np
import numpy.linalg as npl
import numpy.random as npr

from envs import Context
from envs import Feedback

from utils import DataSummary


class Metric:
    regrets: List[float]

    def __init__(self):
        self.regrets = []

    def add_obs(self, feedback: Feedback):
        self.regrets.append(feedback.regret)


class Policy:
    metrics: Metric

    def __init__(self):
        self.metrics = Metric()

    @abc.abstractmethod
    def choose_arm(self, ctx: Context) -> int:
        pass

    def update(self, feedback: Feedback):
        self.update_model(feedback)
        self.update_metrics(feedback)

    @abc.abstractmethod
    def update_model(self, feedback: Feedback):
        pass

    def update_metrics(self, feedback: Feedback):
        self.metrics.add_obs(feedback)


class WorthFunction:
    summary: DataSummary

    def bind(self, summary: DataSummary):
        self.summary = summary

        self.update()

    def update(self):
        pass

    @abc.abstractmethod
    def compute(self, ctx: Context) -> np.ndarray:
        pass


class ProductWorthFunction(WorthFunction):

    def compute(self, ctx: Context) -> np.ndarray:
        values = ctx.arms @ self.candidates()

        if values.ndim == 2:
            values = np.max(values, axis=1)

        return values

    @abc.abstractmethod
    def candidates(self):
        pass


class Roful(Policy):
    summary: DataSummary
    worth_func: WorthFunction

    def __init__(self, d, alpha, worth_func: WorthFunction):
        super().__init__()

        self.summary = DataSummary(d, alpha)
        self.worth_func = worth_func

    @staticmethod
    def ts(d, alpha, inflation=1.0, state=npr):
        if isinstance(inflation, float):
            inflation = Roful._const_inflation(inflation)

        return Roful(d, alpha, ThompsonWorthFunction(inflation, state=state))

    @staticmethod
    def dts(d, alpha, inflation=1.0, state=npr):
        if isinstance(inflation, float):
            inflation = Roful._const_inflation(inflation)

        return Roful(d, alpha, DirectionalThompsonWorthFunction(inflation, state=state))

    @staticmethod
    def _const_inflation(value):
        return lambda: value

    @staticmethod
    def oful(d, alpha, radius):
        return Roful(d, alpha, SievedGreedyWorthFunction(radius, 1.0))

    @staticmethod
    def greedy(d, alpha):
        return Roful(d, alpha, GreedyWorthFunction())

    @staticmethod
    def sieved_greedy(d, alpha, radius, tolerance=1.0):
        return Roful(d, alpha, SievedGreedyWorthFunction(radius, tolerance))

    @property
    def d(self):
        return self.summary.d

    def choose_arm(self, ctx: Context) -> int:
        self.worth_func.bind(self.summary)
        values = self.worth_func.compute(ctx)
        return np.argmax(values).item()

    def update_model(self, feedback: Feedback):
        x = feedback.chosen_arm
        y = feedback.rew

        self.summary.add_obs(x, y)


class ThompsonWorthFunction(ProductWorthFunction):
    compensator: np.ndarray

    def __init__(self, inflation, state=npr):
        self.inflation = inflation
        self.state = state

    def update(self):
        rand = self.state.randn(self.summary.d)

        basis = self.summary.basis
        scale = self.summary.scale

        self.compensator = self.inflation() * basis.T @ (rand / scale ** 0.5)

    def candidates(self):
        return self.summary.mean + self.compensator


class DirectionalThompsonWorthFunction(ProductWorthFunction):
    compensator: np.ndarray

    def __init__(self, inflation, state=npr):
        self.inflation = inflation
        self.state = state

    def update(self):
        rand = self.state.randn(self.summary.d)

        basis = self.summary.basis
        scale = self.summary.scale

        self.compensator = self.inflation() * basis.T @ np.diag(rand * (self.summary.d / scale) ** 0.5)

    def candidates(self):
        return self.summary.mean + self.compensator


class GreedyWorthFunction(ProductWorthFunction):

    def __init__(self, inflation=1.0):
        self.inflation = inflation

    def candidates(self):
        return self.summary.mean


class SievedGreedyWorthFunction(WorthFunction):
    radius: Callable
    tolerance: float

    def __init__(self, radius, tolerance):
        self.radius = radius
        self.tolerance = tolerance

    def compute(self, ctx: Context) -> np.ndarray:
        lowers, centers, uppers = self.confidence_bounds(ctx.arms)

        # sieving arms
        baseline = lowers.max()
        optimal = uppers.max()

        threshold = self.tolerance * optimal + (1.0 - self.tolerance) * baseline
        survivors = (uppers >= threshold)

        # computing the values
        return np.where(survivors, centers, lowers)

    def confidence_center(self, arms):
        return arms @ self.summary.mean

    def confidence_width(self, arms):
        scale = arms @ npl.solve(self.summary.xx, arms.T)

        if len(scale.shape) == 2:
            scale = np.diag(scale)

        return self.radius() * scale ** 0.5

    def confidence_bounds(self, arms):
        centers = self.confidence_center(arms)
        widths = self.confidence_width(arms)

        return centers - widths, centers, centers + widths
