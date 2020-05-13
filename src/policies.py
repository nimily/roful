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


class SearchSet:
    summary: DataSummary
    context: Context

    def bind(self,
             summary: DataSummary,
             context: Context):
        self.summary = summary
        self.context = context

        self.update()

    def update(self):
        pass

    @abc.abstractmethod
    def find_optimal_arm(self) -> int:
        pass


class ProductSearchSet(SearchSet):

    def find_optimal_arm(self) -> int:
        perceived = self.max_perceived_reward(self.context.arms)
        return np.argmax(perceived).item()

    @abc.abstractmethod
    def max_perceived_reward(self, arms):
        pass


class Roful(Policy):
    summary: DataSummary
    search_set: SearchSet

    def __init__(self, d, alpha, search_set: SearchSet):
        super().__init__()

        self.summary = DataSummary(d, alpha)
        self.search_set = search_set

    @staticmethod
    def ts(d, alpha, inflation=1.0, state=npr):
        if isinstance(inflation, float):
            inflation = Roful._const_inflation(inflation)

        return Roful(d, alpha, ThompsonSearchSet(inflation, state=state))

    @staticmethod
    def dts(d, alpha, inflation=1.0, state=npr):
        if isinstance(inflation, float):
            inflation = Roful._const_inflation(inflation)

        return Roful(d, alpha, DirectionalThompsonSearchSet(inflation, state=state))

    @staticmethod
    def _const_inflation(value):
        return lambda: value

    @staticmethod
    def oful(d, alpha, radius):
        return Roful(d, alpha, SievedGreedySearchSet(radius, 1.0))

    @staticmethod
    def greedy(d, alpha):
        return Roful(d, alpha, GreedySearchSet())

    @staticmethod
    def sieved_greedy(d, alpha, radius, tolerance=1.0):
        return Roful(d, alpha, SievedGreedySearchSet(radius, tolerance))

    @property
    def d(self):
        return self.summary.d

    def choose_arm(self, ctx: Context) -> int:
        self.search_set.bind(self.summary, ctx)
        return self.search_set.find_optimal_arm()

    def update_model(self, feedback: Feedback):
        x = feedback.chosen_arm
        y = feedback.rew

        self.summary.add_obs(x, y)


class ThompsonSearchSet(ProductSearchSet):
    compensator: np.ndarray

    def __init__(self, inflation, state=npr):
        self.inflation = inflation
        self.state = state

    def update(self):
        rand = self.state.randn(self.summary.d)

        basis = self.summary.basis
        scale = self.summary.scale

        self.compensator = self.inflation() * basis.T @ (rand / scale ** 0.5)

    def max_perceived_reward(self, arms):
        return arms @ (self.summary.mean + self.compensator)


class DirectionalThompsonSearchSet(ProductSearchSet):
    compensator: np.ndarray

    def __init__(self, inflation, state=npr):
        self.inflation = inflation
        self.state = state

    def update(self):
        rand = self.state.randn(self.summary.d)

        basis = self.summary.basis
        scale = self.summary.scale

        self.compensator = self.inflation * basis.T @ np.diag(rand * (self.summary.d / scale) ** 0.5)

    def max_perceived_reward(self, arms):
        return arms @ self.summary.mean + np.max(arms @ self.compensator, axis=1)


class GreedySearchSet(ProductSearchSet):
    def __init__(self, inflation=1.0):
        self.inflation = inflation

    def max_perceived_reward(self, arms):
        return arms @ self.summary.mean


class SievedGreedySearchSet(SearchSet):
    radius: Callable
    tolerance: float

    def __init__(self, radius, tolerance):
        self.radius = radius
        self.tolerance = tolerance

    def find_optimal_arm(self) -> int:
        subset = self.sieve_arms()

        if len(subset) == 1:
            return subset[0]

        values = [self.confidence_center(self.context.arms[i]) for i in subset]
        index = np.argmax(values).item()

        return subset[index]

    def sieve_arms(self):
        lowers, uppers = self.confidence_bounds(self.context.arms)

        accept = lowers.max()
        optimal = uppers.max()

        threshold = self.tolerance * optimal + (1.0 - self.tolerance) * accept

        return np.argwhere(uppers >= threshold).flatten()

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

        return centers - widths, centers + widths
