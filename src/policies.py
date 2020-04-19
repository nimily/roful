import abc

from typing import List

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
        perceived = list(map(self.max_perceived_reward, self.context.arms))
        return np.argmax(perceived).item()

    @abc.abstractmethod
    def max_perceived_reward(self, arm):
        pass


class Roful(Policy):
    summary: DataSummary
    search_set: SearchSet

    def __init__(self, d, alpha, search_set: SearchSet):
        super().__init__()

        self.summary = DataSummary(d, alpha)
        self.search_set = search_set

    @staticmethod
    def thompson_sampling(d, alpha, inflation=1.0):
        return Roful(d, alpha, ThompsonSearchSet(inflation))

    @staticmethod
    def oful(d, alpha, radius=1.0):
        return Roful(d, alpha, OfulSearchSet(radius))

    @staticmethod
    def greedy(d, alpha):
        return Roful(d, alpha, GreedySearchSet())

    @staticmethod
    def sieved_greedy(d, alpha, radius=1.0, tolerance=1.0):
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
    sample: np.ndarray

    def __init__(self, inflation=1.0):
        self.inflation = inflation

    def update(self):
        mean = self.summary.mean
        rand = npr.randn(self.summary.d)

        basis = self.summary.basis
        scale = self.summary.scale

        self.sample = mean + basis @ ((basis.T @ rand) / (scale ** 0.5))

    def max_perceived_reward(self, arm):
        return self.sample @ arm


class GreedySearchSet(ProductSearchSet):
    def __init__(self, inflation=1.0):
        self.inflation = inflation

    def max_perceived_reward(self, arm):
        return self.summary.mean @ arm


class SievedGreedySearchSet(SearchSet):
    radius: float
    tolerance: float

    def __init__(self, radius, tolerance):
        self.radius = radius
        self.tolerance = tolerance

    def find_optimal_arm(self) -> int:
        subset = self.sieve_arms()

        values = [self.confidence_center(self.context.arms[i]) for i in subset]
        index = np.argmax(values).item()

        return subset[index]

    def sieve_arms(self):
        lowers, uppers = self.confidence_bounds(self.context.arms)

        accept = lowers.max()
        optimal = uppers.max()

        threshold = self.tolerance * (optimal - accept) * 0.99999 + accept

        return np.argwhere(uppers > threshold).flatten()

    def confidence_center(self, arms):
        return arms @ self.summary.mean

    def confidence_width(self, arms):
        scale = arms @ npl.solve(self.summary.xx, arms.T)

        if len(scale.shape) == 2:
            scale = np.diag(scale)

        return self.radius * scale ** 0.5

    def confidence_bounds(self, arm):
        center = self.confidence_center(arm)
        width = self.confidence_width(arm)

        return center - width, center + width


class OfulSearchSet(SievedGreedySearchSet):
    radius: float

    def __init__(self, radius=1.0):
        super().__init__(radius, tolerance=1.0)

    def max_perceived_reward(self, arm):
        mean = npl.solve(self.summary.xx, self.summary.xy)

        return mean @ arm + self.radius * (arm @ npl.solve(self.summary.xx, arm)) ** 0.5
