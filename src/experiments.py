from collections import defaultdict

import numpy as np
import numpy.random as npr

import matplotlib.pyplot as plt

from envs import Environment
from envs import NoiseGenerator
from envs import StochasticContextGenerator as CtxGenerator

from policies import Roful

from utils import MetricAggregator


def run_single_experiment(d, k, t, sd=1.0):
    param = npr.randn(d)
    ctx_gen = CtxGenerator(k, d)
    noise_gen = NoiseGenerator.gaussian_noise(sd)

    env = Environment(param, ctx_gen, noise_gen)

    algs = {
        'greedy': Roful.greedy(d, 1.0),
        'ts': Roful.thompson_sampling(d, 1.0),
        'oful': Roful.oful(d, 1.0, radius=d ** 0.5),
        'sg(.2)': Roful.sieved_greedy(d, 1.0, radius=d ** 0.5, tolerance=0.2),
        'sg(.5)': Roful.sieved_greedy(d, 1.0, radius=d ** 0.5, tolerance=0.5),
        'sg(1)': Roful.sieved_greedy(d, 1.0, radius=d ** 0.5, tolerance=1.0),
    }

    for i in range(t):
        ctx = env.next()

        for alg in algs.values():
            idx = alg.choose_arm(ctx)
            fb = env.get_feedback(idx)
            alg.update(fb)

        if i % 100 == 0:
            print(i)

    return {
        name: alg.metrics for name, alg in algs.items()
    }


def run_experiments(n, d, k, t):
    aggregates = defaultdict(MetricAggregator)
    for i in range(n):
        print(f'Running experiment [{i}]...')
        metrics = run_single_experiment(d, k, t)
        for name, metric in metrics.items():
            aggregates[name].aggregate(np.cumsum(metric.regrets))

    for name, aggregate in aggregates.items():
        mean, sd, se = aggregate.confidence_band()

        plt.fill_between(range(t), mean - se, mean + se, alpha=0.2)
        plt.plot(range(t), mean, label=name)

    plt.legend()
    plt.savefig(f'plots/regret-{d}-{k}.pdf')
    plt.show()

    print(f'All the experiments finished successfully.')


if __name__ == '__main__':
    run_experiments(50, d=25, k=100, t=1000)
