from collections import defaultdict

import argparse

import numpy as np

import matplotlib.pyplot as plt

from envs import Environment
from envs import NoiseGenerator
from envs import StochasticContextGenerator as CtxGenerator

from policies import Roful

from utils import MetricAggregator
from utils import StateFactory


def run_single_experiment(d, k, t, state_factory, sd=1.0):
    param = state_factory().randn(d)
    ctx_gen = CtxGenerator(k, d, state=state_factory())
    noise_gen = NoiseGenerator.gaussian_noise(sd, state=state_factory())

    env = Environment(param, ctx_gen, noise_gen)

    algs = {
        'greedy': Roful.greedy(d, 1.0),
        'oful': Roful.oful(d, 1.0, radius=d ** 0.5),
        'ts': Roful.ts(d, 1.0, state=state_factory()),
        'dts': Roful.dts(d, 1.0, state=state_factory()),
        'sg(.5)': Roful.sieved_greedy(d, 1.0, radius=d ** 0.5, tolerance=0.5),
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


def run_experiments(n, d, k, t, s):
    state_factory = StateFactory(s)

    aggregates = defaultdict(MetricAggregator)
    for i in range(n):
        print(f'Running experiment [{i}]...')
        metrics = run_single_experiment(d, k, t, state_factory)
        for name, metric in metrics.items():
            aggregates[name].aggregate(np.cumsum(metric.regrets))

    for name, aggregate in aggregates.items():
        mean, sd, se = aggregate.confidence_band()

        plt.fill_between(range(t), mean - se, mean + se, alpha=0.2)
        plt.plot(range(t), mean, label=name)

    plt.legend()
    plt.savefig(f'plots/regret-{n}-{d}-{k}-{t}-{s}.pdf')
    plt.show()

    print()
    print('  policy   |   regret')
    print('-' * 25)
    for name, aggregate in aggregates.items():
        mean = aggregate.confidence_band()[0][-1]
        print(f'{name:10} | {mean:.2f}')
    print()

    print(f'All the experiments finished successfully.')


def __main__():
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('-n', type=int, help='number of iterations', default=25)
    parser.add_argument('-k', type=int, help='number of actions', default=100)
    parser.add_argument('-d', type=int, help='dimension', default=125)
    parser.add_argument('-t', type=int, help='time horizon', default=500)
    parser.add_argument('-s', type=int, help='random seed', default=1)

    args = parser.parse_args()

    run_experiments(n=args.n, d=args.d, k=args.k, t=args.t, s=args.s)


if __name__ == '__main__':
    __main__()
