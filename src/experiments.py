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


def run_single_experiment(d, k, t, g, l, state_factory, sd=1.0):
    param = state_factory().randn(d)
    ctx_gen = CtxGenerator(k, d, grouped=g, state=state_factory())
    noise_gen = NoiseGenerator.gaussian_noise(sd, state=state_factory())

    env = Environment(param, ctx_gen, noise_gen)
    alpha = 1.0

    def radius():
        return (d * np.log(1 + (l * env.t) ** 2)) ** 0.5 + (alpha * d) ** 0.5

    algs = {
        'greedy': Roful.greedy(d, alpha=alpha),
        'oful': Roful.oful(d, alpha=alpha, radius=radius),
        'ts': Roful.ts(d, alpha=alpha, state=state_factory()),
        'freq-ts': Roful.ts(d, alpha=alpha, inflation=radius, state=state_factory()),
        'sg(.5)': Roful.sieved_greedy(d, alpha=alpha, radius=radius, tolerance=0.5),
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


def run_experiments(n, d, k, t, g, l, s):
    state_factory = StateFactory(s)

    aggregates = defaultdict(MetricAggregator)
    for i in range(n):
        print(f'Running experiment [{i}]...')
        metrics = run_single_experiment(d, k, t, g, l, state_factory)
        for name, metric in metrics.items():
            aggregates[name].aggregate(np.cumsum(metric.regrets))

    for name, aggregate in aggregates.items():
        mean, sd, se = aggregate.confidence_band()

        lower = mean - 2 * se
        upper = mean + 2 * se

        plt.fill_between(range(t), lower, upper, alpha=0.2)
        plt.plot(range(t), mean, label=name)

    plt.legend()
    plt.savefig(f'plots/regret-{n}-{d}-{k}-{t}-{s}-{"grouped" if g else "ungrouped"}.pdf')
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
    parser = argparse.ArgumentParser(description='Run simulations for various ROFUL algorithms.')

    parser.add_argument('-n', type=int, help='number of iterations', default=10)
    parser.add_argument('-k', type=int, help='number of actions', default=10)
    parser.add_argument('-d', type=int, help='dimension', default=120)
    parser.add_argument('-t', type=int, help='time horizon', default=2000)
    parser.add_argument('-s', type=int, help='random seed', default=1)
    parser.add_argument('-g', type=int, help='if set, action will be grouped', default=0)
    parser.add_argument('-l', type=float, help='length of the actions', default=1.0)

    args = parser.parse_args()

    run_experiments(n=args.n, d=args.d, k=args.k, t=args.t, g=args.g, l=args.l, s=args.s)


if __name__ == '__main__':
    __main__()
