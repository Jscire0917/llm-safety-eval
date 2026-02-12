import random
## This module provides statistical functions for evaluating AI performance.
def bootstrap_ci(values, n=1000):
    if not values:
        return (0.0, 0.0)
    means = []
    for _ in range(n):
        sample = [random.choice(values) for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return (means[int(0.025*n)], means[int(0.975*n)])
