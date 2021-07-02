import random


samplesize = 50000


def step_sample(i, a, b, samplesize):
    return (i / samplesize) * (b - a) + a


def uniform_sample(i, a, b, samplesize):
    return random.uniform(a, b)


def T(x):
    # return -4*x*x + 4*x a non-injective transport map that produces no jump discontinuity
    return -2 * pow((1 - x), 3) + 1.5 * (1 - x) + 0.5


def sample(observations, y_x_fun=T, sample_fun=step_sample, delta=0.1):
    random.seed(0)
    ys = [[]] * samplesize

    rands = [sample_fun(i, 0, 1, samplesize) for i in range(samplesize)]
    rands.sort() # This step allows us to easily calculate the arclength. It is only required for non-monotonic sample functions
    for i in range(samplesize):
        ys[i] = [y_x_fun(rands[i] - x * delta) for x in range(observations + 1)]
    return ys
