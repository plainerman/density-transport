import random
import observations.dimensions.one.sampling as oneDimensionalObserve
import numpy

samplesizex = 100
samplesizey = 100


def cusp(xs):
    return xs[0]**3 - xs[0] * xs[1]


def cusp_vectorized(X, Y):
    return X**3 - X * Y


def sample_beta_dir(delta=0.05):
    random.seed(0)

    ys = []

    for i in range(samplesizey):
        for j in range(samplesizex):
            x = oneDimensionalObserve.step_sample(i, -1, 1, samplesizex)
            beta1 = oneDimensionalObserve.step_sample(j, -1, 1, samplesizey)
            ys.append([beta1 + delta, cusp([x, beta1]), cusp([x, beta1 + delta])])

    return ys


def sample_beta_2(samplesizex=samplesizex, samplesizey=samplesizey, delta=0.3):
    ys = []

    xs = numpy.linspace(-1, 1, samplesizex)
    beta1s = numpy.linspace(-1, 1, samplesizex)
    for x in xs:
        for beta1 in beta1s:
            # x = step_sample(i, -1, 1, samplesizex)
            # beta1 = step_sample(j, -1, 1, samplesizey)
            ys.append([cusp([x, beta1]), cusp([x - delta, beta1]), cusp([x - 2 * delta, beta1])])

    return ys
