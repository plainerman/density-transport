import random
import observations.dimensions.one.sampling as oneDimensionalObserve

samplesizex = 200
samplesizey = 200

def cusp(xs):
    return xs[0]**3 - xs[0] * xs[1]


def sample_beta_dir(delta=0.05):
    random.seed(0)

    ys = []

    for i in range(samplesizex):
        for j in range(samplesizey):
            x = oneDimensionalObserve.step_sample(i, -1, 1, samplesizex)
            beta1 = oneDimensionalObserve.step_sample(j, -1, 1, samplesizey)
            ys.append([beta1 + delta, cusp([x, beta1]), cusp([x, beta1 + delta])])

    return ys




