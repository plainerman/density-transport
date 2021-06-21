import numpy

def arclen(points):
    """Takes points is a list with vectors of the points in every dimension e.g. [xs, ys, zs]"""
    p1 = numpy.array([x[:-1] for x in points])
    p2 = numpy.array([x[1:] for x in points])

    dist = numpy.sqrt(numpy.sum(numpy.power((p2 - p1), 2), axis=0))
    arclen = [0] * len(dist)
    for i in range(1, len(dist)):
        arclen[i] = arclen[i - 1] + dist[i - 1]

    return numpy.array(arclen)