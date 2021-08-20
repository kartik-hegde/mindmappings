"""
    All the useful functions.
"""
import math
from functools import reduce

def factors(n):
    """
        Returns factors of n.
    """
    return set(reduce(list.__add__,
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

def replicate(l):
    """
        Replicates if there is only one element.
    """
    if(len(l) == 1):
        l.append(l[0])
    return l

def non_increasing(arr):
    """
        Returns the monotonically non-increasing array
    """
    best_min = arr[0]
    return_arr = []
    for val in arr:
        return_arr.append(min(best_min, val))
        if(val<best_min):
            best_min = val
    return return_arr

def inclusive_range(start, end, stride=1):
    """ Returns a range that includes the start and end. """
    temp_range = list(range(start, end, stride))
    if(end not in temp_range):
        temp_range.append(end)
    return temp_range

def getTotalIterations(tile_size, shape):
    """
        Returns the number of tile iterations.
    """

    num_dims = len(shape)
    iters = 1
    for i in range(num_dims):
        iters *= math.ceil(float(tile_size[i])/shape[i])

    return iters

def getIters(tile_size, dim):

    return math.ceil(float(dim/tile_size))

def getReuseVector(loopOrder, iterVector):
    """
        The reuse embedded in each loop dim.
    """
    reuseVector = [0]*len(iterVector)
    for idx,lord in enumerate(loopOrder):
        reuseVector[idx] = iterVector[lord]

    return reuseVector
