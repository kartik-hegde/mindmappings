import multiprocessing
from itertools import product
from contextlib import contextmanager

def merge_names(a, b):
    """
        Dummy function.
    """
    return a+b

def merge_names_unpack(args):
    """
        Returns unpacked arguments
    """
    return merge_names(*args)

# Helper function to create pools
@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

def parallelProcess(func, args, num_cores=None):
    # Use the maximum number of cores
    if(num_cores == None):
        num_cores = multiprocessing.cpu_count()

    #Create the pool
    with poolcontext(num_cores) as pool:
        results = pool.map(func, args)
    return results

if __name__ == '__main__':
    results = parallelProcess(merge_names_unpack, [(1,2),(2,3)])
    print(results)
