import numpy as np
from mindmappings.utils.parallelProcess import parallelProcess
from copy import deepcopy

class GradientDescent:
    """
        Basic Gradient Descent Algorithm.
    """

    def __init__(self, f, df, constraints, learning_factor=0.5, decay_factor=0.5, error=1e-10):
        """
            Initialize.
        """
        self.alpha = learning_factor
        self.decay_factor = decay_factor
        self.epsilon = error
        self.f = f
        self.df = df
        self.constraints = constraints

    def gradient_descent(self, x0, steps=100):
        """
            Simple Gradient Descent Algorithm.
        """
        alpha = self.alpha
        iters = 0
        x = x0
        best_x = x0
        cost = self.f(x0)
        result = [cost,]
        while iters < steps:
            x_next = x - (alpha * self.df(x))
            next_cost = self.f(x_next)
            # print("x: {0}, x_next: {1}, cost: {2}, next_cost: {3}".format(x, x_next, min(result), next_cost))
            if((cost < next_cost) or (not self.constraints(x_next))):
                x_next = x
                alpha = alpha * self.decay_factor
                # print("Updated Alpha", alpha)
                result.append(min(result))
            else:
                result.append(next_cost)
                best_x = x
            # input()
            x = x_next
            iters += 1

        return result, best_x

    def gradDescent_unpack(self, args):
        return self.gradient_descent(*args)[0]

    def runGradDesc(self, x0=None, steps=100, average=100):

        # Avg iters
        n=average
        # Random init
        # Fixed init point
        x0 = np.array([10.0, 10.0, 10.0, 10.0, 10.0]) if(x0==None) else x0 #Fixed Init
        # x0 = np.array([.25, .25, .25, 0.25, 0.25])  #parse_initial_guess(args)
        init_points = [x0 for _ in range(n)]

        # Launch work
        work = [deepcopy((init_points[i], steps)) for i in range(n)]
        costArr = parallelProcess(self.gradDescent_unpack, work, num_cores=None)

        # Slice the array
        allMins = [np.mean(costArr[i:i+n], axis=0) for i in range(0, len(costArr), n)]
        stdDev = [np.std(costArr[i:i+n], axis=0) for i in range(0, len(costArr), n)]
        # print("Mins", allMins)

        print("Done!")

        return allMins

if __name__ == '__main__':
    def f(x):
        return x*x+x
    def df(x):
        return 2*x+1
    def constraints(x):
        return True
    grad_descent = GradientDescent(f,df,constraints, learning_factor=0.1, decay_factor=0.5)
    print(grad_descent.runGradDesc(10.0))

