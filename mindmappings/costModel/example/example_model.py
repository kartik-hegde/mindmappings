import random
import numpy as np
from mindmappings.costModel.model import Model
from mindmappings.costModel.example.grad_descent import GradientDescent

"""

This file should generate random valid mappings to the planned architecture.

This is a simple example to give the cost of a simple quadratic equation.

"""


class ExampleModel(Model):
    """
        An Example Cost Model
    """
    def __init__(self, **kwargs):
        self.dims = 5
        self.Q, self.b, self.c = self.gen_function()
        self.base_x, self.bound_x = (-25.0, 25.0)
        self.domain = self.getDomain()
        super().__init__(**kwargs)

    def costFn(self, mapping, **kwargs):
        """ 
            Return the cost of the mapping

            Input
            -------------
            mapping: The mapping for which the cost is to be returned

            Returns
            -------------
            cost: Returns the cost of the function. Additional argumetns can be used to
                    decide the metric of the cost.

        """
        x = np.array(mapping)
        # Evaluate the polynomial
        cost = x.T @ self.Q @ x + self.b.T @ x + self.c
        # Return a cost vector if needed 
        metric = kwargs.get('metric', None)
        if(metric == 'RAW'):
            return np.array([cost])
        else:
            return cost

    def getMapping(self, **kwargs):
        """ 
            Gives a random mapping.

            Input
            -------------
            None

            Returns
            -------------
            mapping: A mapping object, which needs to be understood by the costFn function.

        """
        return [random.uniform(self.domain[m][0], self.domain[m][1]) for m in range(self.dims)]

    def getMapCost(self, **kwargs):
        """ 
            Choose a random mapping and return its cost.

            Input
            -------------
            kwargs: Any argument that the subclass may want for cost.

            Returns
            -------------
            cost
            mapping

        """
        metric = kwargs.get('metric', None)
        mapping = self.getMapping()
        cost = self.costFn(mapping, metric=metric)
        return mapping, cost

    def getDomain(self, **kwargs):
        """ 
            Get the Mapping domain.

            Input
            -------------
            None

            Returns
            -------------
            domain: list of length of input parameters, where each entry is [base, bound]
        """
        return [[self.base_x,self.bound_x] for _ in range(self.dims)]

    def getOracleCost(self, **kwargs):
        """ 
            Get the theoretical lower bound cost. This is useful to normalize the dataset across
            different problems.

            Input
            -------------
            kwargs: Any argument that the subclass may want for cost.

            Returns
            -------------
            cost

        """
        # Return a cost vector if needed 
        metric = kwargs.get('metric', None)
        if(metric == 'RAW'):
            return np.array([1.0,])
        else:
            return 1.0

    def getInputVector(self, mapping):
        """
        
            Returns an input vector for training: <mapspace_id + mapping as a vector >. This can be fed to an MLP.
            
            Input
            -------------
            mapping  

            Returns
            -------------
            a list of [*mapspace_id, *flattened_mapping]

            """
        return mapping

    def grad(self, x):
        """ 
            Get gradient of f at x

            Input
            -------------
            x (input)

            Returns
            -------------
            gradient
        """
        x = np.array(x)
        return 2 * self.Q @ x + self.b

    def within_constraint(self,x):
        """ 
            Check whether the input is within the vounds or not

            Input
            -------------
            x (input)

            Returns
            -------------
            bool: 
        """
        return np.all(x > self.base_x) and np.all(x < self.bound_x)

    def random_search(self, steps=100, average=100):
        """ 
            Perform a random search to find the mins

            Input
            -------------
            steps: Number of steps
            average: Number of runs to average on

            Returns
            -------------
            best_costs: array of length steps, monotonically decreasing
            best_x: Optimum Choice
        """
        mapping = model.getMapping()
        cost = model.costFn(mapping)
        best_x = mapping
        best_costs = [cost,]
        for step in range(steps-1):
            mapping = model.getMapping()
            cost = model.costFn(mapping)
            if(cost < min(best_costs)):
                best_x = mapping
            best_costs.append(min(min(best_costs), cost))

        return best_costs, best_x

    def gradient_search(self, steps=100):
        """ 
            Perform a gradient search to find the mins.

            Input
            -------------
            steps: Number of steps
            average: Number of runs to average on

            Returns
            -------------
            best_costs: array of length steps, monotonically decreasing
            best_x: Optimum Choice
        """
        grad_descent = GradientDescent(self.costFn,self.grad,self.within_constraint, learning_factor=0.1, decay_factor=0.5)
        return grad_descent.gradient_descent(self.getMapping(), steps)

    def gen_function(self):
        Q = np.array([
            [1.596289104158055361e+00, 1.519878384043357400e+00, 1.401695563741551576e+00, 7.558383570476635560e-01, 1.694225410021412026e+00],
            [1.519878384043357400e+00, 2.026586923317563294e+00, 1.361733716144521544e+00, 1.022542464160217435e+00, 1.777975206631031035e+00],
            [1.401695563741551576e+00, 1.361733716144521544e+00, 1.654686339316024934e+00, 7.738649790748873825e-01, 1.458829996658279615e+00],
            [7.558383570476635560e-01, 1.022542464160217435e+00, 7.738649790748873825e-01, 7.629588167864660431e-01, 8.693506443055185606e-01],
            [1.694225410021412026e+00, 1.777975206631031035e+00, 1.458829996658279615e+00, 8.693506443055185606e-01, 2.140362235979225591e+00]])

        b = np.array(
            [3.192138092317694520e-01,
            1.421810329431174580e-01,
            5.075663514331025805e-01,
            5.376033097754365775e-01,
            3.500027406765493510e-01])

        c = 9.541640489231357769e-01

        return Q, b, c

if __name__ == "__main__":
    model = ExampleModel()
    for _ in range(10):
        mapping = model.getMapping()
        cost = model.costFn(mapping)
        print("For input: {0}, the cost was {1}".format(mapping, cost))

    print(model.random_search())
    print(model.gradient_search())
