import numpy as np
from scipy.spatial.distance import cdist
import random
import os, sys

class GsearchUtils:

    def __init__(self, costmodel, parameters, minmax, id):
        self.costmodel = costmodel
        self.parameters = parameters
        self.id = id
        self.inpminmax, self.otpminmax= minmax

    def normalize(self, arr, max_min):
        """
            Returns a normalized version.
        """
        mean = max_min[:,0]
        std = max_min[:,1]
        return np.array([(arr[i]-mean[i])/float(std[i]) for i in range(len(arr))])

    def denormalize(self, arr, max_min):
        """
            Returns a denormalized version.
        """
        mean = max_min[:,0]
        std = max_min[:,1]
        return np.array([arr[i]*float(std[i])+mean[i] for i in range(len(arr))])

    def get_lower_bound(self):
        """
            The oracle cost towards which we will guide the results towards.
        """
        # Initial version is simply all 0s.
        return np.ones(self.parameters.OUTPUT_VEC_LEN)

    def generateMapping(self, normalization=None):
        """
            Generate a random valid mapping.

            if normalization is given, return with normalization.
        """
        # Uniform random sampling from the sample space and get mapping
        mapping, cost = self.costmodel.getMapCost(threadID=self.id, metric=self.parameters.COST_METRIC)

        return self.flattenMapping(mapping, normalization=normalization), cost

    def getCost(self, arr, metric=None):
        """ Return the actual cost we care about for the cost predicted by DNN."""
        # Set the metric
        metric = metric if(metric!=None) else self.parameters.COST_METRIC
        # Denoirmalize the array
        arr = self.denormalize(arr, self.otpminmax)
        # Call the cost function
        return self.costmodel.getOutputCost(arr, metric=metric)

    def flattenMapping(self, mapping, normalization=None):
        """
            Flatten and normalize.
        """
        # Club them to form input vector
        input_vector = np.array(self.costmodel.getInputVector(mapping))

        # Normalize (NOTE: We need to add hyperparams because of denorm/norm) 
        if(normalization is not None):
            input_vector = self.normalize(input_vector, normalization)

        return np.array(input_vector)

    def acceptPt(self, cost, new_cost, T):
        """
            Probablity of accepting (Like SimAnneal)
        """
        np.seterr(all='ignore')

        if(cost > new_cost):
            accept_prob = 1.0
        else:
            accept_prob = 1.0/(1.0+np.exp(float(new_cost-cost)/float(T)))
        # print("For the requested new cost {0}, changed from original {1}, the p is {2}".format(new_cost, cost, accept_prob))
        accept =accept_prob > random.random()
        # raw_input(str(accept))
        return accept

    def generateMapCost(self, metric=None):
        """
            Generates a valid mapping and returs the cost.
        """
        metric = metric if(metric!=None) else self.parameters.COST_METRIC
        # Uniform random sampling from the sample space and get mapping
        mapping, cost = self.costmodel.getMapCost(metric=metric, threadID=self.id)

        return mapping, cost

    def getProjection(self, mapping, inpminmax):
        """
            Important function that projects the current inputs to a projected mapping.
        """

        # 1. Denormalize
        mapping = self.denormalize(mapping, inpminmax)

        # 2. Get a Projection
        mapping = self.costmodel.getProjection(mapping)
        
        # Success
        if(mapping is not None):
            return mapping, True
        # Failure
        else:
            return None, False