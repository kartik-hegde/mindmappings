"""
    This file should generate random valid mappings to the planned architecture.

The parameters of the mapping:

    1. Tiling Factors
    2. Loop orders
    3. Buffer partitioning
    4. Parallelism
"""

import os
import sys
import random

from mindmappings.parameters import Parameters
from mindmappings.costModel.timeloop.model_cnn import Model_CNN
from mindmappings.costModel.timeloop.model_mttkrp import Model_MTTKRP
from mindmappings.costModel.model import Model

class TimeloopModel(Model):
    """
        This chooses the right Model based on the problem.

        Arguments:

        problem: List of problem parameterization.
        algorithm: Choice of algorithm (CNN-layer, MTTKRP are supported)
        parameters: Parameters object from the main directory

    """
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        # Parameters Object
        self.parameters = kwargs.get('parameters', Parameters())
        # Architecture
        self.arch = self.parameters.ARCHITECTURE if(self.arch == None) else self.arch
        # Algorithm
        self.algorithm = self.parameters.ALGORITHM if(self.algorithm == None) else self.algorithm
        # Problem: Should be supplied
        assert self.problem!=None, "Problem needs to be supplied."

        # Instantiate the correct model
        if(self.algorithm=='CNN-layer'):
            self.model = Model_CNN(self.problem, self.parameters, self.parameters.ARCHITECTURE)
        elif(self.algorithm=='MTTKRP'):
            self.model = Model_MTTKRP(self.problem, self.parameters, self.parameters.ARCHITECTURE)
        else:
            sys.exit("Algorithm chosen is not defined.")

        # Update them based on model
        self.arch, self.problem = self.model.arch, self.model.problem
        self.references = self.model.references

    # --- Core Functions --- #


    def costFn(self, mapping, **kwargs):
        """ 
            Return the cost of the mapping

            Required Input
            -------------
            mapping: The mapping for which the cost is to be returned

            Optional Input
            -------------
            metric: 'EDP' or 'RAW' or 'energy' or 'perf'
            threadID: Provide a unique ID for the cost estimation, useful for multi-threading
            projection: If the provided mapping is invalid, this performs a projections to closest neighbor.

            Returns
            -------------
            cost: Returns the cost of the function.

        """
        metric = kwargs.get('metric', 'EDP') 
        threadID = kwargs.get('threadID', str(random.randint(1,10e10)))
        projection = kwargs.get('projection', False)

        return self.model.costFn(mapping, metric, threadID, projection)

    def getMapping(self):
        """ 
            Gives a random mapping.

            Input
            -------------
            None

            Returns
            -------------
            mapping: A mapping object, which needs to be understood by the costFn function.

        """        
        return self.model.getMapping()

    def getMapCost(self, **kwargs):
        """ 
            Choose a random mapping and return its cost.

            Input
            -------------
            None

            Optional Input
            -------------
            metric: 'EDP' or 'RAW' or 'energy' or 'perf'
            threadID: Provide a unique ID for the cost estimation, useful for multi-threading

            Returns
            -------------
            cost
            mapping

        """
        metric = kwargs.get('metric', 'EDP') 
        threadID = kwargs.get('threadID', str(random.randint(1,10e10)))
        return self.model.getMapCost(metric, threadID)

    def getOracleCost(self, **kwargs):
        """ 
            Get the theoretical lower bound cost. This is useful to normalize the dataset across
            different problems.

            Input
            -------------
            None

            Optional Input
            -------------
            metric: 'EDP' or 'RAW' or 'energy' or 'perf'

            Returns
            -------------
            cost

        """
        metric = kwargs.get('metric', 'EDP') 
        return self.model.generateOracleCost(metric)

    def getDomain(self):
        """ 
            Get the domain of each programmable attribute.

            Input
            -------------
            None

            Returns
            -------------
            domain

        """
        return self.model.get_domain()

    def parseMapping(self, meta_mapping):
        """ 
            Parses a meta mapping, which is a mapping that only has an identifier per programmable attribute.
            Useful for Bayesian.

            Input
            -------------
            meta_mapping  

            Returns
            -------------
            mapping

        """
        return self.model.parseMetaMapping(meta_mapping)

    def getInputVector(self, mapping, **kwargs):
        """
        
            Returns an input vector for training: <mapspace_id + mapping as a vector >. This can be fed to an MLP.
            
            Input
            -------------
            mapping  

            Returns
            -------------
            a list of [*mapspace_id, *flattened_mapping]

        """
        return self.model.getInputVector(mapping)

    def getProjection(self, mapping, **kwargs):
        """ 
            Returns the projection of the mapping to valid map space.

            Input
            -------------
            mapping (can be valid or invalid)

            Returns
            -------------
            mapping (valid)

        """
        return self.model.getProjection(mapping)

    def getOutputCost(self, outvector, **kwargs):
        """
        
            Returns the cost we are optimizing, given the prediction from a DNN.
            
            Input
            -------------
            outvector (a vector of meta-stats)

            Returns
            -------------
            cost (float)

        """
        metric = kwargs.get('metric', self.parameters.COST_METRIC) 
        return self.model.getCost(outvector, metric)

    # Optional
    def getMapSpaceSize(self):
        """ 
            Returns the Size of the Map Space

            Input
            -------------
            None

            Returns
            -------------
            Size (int)

        """
        return self.model.getMapSpaceSize()

if __name__ == '__main__':
    import time
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", default='CNN-layer', required=False)
    parser.add_argument("--metric", default='EDP', required=False)
    args = parser.parse_args()

    params = Parameters(args.algorithm)
    print("\n\nChose {0} problem shape for {1} algorithm.".format(params.problem_test[0], params.ALGORITHM))
    costmodel = TimeloopModel(problem=params.problem_test[0], algorithm=args.algorithm, parameters=params)
    arch, problem, references = costmodel.arch, costmodel.problem, costmodel.references

    print("\n\nGetting a valid mapping\n")
    mapping = costmodel.getMapping()
    print("\nMapping\n------------\nTile Sizes: {0} \nLoop Orders: {1}\nPartitions: {2}".format(mapping[0], mapping[1], mapping[2]))

    print("\ncost\n------------")
    start = time.time()
    data = costmodel.costFn(mapping, metric=args.metric)
    end = time.time()

    print("Cost ({2}) was {0}. \nTime Elapsed: {1}s\n\n".format(data[0], end-start, args.metric))
