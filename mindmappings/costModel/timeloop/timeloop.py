import random
import itertools
import subprocess as sp
import os
import shutil
from subprocess import STDOUT
import os, sys
import numpy as np
from scipy.spatial.distance import cdist

# from mindmappings.parameters import parameters
from mindmappings.utils.parallelProcess import parallelProcess
from mindmappings.utils.utils import factors

class Timeloop:
    """
        Super class that contains common routines for any Timeloop based models.
    """
    def checkParallelValidity(self, tiling, flatten=False):
        """
            This check is to ensure the that assigned parallelism does not exceed the available compute.
            This is timeloop specific problem (one could imagine wrapping around if the parallelism exceeds.)
            We will reject a mapping here, in case it does not satisfy this constraint.
        """
        # Get all the hierarchy data
        parHierarchy = self.arch['parallelHierarchy']

        # This makes sure that the created configuration does not have parallelism > Num PEs
        return np.prod(tiling[parHierarchy]) <= self.arch['numPEs']

    def getTileGen(self, dimension, numHierarchy):
        return [p for p in itertools.product(factors(dimension), repeat=numHierarchy) if(np.prod(p)==dimension)]

    def getPaths(self, unique_ID):
        """
            Create a random directory, move there and perform. (To enable parallelism).
            Make sure it does not exist already
        """
        # unique_ID = str(random.randint(1,10e10))

        # Append the threadID to create a unique directory
        out_dir =  self.parameters.OUTPUT_DIR_BASE + unique_ID
        if(not os.path.exists(out_dir)):
            os.mkdir(out_dir)
        cfg_out = out_dir + '/arch.yaml', out_dir + '/map.yaml', out_dir + '/prob.yaml', out_dir + '/model.yaml'

        return out_dir, cfg_out, os.path.join(out_dir, str(unique_ID)+ self.parameters.OUTPUT_FILE)
        # return out_dir, cfg_out, os.path.join(out_dir, 'timeloop-model'+ parameters.OUTPUT_FILE)

    def getMapping(self):
        """
            This is a mandatory function to be defined for a cost model.
            It should return a random valid mapping from the search space.
        """

        numHierarchy = self.arch['numHierarchy']
        parHierarchy = self.arch['parallelHierarchy']
        bound = self.problem['dimension_sizes'] # this should be in the same order as the above
        ref_tiles, ref_loop_orders, ref_partition = self.references

        # Fill in the mapping template

        while(True):

            # Make sure the parallelism is well defined - this returns a valid mapping.
            tile_choices = [random.choice(ref_tiles[d]) for d in range(len(bound))]
            mapping_tiles = [list(zip(*tile_choices))[h] for h in range(numHierarchy+1)]
            mapping_partitions = [random.choice(ref_partition[h]) for h in range(numHierarchy-1)]

            # Check for parallelism.
            validity = self.checkParallelValidity(mapping_tiles)

            # This call is is made based on the flag from parameters.
            if(self.parameters.CHECK_TILE_VALIDITY):
                validity &= self.checkTileValidity(mapping_tiles, mapping_partitions)

            # Break the loop if the chosen mapping is good.
            if(validity):
                break

        # Loop order return
        mapping_orders = [random.choice(ref_loop_orders) for h in range(numHierarchy)]

        return [mapping_tiles, mapping_orders, mapping_partitions]

    def getMapSpaceSize(self):
        """
            Gets the size of the map space.
        """
        numHierarchy = self.arch['numHierarchy']
        parHierarchy = self.arch['parallelHierarchy']
        bound = self.problem['dimension_sizes'] # this should be in the same order as the above
        # tiles
        ref_tiles, ref_loop_orders, ref_partition = self.references
        tile_choices = [len(ref_tiles[d]) for d in range(len(bound))]
        partitions_choices = [len(ref_partition[h]) for h in range(numHierarchy-1)]
        order_choices = [len(ref_loop_orders) for h in range(numHierarchy)]
        all_choices = tile_choices + partitions_choices + order_choices
        total_choices = 1
        for c in all_choices:
            total_choices *= c

        # Check validity
        valid_tiles = 0
        total_tiles = 1000000
        for _ in range(total_tiles):
            tile_choices = [random.choice(ref_tiles[d]) for d in range(len(bound))]
            mapping_tiles = [list(zip(*tile_choices))[h] for h in range(numHierarchy+1)]
            if(self.checkParallelValidity(mapping_tiles)):
                valid_tiles += 1
        from math import ceil,log10

        # Contains valids stuff
        mapspace_size = (valid_tiles/total_tiles)*total_choices
        print("Relaxed Mapping Space is of approx size 10^{0}, with around {1} percent of them valid = {2}.".format(ceil(log10(total_choices)),100*valid_tiles/total_tiles, ceil(log10(mapspace_size))))

    def getProjection(self, mapping):
       
        # extract
        numHierarchy = self.arch['numHierarchy'] + 1
        numDims = len(self.problem['dimension_names'])
        ref_partition = self.references[-1]
        num_partitions = len(ref_partition[0][0])
    

        # 2. Extract
        tiling, loop_orders, partitions = mapping[self.parameters.TILING_IDX:self.parameters.LOOP_ORDER_IDX], \
                                            mapping[self.parameters.LOOP_ORDER_IDX:self.parameters.PARTITION_IDX], \
                                                mapping[self.parameters.PARTITION_IDX:]

        # Projection main procedure


        
        # #######  2. Loop Orders #######

            # We will simply sort the values and order the dimensions accordingly.
        loop_orders = [loop_orders[numDims*idx:numDims*(idx+1)] for idx in range(numHierarchy-1)]
        loop_orders = [list(np.argsort(loop_order)) for loop_order in loop_orders]
        loop_orders = [''.join([self.problem['dimension_names'][idx] for idx in loop_order]) for loop_order in loop_orders]

        # #######  3. Partitions #######

            # Here we will find the Euclidean distances to all the reference tiles per dimension, and choose the one
            # that has the minimum distance. --- strategy: Nearest Neighbor

        # First extract all partitions
        partitions = [partitions[idx*num_partitions:(idx+1)*num_partitions] for idx in range(numHierarchy-2)]
        # Get the partitions with minimum Euclidean distance
        partitions = [ref_partition[idx][np.argmin(cdist(ref_partition[idx], [partition,], metric='euclidean'))] for idx,partition in enumerate(partitions)]
        # Flatten it
        # partitions = [item for partition_sizes in partitions for item in partition_sizes]

        # #######  1. Tiling #######
            # Here we will find the Euclidean distances to all the reference tiles per dimension, and choose the one
            # that has the minimum distance. --- strategy: Nearest Neighbor

        # First, extract dimension wise tiling for memory hierarchies
        tiling = [tiling[numDims*h:numDims*(h+1)] for h in range(numHierarchy)]

        # Reference tiles
        ref_tiles = self.references[0]

        # Flatten
        tiling = [[tiling[h][idx] for h in range(numHierarchy)] for idx in range(numDims)]

        # Get all the tiles that form the minimum Euclidean distance
        distances = [np.reshape(cdist(ref_tiles[idx], [tiling[idx],], metric='euclidean'), (1,len(ref_tiles[idx]))) for idx in range(numDims)]
        indices = [np.argsort(dist[0]) for dist in distances]
        headers = [0]*numDims

        # Go in a loop till you find the closest valid tiling
        while True:
            tiling = [ref_tiles[idx][int(indices[idx][headers[idx]])] for idx in range(numDims)]
            tiling = [list(zip(*tiling))[h] for h in range(numHierarchy)]
            if(self.checkParallelValidity(tiling)):
                break
            else:
                target = np.argsort([distances[idx][0][indices[idx][headers[idx]]] for idx in range(numDims)])
                choice = 0
                validity = [idx for idx in range(numDims) if(headers[idx]+1 < len(ref_tiles[idx]))]
                if(not validity):
                    return None
                while True:
                    if(target[choice] in validity):
                        headers[target[choice]]+=1
                        break
                    else:
                        choice += 1

        return [tiling, loop_orders, partitions]
        

    def getCost(self, arr, metric='EDP'):
        """ Returns the cost we care about."""

        if(metric == 'EDP'):
            return arr[-2]*arr[-1]
        elif(metric == 'ENERGY'):
            return arr[-2]
        elif(metric == 'CYCLES'):
            return arr[-1]
        else:
            print("Cost Metric Not Understood.")
            return arr

    def costFn(self, mapping, metric='EDP', threadID=str(random.randint(1,10e10)), projection=False):
        """
            Return the actual cost.
        """
        # Extract and make sure it is a valid mapping, if not return a false.
        tiling, loop_orders, partitions =  mapping

        validity = True

        #Early exit - Now redundant since we support invalid tiles with higher cost.
        if(self.parameters.CHECK_TILE_VALIDITY):
            if(not self.checkTileValidity(tiling, partitions)):
                validity = False

        # Check if the mapping is invalid in terms of parallelism
        if(not self.checkParallelValidity(mapping[0])):
            validity = False

        # Attempt to perform a projection (closest valid)
        if(projection and not validity):
            tiling = self.getProjection(tiling)
            # Could not project
            if(tiling == None):
                return np.inf, False
            else:
                mapping[0] = tiling
        elif(not validity):
            return np.inf, False

        # Creating random directories.
        OUTPUT_DIR, CFG_FILE_OUT, RESULT = self.getPaths(threadID)

        # Write mapping to a timeloop config
        success = self.writeConfig(mapping, (OUTPUT_DIR,CFG_FILE_OUT), threadID)

        if(success):
            # Energy as the cost.
            cost = self.parse(RESULT)
            if(metric == 'EDP'):
                # Parse the output file and return cost (EDP)
                cost = cost[-2] * cost[-1] * 1e-15
            elif(metric == 'energy'):
                cost = cost[-2]*1e-6
            elif(metric == 'perf'):
                cost = cost[-1]*1e-9
            elif(metric == 'RAW'):
                cost = list(cost)
            else:
                sys.exit("Cost metric not understood")
        else:
            cost = np.inf

        # Remove the temp directory
        # print(OUTPUT_DIR)
        shutil.rmtree(OUTPUT_DIR)

        return cost, success

    def getMapCost(self, metric='EDP', threadID=''):
        """
            Generates a valid mapping and always returns a good mapping and cost.
        """
        success = False

        while(not success):
            mapping = self.getMapping()
            cost, success = self.costFn(mapping, metric, threadID)
            # print("SUCCESS" if success else "FAILED")
        return mapping, cost