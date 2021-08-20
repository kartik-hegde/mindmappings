import random
import itertools
import subprocess as sp
import os
import shutil
from subprocess import STDOUT
import os, sys
import numpy as np

from mindmappings.parameters import Parameters 
from mindmappings.costModel.timeloop.timeloop import Timeloop
from mindmappings.utils.utils import factors, replicate
examples=Parameters(algorithm='MTTKRP')

class Model_MTTKRP(Timeloop):
    """
        This is an object implemeted to support Timeloop on a specific architecture for CNN Layer
    """

    def __init__(self, problem=[256,256,256,256], parameters=examples, arch=examples.ARCHITECTURE):
        
        # Create the problem specifications.
        self.arch, self.problem = self.defineProblem(arch, problem)
        self.parameters = parameters
        # Generate the search space.
        self.references = self.refGen()

    def refGen(self):
        """
            Generates a search space. Might include invalid points.
        """

        numHierarchy = self.arch['numHierarchy']
        numBanks = self.arch['numBanks']
        tiled_dimensions = self.problem['dimension_names']
        bounds = self.problem['dimension_sizes'] # this should be in the same order as the above

        # Tile generation
        ref_tiles = [self.getTileGen(bounds[d], numHierarchy+1) for d in range(len(bounds))]

        # Loop order generation
        ref_loop_orders = [''.join(p) for p in list(itertools.permutations(tiled_dimensions))]

        # Partition generation (Hardcoded as of now for 5 partitions) - write a recursive function if you need generalization
        ref_partition = [[[i,j,k,b-(i+j+k)] for i in range(1, b-1, 1) for j in range(1, b-i-1, 1) for k in range(1,b-(i+j),1)] for b in numBanks]

        return ref_tiles, ref_loop_orders, ref_partition

    def checkTileValidity(self, tile_choices, mapping_partitions):
        """
            Make sure the tiles fits in the buffers. This is an optimization to prune out the space.
            Timeloop does not require this.
        """
        raise NotImplementedError

    def checkParallelValidity(self, mapping):
        """
            This check is to ensure the that assigned parallelism does not exceed the available compute.
            This is timeloop specific problem (one could imagine wrapping around if the parallelism exceeds.)
            We will reject a mapping here, in case it does not satisfy this constraint.
        """
        parHierarchy = self.arch['parallelHierarchy']

        # This makes sure that the created configuration does not have parallelism > Num PEs
        return np.prod(mapping[parHierarchy]) <= self.arch['numPEs']

    def generateOracleCost(self, metric='RAW'):
        """
            The oracle cost towards which we will guide the results towards. 
        """
        # Sophisticated Oracle (theoretical lower bound)

        # Get tensor sizes
        I, J, K, L = self.problem['dimension_sizes']
        A_size, B_size, C_size, D_size = I*J, I*K*L, K*J, L*J

        # Memory energy costs
        DRAM_cost = 200.0
        L2_cost, L1_cost = self.arch['buffer_access_energy']

        # Compute costs
        MAC_cost =  self.arch['mac_energy']
        num_flops = I*J*K*L
        num_PE = self.arch['numPEs']

        # Oracle costs per tensor per mem hierarchy
        L1_A_Cost = A_size * L1_cost
        L1_B_Cost = B_size * L1_cost
        L1_C_Cost = C_size * L1_cost
        L1_D_Cost = D_size * L1_cost
        L2_A_Cost = A_size * L2_cost
        L2_B_Cost = B_size * L2_cost
        L2_C_Cost = C_size * L2_cost
        L2_D_Cost = D_size * L2_cost
        DRAM_A_Cost = A_size * DRAM_cost
        DRAM_B_Cost = B_size * DRAM_cost
        DRAM_C_Cost = C_size * DRAM_cost
        DRAM_D_Cost = D_size * DRAM_cost
        compute_energy = num_flops * MAC_cost

        # Oracle utilization
        PE_util = 1.0

        # Energy Array (The order needs to be same as the Timeloop output)
        energy_arr = [L1_D_Cost, L1_C_Cost, L1_B_Cost, L1_A_Cost,
                        L2_D_Cost, L2_C_Cost, L2_B_Cost, L2_A_Cost,
                        DRAM_B_Cost, DRAM_C_Cost, DRAM_D_Cost, DRAM_A_Cost,
                        compute_energy]

        energy = sum(energy_arr)*1e-6
        cycles = num_flops/num_PE

        # Append the return cost array
        cost_arr = np.array(energy_arr[:-1] + [PE_util, energy, cycles])

        if(metric == 'RAW'):
            return cost_arr
        elif(metric == 'ENERGY'):
            return cost_arr[-2]*1e-6
        elif(metric == 'CYCLES'):
            return cost_arr[-1]*1e-9
        else:
            return cost_arr[-2]*cost_arr[-1]*1e-15

    def defineProblem(self, arch, bounds = [16,256,512,56]):
        """
            Define a problem.
        """

        # Arch Spec (only needed to change based on mapping)
        arch['bank_sizes'] = [arch['bufferSizes'][i]/(arch['numBanks'][i] * arch['bufferWidth'][i]) for i in range(arch['numHierarchy']-1)]

        # Define the domain
        dimensions = ['I', 'J', 'K', 'L']
        problem = {'dimension_sizes': bounds, 'dimension_names':dimensions}

        return arch, problem

    def writeConfig(self, mapping, paths, unique_ID):
        """
            This is a highly specialized version to write out a config file, get the cost and return the validity of the mapping.
        """
        OUTPUT_DIR, (CFG_FILE_OUT_ARCH, CFG_FILE_OUT_MAP, CFG_FILE_OUT_PROB, CFG_FILE_OUT_MODEL) = paths

        tiling, loop_orders, partitions =  mapping
        numHierarchy = self.arch['numHierarchy']
        I,J,K,L = self.problem['dimension_sizes']


        # Extract
        dim_factors = ['    factors: I={0} J={1} K={2} L={3}\n'.format(*tiling[i]) for i in range(numHierarchy+1)]

        # Buffer sizes
        DRAM_factors, L2_factors, spatial_factors, L1_factors = dim_factors
        DRAM_orders, L2_orders, L1_orders = loop_orders
        L2_partitions, L1_partitions = partitions

        L2_A, L2_B, L2_C, L2_D = [int(self.arch['bank_sizes'][0]*L2_partitions[i]) for i in range(4)]
        L1_A, L1_B, L1_C, L1_D = [int(self.arch['bank_sizes'][0]*L1_partitions[i]) for i in range(4)]

        # Open the sample file
        with open(self.parameters.SAMPLE_CFG_FILE, 'r') as f:
            data = f.readlines()

        # Do the replacements
        data[20] = '                depth: {0}\n'.format(L2_A)
        data[30] = '                depth: {0}\n'.format(L2_B)
        data[40] = '                depth: {0}\n'.format(L2_C)
        data[50] = '                depth: {0}\n'.format(L2_D)
        data[63] = '                    depth: {0}\n'.format(L1_A)
        data[74] = '                    depth: {0}\n'.format(L1_B)
        data[85] = '                    depth: {0}\n'.format(L1_C)
        data[96] = '                    depth: {0}\n'.format(L1_D)
        data[112] = DRAM_factors
        data[113]  = '    permutation: {0}\n'.format(DRAM_orders)
        data[117]  = L2_factors
        data[114]  = '  - permutation: {0}\n'.format(L2_orders)
        data[118]  = '  - permutation: {0}\n'.format(L2_orders)
        data[122]  = '  - permutation: {0}\n'.format(L2_orders)
        data[126]  = '  - permutation: {0}\n'.format(L2_orders)
        data[133] = spatial_factors
        data[134] = '  - permutation: {0}\n'.format(L1_orders)
        data[138] = '  - permutation: {0}\n'.format(L1_orders)
        data[142] = '  - permutation: {0}\n'.format(L1_orders)
        data[146] = '  - permutation: {0}\n'.format(L1_orders)
        data[137] = L1_factors
        data[241] = '   I: {0}\n'.format(I)
        data[242] = '   J: {0}\n'.format(J)
        data[243] = '   K: {0}\n'.format(K)
        data[244] = '   L: {0}\n'.format(L)

        data[248] = '    out_prefix: {0}'.format(unique_ID)

        # Write the file back
        with open(CFG_FILE_OUT_ARCH, 'w') as f:
            f.writelines(data[:109])
        with open(CFG_FILE_OUT_MAP, 'w') as f:
            f.writelines(data[109:215])
        with open(CFG_FILE_OUT_PROB, 'w') as f:
            f.writelines(data[215:246])
        with open(CFG_FILE_OUT_MODEL, 'w') as f:
            f.writelines(data[246:])

        os.chdir(OUTPUT_DIR)
        # print(OUTPUT_DIR)

        # Run the config file and check the validity
        command = [ self.parameters.COSTMODEL_EXECUTABLE,
                    CFG_FILE_OUT_ARCH,
                    CFG_FILE_OUT_MAP,
                    CFG_FILE_OUT_PROB,
                    CFG_FILE_OUT_MODEL
                ]
        DEVNULL = open(os.devnull, 'wb')
        prnt = sp.call(command, shell=False,stdout=DEVNULL , stderr=DEVNULL)
        # os.system("{0} {1} {2} {3} {4}".format(COSTMODEL_EXECUTABLE, CFG_FILE_OUT_ARCH, CFG_FILE_OUT_MAP, CFG_FILE_OUT_PROB, CFG_FILE_OUT_MODEL))
        if(prnt ==0):
            return True
        else:
            return False
        # try:
            # DEVNULL = open(os.devnull, 'wb')
            # prnt = sp.call(command, shell=False,stdout=DEVNULL, stderr=STDOUT)
            # print(prnt)
            # # os.system(COSTMODEL_EXECUTABLE + ' ' + CFG_FILE_OUT)
        # except:
            # return False

        return True

    def parse(self, PATH):
        """
            Parse the output file to get the stats we want.
        """
        with open(PATH, 'r') as f:
            data=f.readlines()

        # energy_IDs = [2,11, 20, 29,38,47,56,59,62]
        energy_IDs = [2,5, 8, 11,14,17,20,23,26,29,32,35]
        energy_count = 0

        cost = []

        for idx,line in enumerate(data):
            if('Energy (total)' in line):
                energy_count += 1
                if(energy_count in energy_IDs):
                    cost.append(float(data[idx].split(" ")[-2]))
                elif(energy_count > 62):
                    break
        cost.append(float(data[-28].split(" ")[-1])) # Utilization
        cost.append(float(data[-26].split(" ")[-2])) # Energy (uJ)
        cost.append(float(data[-27].split(" ")[-1])) # Cycles

        return cost

    def get_domain(self):
        """
            Problem domain
        """
        ref_tiles, ref_loop_orders, ref_partition = self.references

        domain = [
                {'name': 'It', 'type': 'discrete', 'domain': (0,len(ref_tiles[0])-1)},
                {'name': 'Jt', 'type': 'discrete', 'domain': (0,len(ref_tiles[1])-1)},
                {'name': 'Kt', 'type': 'discrete', 'domain': (0,len(ref_tiles[2])-1)},
                {'name': 'Lt', 'type': 'discrete', 'domain': (0,len(ref_tiles[3])-1)},
                {'name': 'loop_order_DRAM', 'type': 'discrete', 'domain': (0,len(ref_loop_orders)-1)},
                {'name': 'loop_order_L2',   'type': 'discrete', 'domain': (0,len(ref_loop_orders)-1)},
                {'name': 'loop_order_L1',   'type': 'discrete', 'domain': (0,len(ref_loop_orders)-1)},
                {'name': 'buffer_part_L2', 'type': 'discrete', 'domain': (0,len(ref_partition[0])-1)},
                {'name': 'buffer_part_L1', 'type': 'discrete', 'domain': (0,len(ref_partition[1])-1)}
                ]

        return domain

    def parseMetaMapping(self, meta_mapping):
        """
            Given a flat mapping, turn it to a mapping that cost model understands.
        """

        # Extracct
        numHierarchy = self.arch['numHierarchy']
        parHierarchy = self.arch['parallelHierarchy']
        bound = self.problem['dimension_sizes'] # this should be in the same order as the above
        ref_tiles, ref_loop_orders, ref_partition = self.references
        
        tiling, orders, partitions = meta_mapping[:4], meta_mapping[4:7], meta_mapping[7:9]

        tile_choices = [ref_tiles[idx][int(t)] for idx,t in enumerate(tiling)]
        mapping_tiles = [list(zip(*tile_choices))[h] for h in range(numHierarchy+1)]
        mapping_orders = [ref_loop_orders[int(o)] for o in orders]
        mapping_partitions = [ref_partition[idx][int(p)] for idx,p in enumerate(partitions)]

        return [mapping_tiles, mapping_orders, mapping_partitions]


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

    def getInputVector(self, mapping):
        """
            This function returns a flattened mapping vector.
        """
        
        # Extract
        tiling, loop_orders, partitions = mapping

        ##### Form input tuple: Hyperparameters + Mapping

        # Hyperparameters
        input_hyperparams = self.problem['dimension_sizes']
        # Tiling is represented as is
        input_tiling = [item for tile_factors in tiling for item in tile_factors]
        # Loop order is mentioned as the index of each of the dimension.
        input_loop_order = [lord.index(dim) for lord in loop_orders for dim in self.problem['dimension_names']]
        # Partition is mentioned as is.
        input_partitions = [item for partition_sizes in partitions for item in partition_sizes]

        # Club them to form input vector
        return input_hyperparams + input_tiling + input_loop_order + input_partitions