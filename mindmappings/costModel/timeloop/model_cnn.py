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
examples = Parameters(algorithm='CNN-layer')

class Model_CNN(Timeloop):
    """
        This is an object implemeted to support Timeloop on a specific architecture for CNN Layer
    """

    def __init__(self, problem=[16,256,256,3,3,14,14], parameters=examples, arch=examples.ARCHITECTURE):
        
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
        # Corner case, we want minimum two items (don't ask why, ask SKOPT guys)
        ref_tiles = [replicate(r) for r in ref_tiles]

        # Loop order generation
        ref_loop_orders = [''.join(p) for p in list(itertools.permutations(tiled_dimensions))]

        # Partition generation (Hardcoded as of now for 3 partitions) - write a recursive function if you need generalization
        ref_partition = [list(([i,j,b-(i+j)] for i in range(1, b, 1)
                                                            for j in range(1, b-i, 1))) for b in numBanks]
        return ref_tiles, ref_loop_orders, ref_partition

    def checkTileValidity(self, tile_choices, mapping_partitions):
        """
            Make sure the tiles fits in the buffers. This is an optimization to prune out the space.
            Timeloop does not require this.
        """
        # print(tile_choices, mapping_partitions)
        L2_partitions, L1_partitions = mapping_partitions

        # Buffer sizes
        L2_input, L2_weight, L2_psum = [self.arch['bank_sizes'][0]*L2_partitions[i] for i in range(3)]
        L1_input, L1_weight, L1_psum = [self.arch['bank_sizes'][1]*L1_partitions[i] for i in range(3)]

        # Tile sizes
        N,C,K,R,S,P,Q = [np.prod(list(zip(*tile_choices))[i][1:]) for i in range(7)]
        L2_input_tile, L2_weight_tile, L2_psum_tile = N*(P+R-1)*(Q+S-1)*C, K*R*S*C, P*Q*K*N
        N,C,K,R,S,P,Q = [np.prod(list(zip(*tile_choices))[i][2:]) for i in range(7)]
        L1_input_tile, L1_weight_tile, L1_psum_tile = N*(P+R-1)*(Q+S-1)*C, K*R*S*C, P*Q*K*N

        if( (L2_input_tile>L2_input) or (L2_weight_tile>L2_weight) or (L2_psum_tile>L2_psum) or
            (L1_input_tile>L1_input) or (L1_weight_tile>L1_weight) or (L1_psum_tile>L1_psum)):
            return False
        else:
            return True

    def generateOracleCost(self, metric='RAW'):
        """
            The oracle cost towards which we will guide the results towards. 
        """
        # Sophisticated Oracle (theoretical lower bound)

        # Get tensor sizes
        N,C,K,R,S,P,Q = self.problem['dimension_sizes']
        input_size, weight_size, output_size = [N*P*Q*C, R*S*C*K, N*P*Q*K] # Input, weight, output

        # Memory energy costs
        DRAM_cost = 200.0
        L2_buf, L1_buf = self.arch['buffer_access_energy']

        # Compute costs
        MAC_cost =  self.arch['mac_energy']
        num_flops = N*R*S*C*P*Q*K
        num_PE = self.arch['numPEs']

        # Oracle costs per tensor per mem hierarchy
        L1_input_energy = input_size * L1_buf
        L1_weight_energy = weight_size * L1_buf
        L1_output_energy = output_size * L1_buf
        L2_input_energy = input_size * L2_buf
        L2_weight_energy = weight_size * L2_buf
        L2_output_energy = output_size * L2_buf
        DRAM_input_energy = input_size * DRAM_cost
        DRAM_weight_energy = weight_size * DRAM_cost
        DRAM_output_energy = output_size * DRAM_cost
        compute_energy = num_flops * MAC_cost
        PE_util = 1.0
        energy = sum([L1_input_energy,L1_weight_energy,L1_output_energy,
                        L2_input_energy,L2_weight_energy,L2_output_energy,
                            DRAM_input_energy, DRAM_weight_energy, DRAM_output_energy,
                            compute_energy]) * 1e-6
        cycles = num_flops/num_PE

        cost_arr = np.array([L1_input_energy,L1_weight_energy,L1_output_energy,
                            L2_output_energy,L2_weight_energy,L2_input_energy,
                            DRAM_weight_energy, DRAM_input_energy, DRAM_output_energy,
                            PE_util,energy,cycles])

        if(metric == 'RAW'):
            return cost_arr
        elif(metric == 'ENERGY'):
            return cost_arr[-2]*1e-6
        elif(metric == 'CYCLES'):
            return cost_arr[-1]*1e-9
        else:
            return cost_arr[-2]*cost_arr[-1]*1e-15

    def defineProblem(self, arch, bounds = [16,256,512,3,3,56,56]):
        """
            Define a problem.
        """

        # Arch Spec (only needed to change based on mapping)
        arch['bank_sizes'] = [arch['bufferSizes'][i]/(arch['numBanks'][i] * arch['bufferWidth'][i]) for i in range(arch['numHierarchy']-1)]

        # Define the domain
        dimensions = ['N','C','K','R','S','Q','P']
        problem = {'dimension_sizes': bounds, 'dimension_names':dimensions}

        return arch, problem

    def writeConfig(self, mapping, paths, unique_ID):
        """
            This is a highly specialized version to write out a config file, get the cost and return the validity of the mapping.
        """
        OUTPUT_DIR, (CFG_FILE_OUT_ARCH, CFG_FILE_OUT_MAP, CFG_FILE_OUT_PROB, CFG_FILE_OUT_MODEL) = paths

        tiling, loop_orders, partitions =  mapping
        numHierarchy = self.arch['numHierarchy']
        N,C,K,R,S,P,Q = self.problem['dimension_sizes']


        # Extract
        dim_factors = ['    factors: N={0} C={1} K={2} R={3} S={4} P={5} Q={6}\n'.format(*tiling[i]) for i in range(numHierarchy+1)]

        # Buffer sizes
        DRAM_factors, L2_factors, spatial_factors, L1_factors = dim_factors
        DRAM_orders, L2_orders, L1_orders = loop_orders
        L2_partitions, L1_partitions = partitions

        L2_input, L2_weight, L2_psum = [int(self.arch['bank_sizes'][0]*L2_partitions[i]) for i in range(3)]
        L1_input, L1_weight, L1_psum = [int(self.arch['bank_sizes'][1]*L1_partitions[i]) for i in range(3)]

        # Open the sample file
        with open(self.parameters.SAMPLE_CFG_FILE, 'r') as f:
            data = f.readlines()

        # Do the replacements
        data[20] = '                depth: {0}\n'.format(L2_input)
        data[30] = '                depth: {0}\n'.format(L2_weight)
        data[40] = '                depth: {0}\n'.format(L2_psum)
        data[53] = '                    depth: {0}\n'.format(L1_input)
        data[64] = '                    depth: {0}\n'.format(L1_weight)
        data[75] = '                    depth: {0}\n'.format(L1_psum)
        data[91] = DRAM_factors
        data[92]  = '    permutation: {0}\n'.format(DRAM_orders)
        data[104]  = L2_factors
        data[93]  = '  - permutation: {0}\n'.format(L2_orders)
        data[97]  = '  - permutation: {0}\n'.format(L2_orders)
        data[101]  = '  - permutation: {0}\n'.format(L2_orders)
        data[108] = spatial_factors
        data[109] = '  - permutation: {0}\n'.format(L1_orders)
        data[113] = '  - permutation: {0}\n'.format(L1_orders)
        data[117] = '  - permutation: {0}\n'.format(L1_orders)
        data[120] = L1_factors
        data[201] = '   C: {0}\n'.format(C)
        data[202] = '   K: {0}\n'.format(K)
        data[203] = '   R: {0}\n'.format(R)
        data[204] = '   S: {0}\n'.format(S)
        data[205] = '   P: {0}\n'.format(P)
        data[206] = '   Q: {0}\n'.format(Q)
        data[207] = '   N: {0}\n'.format(N)

        data[211] = '    out_prefix: {0}'.format(unique_ID)

        # Write the file back
        with open(CFG_FILE_OUT_ARCH, 'w') as f:
            f.writelines(data[:88])
        with open(CFG_FILE_OUT_MAP, 'w') as f:
            f.writelines(data[88:164])
        with open(CFG_FILE_OUT_PROB, 'w') as f:
            f.writelines(data[164:209])
        with open(CFG_FILE_OUT_MODEL, 'w') as f:
            f.writelines(data[209:])

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

        energy_IDs = [2,5, 8, 11,14,17,20,23,26]
        energy_count = 0

        cost = []

        for idx,line in enumerate(data):
            if('Energy (total)' in line):
                energy_count += 1
                if(energy_count in energy_IDs):
                    cost.append(float(data[idx].split(" ")[-2]))
                elif(energy_count > 62):
                    break
        cost.append(float(data[-24].split(" ")[-1]))
        cost.append(float(data[-22].split(" ")[-2]))
        cost.append(float(data[-23].split(" ")[-1]))

        return cost

    def get_domain(self):
        """
            Problem domain
        """
        ref_tiles, ref_loop_orders, ref_partition = self.references

        domain = [
                {'name': 'Nt', 'type': 'discrete', 'domain': (0,len(ref_tiles[0])-1)},
                {'name': 'Ct', 'type': 'discrete', 'domain': (0,len(ref_tiles[1])-1)},
                {'name': 'Kt', 'type': 'discrete', 'domain': (0,len(ref_tiles[2])-1)},
                {'name': 'Rt', 'type': 'discrete', 'domain': (0,len(ref_tiles[3])-1)},
                {'name': 'St', 'type': 'discrete', 'domain': (0,len(ref_tiles[4])-1)},
                {'name': 'Pt', 'type': 'discrete', 'domain': (0,len(ref_tiles[5])-1)},
                {'name': 'Qt', 'type': 'discrete', 'domain': (0,len(ref_tiles[6])-1)},
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

        tiling, orders, partitions = meta_mapping[:7], meta_mapping[7:10], meta_mapping[10:12]

        tile_choices = [ref_tiles[idx][int(t)] for idx,t in enumerate(tiling)]
        mapping_tiles = [list(zip(*tile_choices))[h] for h in range(numHierarchy+1)]
        mapping_orders = [ref_loop_orders[int(o)] for o in orders]
        mapping_partitions = [ref_partition[idx][int(p)] for idx,p in enumerate(partitions)]

        return [mapping_tiles, mapping_orders, mapping_partitions]

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