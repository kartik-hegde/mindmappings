import random
import os,sys
import numpy as np
import multiprocessing

from mindmappings.parameters import Parameters
from mindmappings.utils.parallelProcess import parallelProcess

class DataGen:

    def __init__(self, model, parameters=Parameters(), path=None, num_files=None, samples_per_file=None, samples_per_problem=None):
        self.model = model
        self.parameters = parameters
        self.path = parameters.DATASET_UNPROCESSED_PATH if(path==None) else path
        self.num_files = parameters.DATASET_NUMFILES if(num_files==None) else num_files
        self.samples_per_file = parameters.DATASET_NUMSAMPLES_FILE if(samples_per_file==None) else samples_per_file
        self.samples_per_problem = parameters.DATASET_MAPPINGS_PER_PROBLEM if(samples_per_problem==None) else samples_per_problem

    def getDataset(self, index):
        """
            1. Creates a random problem.
            2. Samples a mapping from that.
            3. Generates data from that.

            Above steps are iterated until the number of samples requested are reached.
        """

        data_arr = []
        print("We will run {0} problems, with {1} mappings in each.".format(self.samples_per_file // self.samples_per_problem, self.samples_per_problem))
        for idx in range(self.samples_per_file // self.samples_per_problem):

            # Create a random problem
            bounds = self.parameters.random_problem_gen()

            # Instantiate a cost model
            costmodel = self.model(problem=bounds, parameters=self.parameters)

            # For the given problem, generate oracle cost
            oracle_cost = costmodel.getOracleCost(metric='RAW')

            # Print Progress
            if(idx !=0):
                print("{0} problems completed for {1}".format(idx, index))
            threadID = str(multiprocessing.current_process()._identity[0])

            for n in range(self.samples_per_problem):
                # print("Done %d in %d", n, threadID)
                success = False
                # mapping, cost = costmodel.getMapCost(metric='RAW', threadID=threadID)

                # Get a random mapping and cost
                while not success:
                    try:
                        # Uniform random sampling from the sample space and get cost
                        mapping, cost = costmodel.getMapCost(metric='RAW', threadID=threadID)
                        success = True
                    except Exception as e:
                        print(e)
                        success = False

                # Generate input vector
                input_vector = costmodel.getInputVector(mapping)
                
                # Cost vector is normalized to the oracle cost
                cost = [cost[i]/float(oracle_cost[i]) for i in range(len(cost))]

                data_arr.append([np.array(input_vector), np.array(cost)])

                # Print Progress
                if(n%100 == 0):
                    print("{0} mappings, {1} problems completed for {2}".format(n, idx, threadID))

        name = self.path + 'data_' + str(index) + '.npy'
        np.save(name, data_arr)
        print("Wrote to " + name)

        return  None

    def run(self):
        """
            Main File to generate data.
        """
        # Setup Path
        if(not os.path.isdir(self.path)):
            print("Creating the dataset path at {0}".format(self.path))
            os.mkdir(self.path)

        # Call threads in parallel to write the data
        # Processed = Parallel(n_jobs=-1)(delayed(getDataset)(path, ind, samplesperFile) for ind in range(numFiles))
        work = [ind for ind in range(self.num_files)]
        parallelProcess(self.getDataset, work, num_cores=None)

        print("All Done!")

        return None

if __name__ == '__main__':
    from mindmappings.costModel.timeloop.model_timeloop import TimeloopModel
    print("Example run: python dataGen.py <path to write> <total files><samples per file><samples per problem>")
    datagen = DataGen(TimeloopModel, Parameters(), sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    datagen.run()
