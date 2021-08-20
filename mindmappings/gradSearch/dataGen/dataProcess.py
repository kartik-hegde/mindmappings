"""
    This module reads all the data and normalizes it.

"""

import numpy as np
import os, sys
import pickle
import random
import math

from mindmappings.parameters import Parameters
from mindmappings.utils.parallelProcess import parallelProcess


class DataProcess:

    def __init__(self, path=None, out_path=None, parameters=Parameters(), num_outfiles=None) -> None:
        self.parameters = parameters
        self.path = parameters.DATASET_UNPROCESSED_PATH if(path==None) else path
        self.out_path = parameters.DATASET_PATH if(out_path==None) else out_path
        self.num_outfiles = parameters.DATASET_NUM_FILES_PER_THREAD if(num_outfiles==None) else num_outfiles

    def getData(self, path):
        files = []
        for npy in os.listdir(path):
            if('npy' in npy.rstrip().split(".")[-1]):
                files.append(npy)
        return files

    def getMeanData(self, arr):
        return np.mean(arr), np.std(arr)

    def normalize(self, arr, mean, std):
        return (arr-mean)/std

    def denormalize(self, arr, meanstd):
        mean = meanstd[:,0]
        std = meanstd[:,1]

        return [arr[idx]*std[idx]+mean[idx] for idx in range(len(arr))]

    def run(self):

        # Go to that directory (TODO: Remove this, currently for multi-threading, this is useful)
        os.chdir(self.path)

        # Get all the files
        dataiter = self.getData(self.path)

        NUM_FILES = min(len(dataiter),self.parameters.DATASET_NUM_FILES_PER_THREAD)
        # We will perform this in iterations.
        numIters = int(len(dataiter)/NUM_FILES)

        # Placeholder
        meanData = [None, None]

        # Parallel processing
        work = [(dataiter[iters*NUM_FILES:(iters+1)*NUM_FILES], iters*NUM_FILES,self.out_path) for iters in range(numIters)]
        Processed = parallelProcess(self.dataPreProcessUnpack, work, num_cores=self.parameters.DATASET_NUM_THREADS)

        for iters in range(numIters):
            inp_meanstd, out_meanstd = Processed[iters]
            if(iters == 0):
                meanData = [inp_meanstd, out_meanstd]
            else:
                meanData = [np.mean((inp_meanstd,meanData[0]),axis=0),np.mean((out_meanstd,meanData[1]), axis=0)]

        print("Writing the data out")
        # Save the mean and std information
        np.save(os.path.join(self.out_path, self.parameters.MEANSTD), meanData)

        return None

    def dataPreProcessUnpack(self,args):
        return self.dataPreProcess(*args)

    def dataPreProcess(self, npys, offset, out_path):

        print(npys)
        global_data = []
        for npy in npys:
            print("Picking up ", npy, os.getcwd())
            global_data += list(np.load(npy,  allow_pickle=True))

        # extract the number of inputs and outputs
        inp_features = len(global_data[0][0])
        out_features = len(global_data[0][1])

        # Np array by slicing
        global_data = np.array(global_data)

        # Extract
        inp_arr = np.array([xi for xi in global_data[:,0]])
        out_arr = np.array([xi for xi in global_data[:,1]])

        print("\n\nRead Done")

        ### TEST ####
        # print("Input", inp_arr[0])
        # print("Output", out_arr[0])
        # print("Features", inp_features,out_features)

        inp_meanstd = []
        # Get mean and std deviation
        for i in range(inp_features):
            inp_meanstd.append(self.getMeanData(inp_arr[:,i]))
        inp_meanstd = np.array(inp_meanstd)

        out_meanstd = []
        # Get mean and std deviation
        for i in range(out_features):
            out_meanstd.append(self.getMeanData(out_arr[:,i]))
        out_meanstd = np.array(out_meanstd)


        print("Performing Normalization")
        # Normalize the arrays
        inp_arr_mod = np.array([self.normalize(inp_arr[:,i], inp_meanstd[i][0], inp_meanstd[i][1]) for i in range(inp_features)])
        # Reshape the array
        inp_arr = inp_arr_mod.transpose()

        out_arr_mod = np.array([self.normalize(out_arr[:,i], out_meanstd[i][0], out_meanstd[i][1]) for i in range(out_features)])
        # Reshape the array
        out_arr = out_arr_mod.transpose()

        # Put all the arrays together and shuffle
        final_data = np.array([[inp_arr[i], out_arr[i]] for i in range(len(inp_arr))])
        print("Performing Shuffling")
        np.random.shuffle(final_data)

        # Write
        total_writes = len(final_data)/self.num_outfiles
        print("Each file will have %d entries" % total_writes)
        for j in range(self.num_outfiles):
            print("Writing file %d" % j)
            base = int(j*total_writes)
            bound = int((j+1)*total_writes)
            print("Writing from {0} to {1}".format(base, bound))
            file_name = 'shuffled_data' + str(j+offset) + '.npy'
            np.save(os.path.join(out_path, file_name), final_data[base:bound])

        return (inp_meanstd, out_meanstd)

if __name__ == '__main__':

    preprocess = DataProcess(path=sys.argv[1], out_path=sys.argv[2], num_outfiles=1)
    preprocess.run()
    print("Done. Saving.")
