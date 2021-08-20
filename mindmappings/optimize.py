import argparse
from collections import OrderedDict 
import os
import sys
import shutil
from copy import deepcopy
import numpy as np

from mindmappings.parameters import Parameters
from mindmappings.utils.plot_graph import plot_graph
from mindmappings.utils.utils import non_increasing
from mindmappings.utils.parallelProcess import parallelProcess
from mindmappings.gradSearch.dataGen.dataGen import DataGen
from mindmappings.gradSearch.dataGen.dataProcess import DataProcess
from mindmappings.gradSearch.train.train_surrogate import TrainSurrogate
from mindmappings.gradSearch.search.search import Tuner

# Unpacker
def search_unpack(args):
    kwargs, obj = args
    return obj.search(**kwargs)

def main(args):
    """ Main Function """

    # Set the algorithm
    parameters = Parameters(args.algorithm)

    # Choose the cost model
    if(args.costmodel == 'example'):
        from mindmappings.costModel.example.example_model import ExampleModel as Model
        # We will set simpler parameters for example
        
    elif(args.costmodel == 'timeloop'):
        from mindmappings.costModel.timeloop.model_timeloop import TimeloopModel as Model
    else:
        sys.exit("Cost Model {0} unknown".format(args.costmodel))

    # Data generation
    if(args.command == 'datagen'):
        datagen_path = args.path if(args.path!=None) else parameters.DATASET_UNPROCESSED_PATH
        if(not os.path.isdir(datagen_path)):
            os.mkdir(datagen_path)
        print("\n\nLaunching Data Gen. Writing to {0}".format(datagen_path))
        datagen = DataGen(Model, parameters=parameters, path=datagen_path)
        datagen.run()
        print("\n\n Done!")

    # Data processing
    elif(args.command == 'dataprocess'):
        datagen_path = args.path if(args.path!=None) else parameters.DATASET_UNPROCESSED_PATH
        data_path = args.path[:-1]+'_processed/' if(args.path[-1] == '/') else args.path + '_processed/' if(args.path!=None) else parameters.DATASET_PATH
        # Input path
        if(not os.path.isdir(datagen_path)):
            sys.exit("Unprocessed data not found")
        # Output path
        if(not os.path.isdir(data_path)):
            os.mkdir(data_path)
        print("\n\nLaunching Data Process. Writing to {0}".format(data_path))
        preprocess = DataProcess(parameters=parameters, path=datagen_path, out_path=data_path)
        preprocess.run()
        shutil.move(data_path+ '/' + parameters.MEANSTD+'.npy', data_path + '/' + parameters.MEANSTD)
        print("\n\nDone!")

    # Training
    elif(args.command == 'train'):
        data_path = args.path if(args.path!=None) else parameters.DATASET_PATH
        saved_path = args.path[:-1]+'_saved_model/' if(args.path[-1] == '/') else args.path + '_saved_model/' if(args.path!=None) else parameters.MODEL_SAVE_PATH
        if(not os.path.isdir(data_path)):
            sys.exit("Dataset not found")
        if(not os.path.isdir(saved_path)):
            os.mkdir(saved_path)
        shutil.copy(data_path+ '/' + parameters.MEANSTD, saved_path + '/' + parameters.MEANSTD)
        surrogate = TrainSurrogate(parameters=parameters, dataset_path=data_path, saved_model_path=saved_path)
        surrogate.trainer()

    # Search
    elif(args.command == 'search'):
        costmodel = Model(problem=args.problem, algorithm=args.algorithm, parameters=parameters)
        tuner = Tuner(costmodel, parameters=parameters, dataset_path=None, saved_model_path=args.path)
        tuner.search(maxsteps=args.maxsteps)

    # Benchmark time per step
    elif(args.command == 'benchmark'):
        costmodel = Model(problem=args.problem, algorithm=args.algorithm, parameters=parameters)
        tuner = Tuner(costmodel, parameters=parameters, dataset_path=None, saved_model_path=args.path)
        tuner.search(maxsteps=args.maxsteps, benchmarking=True)

    # Reproduce the results from the paper
    elif(args.command == 'reproduce'):
        assert args.costmodel == 'timeloop', "Timeloop cost model was used in the paper."
        
        print("This will reproduce similar results from the paper for {0}.\
                \nRunning the following problems {1}. \n\n See parameters.py if you would like to change something.\n\
                \nNOTE: This run takes longer since we will plot results with ground truth for every step, hence actual cost model (timeloop) is invoked every step. \
                \nIn practice, this is not necessary and search is blazing fast (try search command)\n"\
                .format(args.algorithm,parameters.problems))

        # Create the output directory
        if(not os.path.exists(parameters.GSEARCH_OUTPATH)):
            os.mkdir(parameters.GSEARCH_OUTPATH)
        outfile = parameters.GSEARCH_OUTPATH + 'dataGSearch_isoiter_' + str(parameters.ALGORITHM) + '.npy'
        print("Writing to ", outfile)

        # Placeholders
        average_cost = []
        standard_deviation = []
        oracle_costs = []
        work = []
        # Average Number of runs
        n = parameters.GSEARCH_AVG_ITERS
        # Create Work
        for pid,problem in enumerate(parameters.problems): 

            # Instantiate the cost model
            costmodel = deepcopy(Model(problem=args.problem, algorithm=args.algorithm, parameters=parameters))
            # Instantiate Tuner
            tuner = deepcopy(Tuner(costmodel, parameters=parameters, dataset_path=None, saved_model_path=args.path))
            # Arguments for every call
            work_problem = [({'threadID':str(pid*n+i), 'maxsteps':parameters.GSEARCH_MAXSTEPS, 'reproduce':True}, tuner) for i in range(n)]
            # Add them to main work
            work = work + work_problem

        # Total Amount of threads
        print("We will launch {0} total threads.".format(len(work)))
        # Launch them in parallel (uses maximum available cores/GPU)
        costArr = parallelProcess(search_unpack, work, num_cores=8)

        # Slice the results and average it out.
        average_cost = [np.minimum.accumulate(np.mean(costArr[i:i+n], axis=0)) for i in range(0, len(costArr), n)]
        standard_deviation = [np.minimum.accumulate(np.std(costArr[i:i+n], axis=0)) for i in range(0, len(costArr), n)]

        # Ratio w.r.t. oracle
        # average_cost = [p/oracle_costs[idx] for idx,p in enumerate(average_cost)]

        # Dump the data into a npy
        np.save(outfile, [average_cost, standard_deviation])

        print("Iso-iteration Runs Completed. Results written to {0}".format(parameters.GSEARCH_OUTPATH))

    else:
        sys.exit("Command Not understood.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", help="CNN-layer or MTTKRP", default='CNN-layer')
    parser.add_argument("--problem", help="Enter the problem dimensions", nargs='+', default=[16,256,256,3,3,14,14], type=int)
    parser.add_argument("--costmodel", help="example or timeloop", default='timeloop')
    parser.add_argument("-c", "--command", help="datagen or dataprocess or train or search", required=True)
    parser.add_argument("-p", "--path", help="Path to store results in", default=None)
    parser.add_argument("--avg_runs", help="Number of average runs", default=1, type=int)
    parser.add_argument("--maxsteps", help="Number of steps", default=10, type=int)
    args = parser.parse_args()

    # Run
    main(args)
