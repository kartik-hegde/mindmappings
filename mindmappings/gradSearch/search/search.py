import torch
import math
import time
import csv
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import parameter
import torch.nn.functional as F
from torch.multiprocessing import Pool, Process, set_start_method, Lock, cpu_count
import sys, os
import numpy as np
from copy import copy, deepcopy

from mindmappings.parameters import Parameters
from mindmappings.utils.parallelProcess import parallelProcess
from mindmappings.costModel.timeloop.model_timeloop import TimeloopModel as Model
from mindmappings.gradSearch.gradSearchUtils import GsearchUtils
from mindmappings.gradSearch.train.train_surrogate import Net

# dtype = torch.cuda.float if torch.cuda.is_available() else torch.float

class Tuner:

    def __init__(self, costmodel, parameters=Parameters(), dataset_path=None, saved_model_path=None) -> None:
        """Init"""
        self.costmodel = costmodel
        self.parameters = parameters
        self.dataset_path = parameters.DATASET_PATH if(dataset_path==None) else dataset_path
        self.saved_model_path = parameters.MODEL_SAVE_PATH if(saved_model_path==None) else saved_model_path
        # setting device on GPU if available, else CPU
        # torch.device('gpu' if torch.cuda.is_available() else 'cpu')
        self.device = 'GPU' if torch.cuda.is_available() else 'CPU'
        if(self.device == 'GPU'):
            try:
                set_start_method('spawn', force=True)
            except:
                pass
        print('Using device:', self.device)
        
    ################################# <HELPER FUNCTIONS> #################################
    def getTensor(self, var, grad=False):
        return torch.Tensor([var], requires_grad=grad)

    def getTensorFromArr(self, arr, grad=False):
        # return torch.tensor(torch.from_numpy(arr).float(), requires_grad=grad)
        return torch.tensor(arr, dtype=torch.float, requires_grad=grad)

    def init(self, tensor):
        # return torch.nn.init.normal_(tensor, mean=0, std=1)
        return torch.nn.init.uniform_(tensor, a=0.1, b=0.9) # a: lower_bound, b: upper_bound

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    ################################# </HELPER FUNCTIONS> #################################

    ################################# <MAIN FUNCTION> #################################
    def search(self, learning_rate=None, threadID='0', maxsteps=5000, injection_interval=None, mode=None, clamp=None, benchmarking=False, reproduce=False):
        """
            Search function.

            Inputs
            ----------------
            learning_rate: Set the learning rate
            threadID: Give a unique ID, if running in parallel
            maxsteps: Number of steps to run
            injection_interval: Interval to inject randomness
            mode:
            clamp:
            benchmarking: Set to True if benchmarking time (to avoid querying actual cost model)
        """

        net = Net(self.parameters.INPUT_VEC_LEN, self.parameters.OUTPUT_VEC_LEN).cuda() if(self.device=='GPU') \
                else Net(self.parameters.INPUT_VEC_LEN, self.parameters.OUTPUT_VEC_LEN)
        net.share_memory()
        # print(net)

        # Load the pre-trained model
        saved_model = os.path.join(self.saved_model_path, self.parameters.TRAINED_MODEL)
        if(os.path.isfile(saved_model)):
            print("Loading Saved Model")
            if(self.device == 'GPU'):
                net.load_state_dict(torch.load(saved_model))
            else:
                net.load_state_dict(torch.load(saved_model, map_location='cpu'))            
        else:
            sys.exit("Could not find pre-trained model, exit...")

        # Load dataset's minimax
        minmax_path = os.path.join(self.saved_model_path, self.parameters.MEANSTD)
        if(os.path.isfile(minmax_path)):
            minmax = np.load(minmax_path, allow_pickle=True)
        elif(os.path.isfile(os.path.join(self.dataset_path, self.parameters.MEANSTD))):
            minmax = np.load(os.path.join(self.dataset_path, self.parameters.MEANSTD), allow_pickle=True)
        else:
            sys.exit("Mean-Std file not found. Create this: {0}".format(minmax_path))
        # Extract
        inpminmax, otpminmax = minmax

        # **** Make the model not trainable: Key Step ****
        for param in net.parameters():
            param.requires_grad = False

        # Util functions for the search
        searchutils = GsearchUtils(self.costmodel, self.parameters, minmax, threadID)

        ######################## RANDOM VALID MAPPING ########################

        print("Thread {0} Initializing ...".format(threadID, benchmarking, reproduce))
        start_time = time.time() 
        # generate a random valid mapping for the problem and flatten it
        mapping, reference_cost = self.costmodel.getMapCost(threadID=threadID, metric=self.parameters.COST_METRIC)
        flattened_mapping = searchutils.flattenMapping(mapping, normalization=inpminmax)

        # Get the theoretical lower bound
        oracle_cost = self.costmodel.getOracleCost(metric=self.parameters.COST_METRIC)

        # Save the best mapping/cost as a result
        best_mapping = mapping
        best_cost = reference_cost/oracle_cost
        if(reproduce):
            # Result array will only have ground truth (to report final results - In practice, only use surrogate's prediction to drive the search)
            optimize_time_series = [best_cost,]
        if(benchmarking):
            benchmark_time_series = [time.time()-start_time,]
        assert (not((benchmarking==True) and (reproduce==True))), "Can not benchmark and reproduce, it produces wrong benchmarking results."

        # hyperparameters are not trainable.
        bounds = flattened_mapping[:self.parameters.MAPPING_IDX]
        hyperparams = Variable(self.getTensorFromArr(bounds,grad=False))

        # Keep the previous mapping for future reference
        prev_mapping = copy(mapping)

        # Make the input mapping trainable: Key Step
        mapping = Variable(self.getTensorFromArr(flattened_mapping[self.parameters.MAPPING_IDX:],grad=True), requires_grad=True)

        # Perform a forward prop to update the gradients
        if(self.device == 'GPU'):
            out = net(torch.cat((hyperparams.cuda(), mapping.cuda())))
        else:
            out = net(torch.cat((hyperparams, mapping)))

        # We will use sum to get the gradient matrix without any changes (No Loss functions for vanilla GD)
        loss = out.sum()

        # Output (Denormalize)
        cost = searchutils.getCost(out.data.cpu().numpy(), metric=self.parameters.COST_METRIC)

        ######################## INITIALIZE ########################
        iterations = 0
        gradient_steps = 0
        MAX_ATTEMPTS = 100
        # Set parameters
        factor= self.parameters.GRADSEARCH_LR_DECAYFACTOR
        learning_rate = learning_rate if(learning_rate is not None) else self.parameters.GSEARCH_LR
        mode = mode if(mode is not None) else self.parameters.GSEARCH_MODE
        clamp = clamp if(clamp is not None) else self.parameters.GSEARCH_CLAMP
        injection_interval = self.parameters.GSEARCH_RAND_INJECT_ITER if(injection_interval is None) else injection_interval
        alpha = learning_rate
        start_time = time.time()

        ######################## SEARCH PROCEDURE ########################
        print("\n\nThread {0} Starting the run ...".format(threadID))
        while(iterations < maxsteps):
            # print("\n\n------------------------- Iteration %d --------------------------\n\n" % iterations)
            ######################## RANDOM RESTARTS ########################

            # Every few iterations, we create a random schedule
            if(gradient_steps == injection_interval):

                # print("Iteration {0}, best yet: {1}".format(iterations, min(result)))
                # Get a random valid mapping
                # new_mapping, reference_cost = self.costmodel.getMapCost(metric=self.parameters.COST_METRIC, threadID=threadID)
                new_mapping = self.costmodel.getMapping()
                flattened_mapping = searchutils.flattenMapping(new_mapping, normalization=inpminmax)[self.parameters.MAPPING_IDX:]

                # Update to base learning rate
                alpha = learning_rate

                # Keep the previous mapping for future reference
                prev_mapping = copy(new_mapping)

                # Update the mapping data without touching the Graph
                mapping.data = torch.from_numpy(flattened_mapping).float()

                # Perform a forward prop to update the gradients
                if(self.device == 'GPU'):
                    out = net(torch.cat((hyperparams.cuda(), mapping.cuda())))
                else:
                    out = net(torch.cat((hyperparams, mapping)))

                loss = out.sum()
                next_cost = searchutils.getCost(out.data.cpu().numpy(), metric=self.parameters.COST_METRIC)
                # Update the best costs
                best_cost = next_cost if(next_cost < best_cost) else best_cost
                best_mapping = projected_mapping if(next_cost < best_cost) else best_mapping

                # Udpate the result Menu
                gradient_steps = 0
                iterations += 1

                # If you are not running this to reproduce results, these can go away.
                # Wish there was a preprocessor, sigh.
                if(reproduce):
                    # Result array will only have ground truth (to report final results - In practice, only use surrogate's prediction to drive the search)
                    reference_cost = self.costmodel.costFn(new_mapping, metric=self.parameters.COST_METRIC, threadID=threadID)[0]
                    optimize_time_series.append(reference_cost/oracle_cost)
                if(benchmarking):
                    benchmark_time_series.append(min(min(optimize_time_series),time.time()-start_time))
                    start_time = time.time()
                # Prints for debug
                # print("Accepted the new random mapping")
                # print("Updated Mapping: {0}".format(new_mapping))
                # print("New cost: {0}, predicted cost: {1}".format(reference_cost/oracle_cost, cost))
                continue

            ######################## GRADIENT DESCENT ########################
            # Otherwise perform gradient descent

            #### ****** 1. Estimate Gradients based on the current mapping (Backprop). ***** ####
            loss.backward(retain_graph=True)
            grad = mapping.grad.data

            # Retain
            mapping_retain = mapping.data
            steps_to_move = 0
            force_inject = False

            # Apply gradients repeatedly until we get a new mapping
            while(steps_to_move < MAX_ATTEMPTS):

                steps_to_move += 1
                #### ****** 2. Apply gradients to the current mapping. ***** ####
                # Gradient Descent
                mapping_next = mapping.data - (alpha * grad)

                #### ****** 3. Project to valid map space. ***** ####
                projected_mapping, success = searchutils.getProjection(np.concatenate((hyperparams.data.cpu().numpy(), mapping_next.cpu().numpy())), inpminmax)
                # Check if we ended up in invalid space
                if(not success):
                    continue

                # Check if the move resulted in a new mapping
                if(projected_mapping != prev_mapping):
                    # Flatten the mapping and convert to the DNN format (normalize)
                    projected_flattened_mapping = searchutils.flattenMapping(projected_mapping, inpminmax)[self.parameters.MAPPING_IDX:]
                    # Update the mapping data without touching the Graph
                    mapping.data = torch.from_numpy(projected_flattened_mapping).float() if(clamp) else mapping_next
                    break
                else:
                    alpha = alpha/factor

            # If no valid move in the neighborhood was possible
            if(steps_to_move >= MAX_ATTEMPTS):
                # print("Failed to get a valid mapping", threadID)
                gradient_steps = injection_interval
                continue
            else:
                gradient_steps += 1

            #### ****** 4. Forward Prop . ***** ####
            if(self.device == 'GPU'):
                out = net(torch.cat((hyperparams.cuda(), mapping.cuda())))
            else:
                out = net(torch.cat((hyperparams, mapping)))
            # Update the loss (A dummy sum to get the gradients)
            loss = out.sum()

            #### ****** DEBUG. ***** ####
            next_cost = searchutils.getCost(out.data.cpu().numpy(), metric=self.parameters.COST_METRIC)
            # Update the best costs
            best_cost = next_cost if(next_cost < best_cost) else best_cost
            best_mapping = projected_mapping if(next_cost < best_cost) else best_mapping
            # To get the actual cost (to report the results), perform a projection to the valid map space
            # reference_cost = self.costmodel.costFn(projected_mapping, metric=self.parameters.COST_METRIC, threadID=threadID)[0]/oracle_cost if(success) else np.inf
            # Display the results for DEBUG
            # print("Current Mapping: {0}".format(prev_mapping))
            # print("Next Mapping: {0}".format(projected_mapping))
            # # # print("Predicted Cost: {0}, Previous Cost: {1}".format(next_cost, cost))
            # print("Actual Cost: {0}, Best Cost yet: {1}".format(reference_cost, min(result)))
            # print("Learning Rate: {0}".format(alpha))

            #### ****** 5. Decision to change the learning rate. ***** ####

            if(mode == 0):
                # If the next step (done in a single jump) does not decrease the cost (convex assumption in the neighborhood), the learning rate is reduced by the factor.
                if((steps_to_move == 1) and (cost < next_cost)):
                    # print("Not Accepted")
                    mapping.data = mapping_retain
                    alpha = alpha * factor
                    # print("Updated Alpha", alpha)
                    # Otherwise the step is accepted
                else:
                    # print("Accepted")
                    cost = next_cost
                    prev_mapping = projected_mapping
                    best_mapping = projected_mapping
            else:
                cost = next_cost
                prev_mapping = projected_mapping

            # result.append(min(min(result), reference_cost))

            # Clear the gradients
            mapping.grad.zero_()

            # Perform any limitations if necessary (project inputs back to a valid space)

            # For debug
            iterations += 1

            if((iterations%100 == 0) and (not reproduce)):
                print("Steps {0} complete".format(iterations))

            # If you are not running this to reproduce results, these can go away.
            # Wish there was a preprocessor, sigh.
            if(reproduce):
                # Result array will only have ground truth (to report final results - In practice, only use surrogate's prediction to drive the search)
                reference_cost = self.costmodel.costFn(projected_mapping, metric=self.parameters.COST_METRIC, threadID=threadID)[0]/oracle_cost if(success) else np.inf
                optimize_time_series.append(min(min(optimize_time_series), reference_cost))
            if(benchmarking):
                benchmark_time_series.append(time.time()-start_time)
                start_time = time.time()

        # print("Finished run of {0} problem".format(threadID))

        if(reproduce):
            print("\n\nThread {0} Completed ...".format(threadID))
            return optimize_time_series
        elif(benchmarking):
            print("\n\nThread {0} Completed ...".format(threadID))
            return benchmark_time_series
        else:
            # Calculate the reference cost of the best mapping
            best_cost = self.costmodel.costFn(best_mapping, metric=self.parameters.COST_METRIC, threadID=threadID)[0]
            print("\n\nAlgorithm: {4}, Problem: {5}\n\nSteps Ran: {0}\n\nBest Mapping: {1}\n\nPredicted Cost({2}): {3}\n".format(maxsteps, best_mapping, self.parameters.COST_METRIC, best_cost, self.parameters.ALGORITHM, self.costmodel.problem))
            return best_mapping, best_cost

    def run_nn_unpack(self, args):
        return self.run_nn(*args)


################################# </MAIN FUNCTION> #################################

if __name__ == '__main__':


    import argparse
    from mindmappings.costModel.timeloop.model_timeloop import TimeloopModel

    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", default='CNN-layer', required=False)
    parser.add_argument("--metric", default='EDP', required=False)
    parser.add_argument("--path", default='../saved_models_final/', required=False)
    parser.add_argument("--maxsteps", default=100, required=False, type=int)
    parser.add_argument("--problem", help="Enter the problem dimensions", nargs='+', default=[16,256,256,3,3,14,14], type=int)
    args = parser.parse_args()

    params = Parameters(args.algorithm)
    costmodel = TimeloopModel(problem=args.problem, algorithm=args.algorithm, parameters=params)
    tuner = Tuner(costmodel, parameters=params, dataset_path=None, saved_model_path=args.path)
    tuner.search(maxsteps=args.maxsteps)