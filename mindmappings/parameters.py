#####################################################################
# This file sets all parameters for the entire run.
#####################################################################
import sys, os
import random
import pathlib

class Parameters:

    def __init__(self, algorithm='CNN-layer', metric='EDP'):
        # Base of the directory
        self.BASE = str(pathlib.Path(__file__).parent.absolute())

        # Change this flag for Debug prints
        self.DEBUG = False
        # self.DEBUG = True

        # Set a scratch path to write temporary files.
        # Tip: Use a fast memory to speed-up runs.
        self.SCRATCH = '/scratch/kvhegde2/' # Create this path if does not exist

        # Sets the cost metric
        self.COST_METRIC = metric #'energy'/'perf'/'EDP'

        #####################################################################
        #### ARCHITECTURE CONTROLS ####
        #####################################################################

        ### Architecture description
        self.ARCHITECTURE = {'bufferSizes': [1024*512, 1024*64], 'buffer_access_energy':[2.2, 1.12], 'bufferWidth':[8,2], \
                        'numBanks':[16,16],'numHierarchy':3, 'parallelHierarchy':2, 'numPEs':256, 'mac_energy':1.0}

        #####################################################################
        #### ALGORITHM CONTROLS ####
        #####################################################################

        ### Choice of Algorithm
        self.ALGORITHM = algorithm #'MTTKRP'/'CNN-layer'/'Example'

        # Input vector length setup
        if(self.ALGORITHM == 'CNN-layer'):
            self.PROBLEM_SHAPE = 7
            self.OUTPUT_VEC_LEN = 12
            self.INPUT_VEC_LEN = 62
            self.HYPERPARAM_IDX = 0
            self.MAPPING_IDX = 7 # This is where the mapping starts from
            self.TILING_IDX = 7
            self.LOOP_ORDER_IDX = 35
            self.PARTITION_IDX = 56
        elif(self.ALGORITHM == 'MTTKRP'):
            self.PROBLEM_SHAPE = 4
            self.OUTPUT_VEC_LEN = 15
            self.INPUT_VEC_LEN = 40
            self.HYPERPARAM_IDX = 0
            self.MAPPING_IDX = 4 # This is where the mapping starts from
            self.TILING_IDX = 4
            self.LOOP_ORDER_IDX = 20
            self.PARTITION_IDX = 32
        elif(self.ALGORITHM == 'Example'):
            self.PROBLEM_SHAPE = 0
            self.OUTPUT_VEC_LEN = 1
            self.INPUT_VEC_LEN = 5
            self.HYPERPARAM_IDX = 0
            self.MAPPING_IDX = 0 # This is where the mapping starts from
        else:
            sys.exit("Algorithm not supported")

        # Average Numbers of runs
        self.AVG_ITERS = 50

        # Maximum Iterations
        self.MAXSTEPS = 5000 

        #####################################################################
        #### TIMELOOP CONTROLS ####
        #####################################################################

        # Path to executable: Install timeloop and point to the timeloop directory.
        self.TIMELOOP_PATH = '/home/kvhegde2/softwares/timeloop/'

        # Executable (timeloop-model)
        self.COSTMODEL_EXECUTABLE =  os.path.join(self.TIMELOOP_PATH ,'build/timeloop-model')

        # Use this as a base to edit the timeloop configs
        if(self.ALGORITHM == 'CNN-layer'):
            self.SAMPLE_CFG_FILE =  self.BASE + '/costModel/timeloop/cnn_base.yaml'
        elif(self.ALGORITHM == 'MTTKRP'):
            self.SAMPLE_CFG_FILE =  self.BASE + '/costModel/timeloop/mttkrp_base.yaml'

        # Name of the output file to parse
        self.OUTPUT_FILE = '.stats.txt'
        # Temporary directory where each thread writes/executes
        self.OUTPUT_DIR_BASE = self.SCRATCH + 'timeloop/outputs_' + str(self.ALGORITHM) + '/grun_'
        # Whether to check if Tile fits or not
        self.CHECK_TILE_VALIDITY = False

        #####################################################################
        #### Gradient Search controls ####
        #####################################################################
        self.GRADSEARCH_PATH = self.BASE + '/gradSearch/'


        # ------ Data generation ----------
        self.DATASET_UNPROCESSED_PATH = self.SCRATCH + '/timeloop/dataset_unprocessed_' + self.ALGORITHM + '/'
        self.DATASET_NUMFILES = 10 # 100 # Total number files.
        self.DATASET_NUMSAMPLES_FILE = 10# 100000 # This determines the size of each file. Dataset size = NUMFILES * NUMSAMPLES_FILE
        self.DATASET_MAPPINGS_PER_PROBLEM = 1 # 500 # Set this to control the number of mappings per problem. (higher if each problem is complex)

        # ------- Data Post process --------

        # Note: Best training results when the normalization is done is a single thread: because mean/variance is more accurate.
        # However, if you are running out of memory, you can split it. 
        # Control the above using NUM_FILES_PER_THREAD. If you have 100 files and you set this to 100, only one thread will be launched.
        # Similarly, set NUM_THREADS to more if you want parallelism. Suggested to use 1 for better training.
        # NUM_OUTFILES controls the total number of output files. This is to be set based on your training data size.
        self.DATASET_PATH = self.SCRATCH + '/timeloop/dataset_' + self.ALGORITHM + '/'
        self.DATASET_NUM_FILES_PER_THREAD = 10 # 100
        self.DATASET_NUM_THREADS = 1
        self.DATASET_NUM_OUTFILES = 50

        ## Train
        self.SURROGATE_TRAIN_EPOCHS = 100
        self.SURROGATE_TRAIN_BATCHSIZE = 256
        self.SURROGATE_TRAIN_LR = 1e-2

        # Search Parameters

        # Update these if you add more algorithms
        clamps = {'CNN-layer':0, 'MTTKRP': 1}
        modes = {'CNN-layer':1, 'MTTKRP': 0}
        injection_intervals = {'CNN-layer':10, 'MTTKRP': 25}

        self.GSEARCH_OUTPATH = self.GRADSEARCH_PATH + 'results/'
        self.GSEARCH_AVG_ITERS = self.AVG_ITERS
        self.GSEARCH_MAXSTEPS= self.MAXSTEPS
        self.GSEARCH_NUMCORES= None
        self.GSEARCH_TEMPERATURE = 50
        self.GSEARCH_TEMP_UPDATE_ITER = 50
        self.GSEARCH_TEMP_ANNEAL_FACTOR = 0.75
        self.GSEARCH_RAND_INJECT_ITER = injection_intervals[self.ALGORITHM]
        self.GSEARCH_LR = 1
        self.GSEARCH_CLAMP = clamps[self.ALGORITHM] 
        self.GSEARCH_MODE = modes[self.ALGORITHM]
        self.GRADSEARCH_LR_DECAYSTEP = 500
        self.GRADSEARCH_LR_DECAYFACTOR = 0.5
        self.OPTIMIZER = 'SGD'
        self.MODEL_SAVE_PATH = self.GRADSEARCH_PATH + '/saved_models_final/'
        self.TRAINED_MODEL = 'model_'+ self.ALGORITHM + '.save'
        self.MEANSTD = 'mean_'+ self.ALGORITHM + '.pickle'

        #####################################################################
        #### Parameters for runs ####
        #####################################################################

        #### Problem shapes####
        if(self.ALGORITHM == 'CNN-layer'):
            self.problems_actual = [ #N C K R S P Q
                        [16, 64,128,3,3,112,112], # VGG_conv2_1
                        [32, 64,192,3,3,56,56], # Inception_conv2_1x1
                        [8, 96,256,5,5,27,27],  # AlexNet_conv2
                        [8, 384,384,3,3,13,13], # AlexNet_conv2
                        [16,128,128,3,3,28,28], # ResNet_Conv3_0_3x3
                        [16,256,256,3,3,14,14]] # ResNet_Conv3_0_3x3

            self.problem_test = [#N C K R S P Q
                            [16,256,256,3,3,14,14]] # ResNet_Conv3_0_3x3
                            # [8, 96,256,5,5,27,27]] # AlexNet_conv2

            self.problem_names = ['VGG_conv2_1', 'Inception_conv2_1x1', 
                                    'AlexNet_conv2', 'AlexNet_conv2',
                                    'ResNet_Conv3_0_3x3', 'ResNet_Conv4_0_3x3']

            # Typical hyperparameters
            self.PROBLEM_RANGES = [
                            (1,2,4,8,16), #N 

                            # (3,32, 64,128,196,256,392,512), #C 
                            [3,]+list(range(32,512,32)), #C

                            range(32,512,32), #K
                            # (32,64,128,196,256,392,512), #K

                            (1,3,5,7), #R/S

                            range(7,224), # P/Q
                            # (7,14,28,56,112,224), #P/Q
                            ]



        elif(self.ALGORITHM == 'MTTKRP'):

            self.problems_actual = [ #I J K L
                                [128,1024,4096,2048],
                                [2048,4096,1024,128],
                                ]

            # problem_test = [[1024, 1024, 1024, 1024]]
            self.problem_test = [ #I J K L
                                [256,256,256,256],]

            # Typical hyperparameters
            self.PROBLEM_RANGES = [
                            range(128,4096,128), #I
                            range(128,4096,128), #J
                            range(128,4096,128), #K
                            range(128,4096,128), #L
                            ]

            self.problem_names = ['MTTKRP_0', 'MTTKRP_1']

        self.random_problem_gen = self.getRandomProblem
        self.problems = self.problem_test if self.DEBUG else self.problems_actual
        #####################################################################
        #### Caches for faster simulation
        #####################################################################
        self.REFCACHE_PATH = self.BASE + '/costModel/timeloop/refcache.npy'

    def getProblems(self):
        if(self.ALGORITHM == 'CNN-layer'):
            return [ #N C K R S P Q
                        [16, 64,128,3,3,112,112], # VGG_conv2_1
                        [32, 64,192,3,3,56,56], # Inception_conv2_1x1
                        [8, 96,256,5,5,27,27],  # AlexNet_conv2
                        [8, 384,384,3,3,13,13], # AlexNet_conv2
                        [16,128,128,3,3,28,28], # ResNet_Conv3_0_3x3
                        [16,256,256,3,3,14,14]] # ResNet_Conv3_0_3x3
        else:
            return  [ #I J K L
                                [128,1024,4096,2048],
                                [2048,4096,1024,128],
                                ]

    def getRandomProblem(self):
        if(self.ALGORITHM == 'CNN-layer'):
            N,C,K,R,P = [random.choice(r) for r in self.PROBLEM_RANGES]
            return [N,C,K,R,R,P,P]
        else:
            return [random.choice(r) for r in self.PROBLEM_RANGES]


# Default parameters (import parameters directly)
# parameters = Parameters()
