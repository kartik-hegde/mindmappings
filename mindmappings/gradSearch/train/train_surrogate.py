from numpy.lib.npyio import save
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler  import StepLR
from torch.multiprocessing import set_start_method
import sys, os
import numpy as np

from mindmappings.parameters import Parameters

TEST = True
TRAIN = True
VERIFY = False

class MyDataset(Dataset):
    def __init__(self, data, split=0.8, train=True, transform=None):
        split_loc = int(len(data) * split)
        num_samples, in_vec_len, out_vec_len = data.shape[0], data[0,0].shape[0], data[0,1].shape[0]

        # Create arrays: Unfortunately, this is an expensive operation. (np.stack is even slower)
        # TODO: Take care of this in the dataset creation time.
        mapping = np.empty([num_samples, in_vec_len])
        target = np.empty([num_samples, out_vec_len])
        for sample in range(num_samples):
            mapping[sample] = data[sample,0]
            target[sample] = data[sample, 1]

        if(train):
            self.mapping = torch.from_numpy(mapping[:split_loc]).float()
            self.target  = torch.from_numpy(target[:split_loc]).float()
        else:
            # test
            self.mapping = torch.from_numpy(mapping[split_loc:]).float()
            self.target = torch.from_numpy(target[split_loc:]).float()

        self.split = split
        self.train = train
        self.transform = transform

    def __getitem__(self, index):
        x = self.mapping[index]
        y = self.target[index]
        if self.transform:
            x = self.transform(x)
        return x,y

    def __len__(self):
        return len(self.mapping)

class Net(nn.Module):
    def __init__(self, input_vec_len, output_vec_len):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_vec_len, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 1024)
        self.fc4 = nn.Linear(1024, 2048)
        self.fc5 = nn.Linear(2048, 2048)
        self.fc6 = nn.Linear(2048, 1024)
        self.fc7 = nn.Linear(1024, 256)
        self.fc8 = nn.Linear(256, 64)
        self.fc9 = nn.Linear(64, output_vec_len)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = self.fc9(x)
        return x

class TrainSurrogate:

    def __init__(self, parameters=Parameters(), dataset_path=None, saved_model_path=None) -> None:
        self.parameters = parameters
        self.dataset_path = parameters.DATASET_PATH if(dataset_path==None) else dataset_path
        self.saved_model_path = parameters.MODEL_SAVE_PATH if(saved_model_path==None) else saved_model_path
        # setting device on GPU if available, else CPU
        # torch.device('gpu' if torch.cuda.is_available() else 'cpu')
        self.device = 'GPU' if torch.cuda.is_available() else 'CPU'
        if(self.device == 'GPU'):
            set_start_method('spawn')
        print('Using device:', self.device)

    def getLoader(self, data, batch_size=16, split=0.1, train=True):
        """
            Returns the loader.
        """
        dataset = MyDataset(data, split, train)

        loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4
                )

        return loader


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

    def verify(self,net_out, target, otpminmax):
        """Compare the target and the predicted output"""
        print("Predicted: {0}, Dataset: {1}".format(self.denormalize(net_out.data.cpu().numpy()[0], otpminmax), self.denormalize(target.data.cpu().numpy()[0], otpminmax)))
        input("Enter to continue")

    def trainer(self, batch_size=None, learning_rate=None, epochs=None, log_interval=100):
        """Runs the training"""

        epochs = epochs if(epochs!=None) else self.parameters.SURROGATE_TRAIN_EPOCHS
        batch_size = batch_size if(batch_size!= None) else self.parameters.SURROGATE_TRAIN_BATCHSIZE
        learning_rate = learning_rate if(learning_rate!=None) else self.parameters.SURROGATE_TRAIN_LR

        # Instantiate the network
        if(self.device == 'GPU'):
            net = Net(self.parameters.INPUT_VEC_LEN, self.parameters.OUTPUT_VEC_LEN).cuda()
        else:
            net = Net(self.parameters.INPUT_VEC_LEN, self.parameters.OUTPUT_VEC_LEN)

        print(net)

        # Load the pre-trained model
        saved_model = os.path.join(self.saved_model_path, self.parameters.TRAINED_MODEL)
        if(os.path.isfile(saved_model)):
            print("Loading Saved Model")
            if(self.device == 'GPU'):
                net.load_state_dict(torch.load(saved_model))
            else:
                net.load_state_dict(torch.load(saved_model, map_location='cpu'))            
        else:
            print("WARNING: Could not find pre-trained model, initializing ...")

        # Load dataset's minimax
        minmax_path = os.path.join(self.saved_model_path, self.parameters.MEANSTD)
        if(os.path.isfile(minmax_path)):
            minmax = np.load(minmax_path, allow_pickle=True)
        elif(os.path.isfile(os.path.join(self.dataset_path, self.parameters.MEANSTD))):
            minmax = np.load(os.path.join(self.dataset_path, self.parameters.MEANSTD), allow_pickle=True)
        else:
            sys.exit("Mean-Std file not found. Create this: {0}".format(minmax_path))

        # Load normalization stats
        inpminmax, otpminmax= minmax

        # create an optimizer
        if(self.parameters.OPTIMIZER=='SGD'):
            optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
        elif(self.parameters.OPTIMIZER=='Adam'):
            optimizer = optim.Adam(net.parameters(), lr=learning_rate)

        # create a loss function
        # criterion = nn.MSELoss()
        # criterion = nn.L1Loss()
        criterion = nn.SmoothL1Loss()
        scheduler = StepLR(optimizer, step_size=20, gamma=0.1)


        # ------- run the main training loop ------
        print("\n\n \t\t TRAINING START \n\n")
        for epoch in range(epochs):
            # Go over every saved file (filter .npy files only)
            for p in os.listdir(self.dataset_path):
                file_path = os.path.join(self.dataset_path, p)
                data_loaded = np.load(file_path, allow_pickle=True)
                print("Picking up " + file_path)
                if(file_path.rstrip().split(".")[-1] == 'npy'):

                    ### Train
                    if(TRAIN):
                        print("\n\n ---- TRAIN -----")
                        loader = self.getLoader(data_loaded, batch_size=batch_size, split=0.90, train=True)
                        for batch_idx, (mapping, target) in enumerate(loader):
                            if(self.device == 'GPU'):
                                mapping, target = Variable(mapping).cuda(),Variable(target).cuda()
                            else:
                                mapping, target = Variable(mapping),Variable(target)
                            net_out = net(mapping)
                            # If you are running this the first time, set this to true to see if things are alright.
                            if(VERIFY):
                                self.verify(net_out, target, otpminmax)
                            # Update the parameters
                            optimizer.zero_grad()
                            loss = criterion(net_out, target)
                            loss.backward()
                            optimizer.step()
                            # Print out the loss
                            if batch_idx % log_interval == 0:
                                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                    epoch, batch_idx * len(mapping), len(loader.dataset),
                                        100. * batch_idx / len(loader), loss.item()))

                    if(TEST):
                        print("\n\n ---- TEST -----")
                        loader = self.getLoader(data_loaded, batch_size=batch_size, split=0.90, train=False)
                        for batch_idx, (mapping, target) in enumerate(loader):
                            if(self.device == 'GPU'):
                                mapping, target = Variable(mapping).cuda(),Variable(target).cuda()
                            else:
                                mapping, target = Variable(mapping),Variable(target)
                            net_out = net(mapping)
                            loss = criterion(net_out, target)
                            if batch_idx % log_interval == 0:
                                print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                    epoch, batch_idx * len(mapping), len(loader.dataset),
                                        100. * batch_idx / len(loader), loss.item()))
                # Save the model                                        
                torch.save(net.state_dict(), saved_model)
            scheduler.step()

if __name__ == "__main__":
    surrogate = TrainSurrogate(parameters=Parameters(), dataset_path=sys.argv[1])
    surrogate.trainer(epochs=1)
