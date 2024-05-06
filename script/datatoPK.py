import pickle
import sys 
sys.path.append("/home/ps/mylin/ARGNP2/")
from data.superpixels import SuperPixDatasetDGL 
from torch.utils.data import DataLoader
from data.superpixels import SuperPixDataset
from data.molecules import MoleculeDatasetDGL
from data.SBMs import SBMsDatasetDGL
import time 
start = time.time()

# DATASET_NAME = 'ZINC-full'
# dataset = MoleculeDatasetDGL(DATASET_NAME) 

# print('Time (sec):',time.time() - start) # 636s=10min


# DATASET_NAME = 'ZINC-full'
# dataset = MoleculeDatasetDGL(DATASET_NAME) 

# print('Time (sec):',time.time() - start) # 636s=10min


# start = time.time()

# with open('data/superpixels/CIFAR10_train.pkl','wb') as f:
#         pickle.dump([dataset.train,dataset.val,dataset.test],f)
        
# print('Time (sec):',time.time() - start) # 58s

DATASET_NAME = 'SBM_CLUSTER'
dataset = SBMsDatasetDGL(DATASET_NAME)  #3983s
print('Time (sec):',time.time() - start)