import pickle
import sys 
sys.path.append("/home/ps/mylin/ARGNP/")
from data.superpixels import SuperPixDatasetDGL 
from torch.utils.data import DataLoader
from data.superpixels import SuperPixDataset
from data.molecules import MoleculeDatasetDGL
import time 
start = time.time()

DATASET_NAME = 'ModelNet10'
dataset = MoleculeDatasetDGL(DATASET_NAME) 

print('Time (sec):',time.time() - start) # 636s=10min


# start = time.time()

# with open('data/superpixels/CIFAR10_train.pkl','wb') as f:
#         pickle.dump([dataset.train,dataset.val,dataset.test],f)
        
# print('Time (sec):',time.time() - start) # 58s