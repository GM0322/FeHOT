import torch
import  numpy as np
import os
import torch.utils.data as data

class ProjAndImageDataset(data.Dataset):
    def __init__(self,root,nViews,nBins,nSize):
        self.root = root
        self.nViews = nViews
        self.nBins = nBins
        self.nSize = nSize
        self.files = os.listdir(self.root)

    def __getitem__(self, item):
        input = torch.from_numpy(np.fromfile(self.root+'/'+self.files[item],dtype=np.float32).reshape(1,self.nViews,self.nBins))
        # label = torch.from_numpy(np.fromfile(self.root+'/../label/'+self.files[item],dtype=np.float32).reshape(1,self.nSize,self.nSize))
        label = np.fromfile(self.root+'/../label/'+self.files[item],dtype=np.float32).reshape(1,self.nSize,self.nSize)
        label = label[:,::-1,:].copy()
        label = torch.from_numpy(label)
        return input,label.permute(0,2,1), self.files[item]

    def __len__(self):
        return len(self.files)

def ProjAndImageDataloader(root,nViews,nBins,nSize,batchSize,shuffle):
    dataset = ProjAndImageDataset(root,nViews,nBins,nSize)
    return data.DataLoader(dataset,batch_size=batchSize,shuffle=shuffle,num_workers=4)
