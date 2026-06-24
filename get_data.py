
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import readcfl,writecfl,ifft2c
from scipy.io import loadmat
class MTP_dataset_allechos_sens(Dataset):  
    # return include sense
    def __init__(self):
        super(MTP_dataset_allechos_sens, self).__init__()
        self.ksp = []
        self.ksp_init = []

        data = loadmat('data/slice.mat') 
        self.sens = data['sens']
        self.mask = data['mask_img']
        self.ksp = data['ksp']
        self.nx, self.nc, self.ne, self.ny, self.nz = self.ksp.shape


    def __getitem__(self, idx):
        idx = 1
        ksp = self.ksp[idx]
        sens_2D = self.sens[idx]
        mask = self.mask[idx]

        ksp = torch.tensor(ksp, dtype=torch.complex64)
        # ksp_init = torch.tensor(ksp_init, dtype=torch.complex64)
        sens = torch.tensor(sens_2D, dtype=torch.complex64)
        mask = torch.tensor(mask, dtype=torch.complex64)
        # sample = (ksp, ksp_init,sens)
        sample = (ksp, sens,mask)
        return sample

    def __len__(self):
        return self.nx