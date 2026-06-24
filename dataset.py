import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import readcfl, writecfl, ifft1c, fft1c

class MTP_dataset_allechos_sens(Dataset):  
    # return include sense
    def __init__(self, root: str, flag: str, vol: str, echo: int, acc: str):
        super(MTP_dataset_allechos_sens, self).__init__()
        self.ksp = []
        self.ksp_init = []

        for i in range(echo):
            echo_name = 'Echo_' + str(i + 1 )
            data_root = os.path.join(root, flag, vol, echo_name)
            assert os.path.exists(data_root), f"path '{data_root}' does not exist. "
            # ksp_path = f'{echo_name}_ksp'
            ksp_path ='ksp'
            ksp = readcfl(os.path.join(data_root,ksp_path))
            # ksp_und_path = f'{echo_name}_ksp_und'
            # ksp_init = readcfl(os.path.join(data_root, ksp_und_path))

            ksp = np.transpose(ksp, (0, 3, 1, 2))  # [nb ny nz nc] -> [nb nc ny nz]
            # ksp_init = np.transpose(ksp_init, (0, 3, 1, 2))

            self.ksp.append(ksp)
            # self.ksp_init.append(ksp_init)

        self.ksp = np.stack(self.ksp, axis=2)  # [nx, nc, ne, ny, nz]
        # self.ksp_init = np.stack(self.ksp_init, axis=2)
        # self.ksp = self.ksp[35:315,:,:,:,:]
        # self.ksp_init = self.ksp_init[35:315,:,:,:,:]
        
        sens_root = os.path.join(root, flag, vol)
        assert os.path.exists(sens_root), f"path '{sens_root}' does not exist. "
        self.sens = readcfl(os.path.join(sens_root, 'csm'))# [nx,ny,nz,nc] 
        self.sens = np.transpose(self.sens,(0,3,1,2))# [nx,nc,ny,nz] 
        self.mask = readcfl(os.path.join(sens_root, 'mask'))
        # self.sens = self.sens[35:315,:,:,:]
        self.nx, self.nc, self.ne, self.ny, self.nz = self.ksp.shape


    def __getitem__(self, idx):
        # idx = 100
        ksp = self.ksp[idx]
        # ksp_init = self.ksp_init[idx]
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
