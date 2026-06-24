import os
import argparse
import time
import datetime
import torch
import random
import numpy as np
from scipy.io import loadmat
# from torch.utils.tensorboard import SummaryWriter
# import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from MCMCCDic_v4_hq_12echo_prompt_hyper_RS_2 import MCMCCDic_v4_hq as mccd
from dataset import MTP_dataset_allechos_sens
from utils import ifft2c, c2r, sos,writecfl,fft2c
from ssim import SSIM
from mdi_function import R_S_2
from scipy.io import savemat
def get_mask(mask_name):
    data = loadmat('pois_10x.mat')
    mask = data['mask']
    mask = torch.tensor(mask, dtype=torch.complex64)

    return mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', metavar='int', nargs=1, default=['1000'], help='number of epochs')
    parser.add_argument('--batch_size', metavar='int', nargs=1, default=['1'], help='batch size')
    parser.add_argument('--learning_rate', metavar='float', nargs=1, default=['0.0001'], help='initial learning rate')
    parser.add_argument('--niter', metavar='int', nargs=1, default=['10'], help='number of network iterations')
    parser.add_argument("--nEcho", metavar='int', nargs=1, default=['12'], help='number of echos')
    parser.add_argument('--net', metavar='str', nargs=1, default=['hyper_fuse'], help='network')
    parser.add_argument('--data', metavar='str', nargs=1, default=['MTP_dateset'], help='dataset name')
    parser.add_argument('--acc', metavar='str', nargs=1, default=['12x'], help='accelerate rate')
    parser.add_argument('--mask', metavar='str', nargs=1, default=['pois_10x'], help='under-sampling pattern')
    parser.add_argument('--manualSeed', type=int, help='manual seed')

    args = parser.parse_args()

    # GPU id
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.manualSeed is None:
        # args.manualSeed = random.randint(1, 10000)
        args.manualSeed = 3407
    print("Random seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # get arguments
    dataset_name = args.data[0]
    model_name = args.net[0]

    epoches = int(args.num_epoch[0])
    batch_size = int(args.batch_size[0])
    learning_rate = float(args.learning_rate[0])
    niter = int(args.niter[0])
    nEcho = int(args.nEcho[0])
    acc = args.acc[0]
    mask_name = args.mask[0]

    print('-----------load data----------')  
    for vol in  ['vol1']:
        test_datasets = []
        test_name = [vol]
        for i in range(len(test_name)):
            vol = test_name[i]
            test_dataset = MTP_dataset_allechos_sens('mtp_dataset', 'test', vol, nEcho, acc)
            test_datasets.append(test_dataset)
        testset = ConcatDataset(test_datasets)
        testnum = int(testset.__len__())
        print('The length of test dataset is:', testnum)
        test_loader = DataLoader(testset, batch_size=1, shuffle=False,num_workers=4)
    
        # creat network

        model = mccd().to(device)
        if torch.cuda.device_count() >1:
            model = torch.nn.DataParallel(model)
            
        model.load_state_dict(
            torch.load('net_params_84.pkl'),strict=True)
        
        param_num = np.sum([np.prod(v.nelement()) for v in model.parameters()])
        print('number of params:', param_num)
        print("-----------------------training---------------------------")

        params = model.parameters()
        mask = get_mask(mask_name=mask_name)
        mask = mask.to(device)


        test_loss_epoch = []
        label_3D = np.zeros((12,336,288,84),dtype = complex)
        recon_3D = np.zeros((12,336,288,84),dtype = complex)
        
        with torch.no_grad():
            test_loss_epoch = []
            for step, sample in enumerate(test_loader):
                # if step > 27 and step< 310:
                if True:
                    t0 = time.time()
                    loss_iter = 0

                    ksp, sens,mask_img = sample
                    ksp, sens,mask_img = ksp.to(device),  sens.to(device),mask_img.to(device)

                    nb, nc, ne, ny, nz = ksp.size()
                    cg = ksp * mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)

                    label = ifft2c(cg)  # [nb nc ne nx ny]
                    image = torch.stack([torch.sum(label[:, :, ne, :, :] * torch.conj(sens), dim=1) for ne in range(ne)], dim=1)
                    image = sos(label)  # [nb ne nx ny]
                    scale_factor, _ = torch.max(abs(image).view(1, -1), dim=1)
                    
                    scale_factor, _ = torch.max(image.view(1, -1), dim=1)
                    k0 = torch.div(ksp, scale_factor.view(1, 1, 1, 1, 1))
                    ku = torch.div(cg, scale_factor.view(1, 1, 1, 1, 1))

                    label_coils = ifft2c(k0)
                    label = torch.stack([torch.sum(label_coils[:, :, ne, :, :] * torch.conj(sens), dim=1) for ne in range(ne)], dim=1)
            
                    k0_u = k0 * mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    img_coils = ifft2c(k0_u)
                    img = torch.stack([torch.sum(img_coils[:, :, ne, :, :] * torch.conj(sens), dim=1) for ne in range(ne)], dim=1)
                    
                    
                   
                    masks = mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
                    masks = torch.tile(masks,(nb,1,1,1,1)).real
                    # if step == 50:
                    #     pass
                    recon,S_recon,R1_recon,R2_recon= model(k0_u, sens,  masks, img,mask_img)
                    
                    loss = F.l1_loss(recon, label)
                    print(step,':',loss.item())
                    # savemat('recon.mat',{'recon':recon.detach().cpu().numpy()})
                    # loss3 = F.l1_loss(R1_recon,R1_label) + F.l1_loss(S_recon,S_label)+ F.l1_loss(R2_recon,R2_label)
                    recon_ksp = fft2c(recon.squeeze(1).repeat(1, nc,1, 1, 1) * sens.unsqueeze(2).repeat(1,1,ne,1,1))
                    # savemat('recon.mat',{'recon_ksp':recon_ksp.detach().cpu().numpy()})
                    recon_coils = ifft2c(recon_ksp* scale_factor.view(1, 1, 1, 1, 1))
                    recon = torch.stack([torch.sum(recon_coils[:, :, ne, :, :] * torch.conj(sens), dim=1) for ne in range(ne)], dim=1)
                    
                    label_ksp = fft2c(label.squeeze(1).repeat(1, nc,1, 1, 1) * sens.unsqueeze(2).repeat(1,1,ne,1,1))
                    label_coils = ifft2c(label_ksp* scale_factor.view(1, 1, 1, 1, 1))
                    label = torch.stack([torch.sum(label_coils[:, :, ne, :, :] * torch.conj(sens), dim=1) for ne in range(ne)], dim=1)
                    
                    label_3D[:,step,:,:] = label.squeeze(0).detach().cpu().numpy()
                    recon_3D[:,step,:,:] = recon.squeeze(0).detach().cpu().numpy()
                    test_loss_epoch.append(loss.item())
        test_Loss = np.mean(test_loss_epoch)
        print('test_loss',test_Loss)
        path = 'recon.mat'
        savemat(path,{'recon':recon_3D,'gt':label_3D } )