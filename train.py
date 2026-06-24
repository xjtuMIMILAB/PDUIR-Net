import os
import argparse
import time
import datetime
import torch
import random
import numpy as np
from scipy.io import loadmat
# import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from MCMCCDic_v4_hq_12echo_prompt_hyper_RS_5 import MCMCCDic_v4_hq as mccd
from dataset import MTP_dataset_allechos_sens
from utils import ifft2c, c2r, sos,writecfl
from ssim import SSIM
from mdi_function import R_S_2,calculate_S_only
from scipy.io import savemat
def get_mask(mask_name):
    data = loadmat('data\\pois_10x.mat')
    mask = data['mask']
    mask = torch.tensor(mask, dtype=torch.complex64)

    return mask


class MyLoss(torch.nn.Module):
    # 不要忘记继承Module
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, output, target):
        loss_nrmse = torch.norm((output - target), 'fro') / torch.norm(target, 'fro')
        return loss_nrmse
    
    
def is_zero(maps):
    nb,nc,h,w = maps.shape
    for i in range(nb):
        map = maps[i,:,:,:]
        if torch.equal(map, torch.zeros_like(map)):
            return True
    return False


def norm_r1(R1_map,max_magnitude=3.0):
    eps = 1e-8
    magnitude = torch.abs(R1_map)
    scale = torch.where(magnitude > max_magnitude, 
                       max_magnitude / (magnitude + eps), 
                       torch.ones_like(magnitude))
    return R1_map * scale
    
if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    
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
    parser.add_argument('--manualSeed',metavar='str', nargs=1, default=42, help='manual seed')

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

    # Datasets
    print('-----------load data----------')
    train_datasets = []
    # vol_name = ['vol1']
    # vol_name = ['vol1', 'vol2', 'vol3', 'vol4', 'vol5', 'vol6', 'vol7', 'vol8', 'vol9']
    vol_name = [f'vol{i+1}'  for i in range(1)]
    for j in range(len(vol_name)):
        vol = vol_name[j]
        print(vol)
        train_dataset = MTP_dataset_allechos_sens('mtp_dataset', 'train', vol, nEcho, acc)
        train_datasets.append(train_dataset)

    trainset = ConcatDataset(train_datasets)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,drop_last=True,num_workers=0)
    num = int(trainset.__len__())
    print('The length of train dataset is:', num)
    

    # log dir
    base_dir = './result/checkpoints_small'
    model_save_path = os.path.join(base_dir, model_name)
    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)

    # ckpt_path = os.path.join(model_save_path, 'result_echo12_dc_5_ft_lossrecon')
    ckpt_path = os.path.join(model_save_path, 'result_echo12_dc_5_ft_lossrecon')
    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)
    in_channel = 2 * nEcho
    # creat network
    # DSNMs = Deep_snms_echos(in_channel=in_channel, niter=niter).to(device)
    model = mccd().to(device)
    if torch.cuda.device_count() >1:
        model = torch.nn.DataParallel(model)
    # model.load_state_dict(
    #     torch.load(
    #         'net_params_42.pkl'
    #     ),strict=False)
    param_num = np.sum([np.prod(v.nelement()) for v in model.parameters()])
    print('number of params:', param_num/1024)

    # Initialize network parameters
    # init_net(DSNMs, init_type='kaiming')
    print("-----------------------training---------------------------")

    params = model.parameters()
    optim = torch.optim.Adam(params, lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.2)
    # scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.9)
    # scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2])
    
    mask = get_mask(mask_name=mask_name)
    mask = mask.to(device)

    # train
    print('----------- Train network----------')
    each_epoch_steps = int(num / batch_size)
    total_loss = []
    mdi_loss = []
    loss = 0
    SSIM = SSIM(data_range=1, size_average=True, channel=nEcho)
    results_file = "./log/results_pois_10x_echo12_5_lossrecon_{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    t0_start = time.time()
    criterion = MyLoss().cuda()
    
    start_epoch = 1
    for epoch in range(start_epoch,epoches):
        model.train()
        train_loss_epoch = []
        mdi_loss_epoch = []

          
        for step, sample in enumerate(train_loader):
            t0 = time.time()
            loss_iter = 0

            ksp, sens,mask_img = sample
            ksp, sens,mask_img = ksp.to(device),  sens.to(device),mask_img.to(device)

            nb, nc, ne, ny, nz = ksp.size()
            cg = ksp * mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            # savemat('cg.mat',{'cg':cg.detach().cpu().numpy()})
            # normalization
            label = ifft2c(cg)  # [nb nc ne nx ny]
            image = torch.stack([torch.sum(label[:, :, ne, :, :] * torch.conj(sens), dim=1) for ne in range(ne)], dim=1)
            image = sos(label)  # [nb ne nx ny]
            scale_factor, _ = torch.max(abs(image).view(batch_size, -1), dim=1)
            k0 = torch.div(ksp, scale_factor.view(batch_size, 1, 1, 1, 1))
            ku = torch.div(cg, scale_factor.view(batch_size, 1, 1, 1, 1))

            label_coils = ifft2c(k0)
            label = torch.stack([torch.sum(label_coils[:, :, ne, :, :] * torch.conj(sens), dim=1) for ne in range(ne)], dim=1)
            
            
            k0_u = k0 * mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            img_coils = ifft2c(k0_u)
            img = torch.stack([torch.sum(img_coils[:, :, ne, :, :] * torch.conj(sens), dim=1) for ne in range(ne)], dim=1)
            
            
            # x0 = ku * (1 - mask).unsqueeze(0).unsqueeze(0).unsqueeze(0) 
            masks = mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            masks = torch.tile(masks,(nb,1,1,1,1)).real
            # savemat('mask.mat',{'mask':masks.detach().cpu().numpy()})
                # recon,S_recon,R1_recon,R2_recon= model(k0_u, sens,  masks, img)  # [nb nc ne nx ny]
            # with torch.no_grad():
            R1_label,R2_label,S_label = R_S_2(label.reshape(nb,ne,1,ny,nz))
            
            R1_label = R1_label * mask_img
            R2_label = R2_label* mask_img
            S_label = S_label * mask_img
            # R1_label = torch.stack((R1_label.real,R1_label.imag),dim=1)
            # R2_label = torch.stack((R2_label.real,R2_label.imag),dim=1)
            # S_label = torch.stack((S_label.real,S_label.imag),dim=1)
            # if is_zero(S_label) or is_zero(R1_label) or is_zero(R2_label):
            #     # savemat('data_nan.mat',{'S_map':S_map_2.detach().cpu().numpy(),'R1_map':R1_map.detach().cpu().numpy(),'R2_map':R2_map.detach().cpu().numpy()})
            #     print(step)
            #     continue
            recon,S_map,R1_map,R2_map= model(k0_u, sens,  masks, img,mask_img)
            # recon= model(k0_u, sens,  masks, img,mask_img)
            
            
            R1_map_2,R2_map_2,S_map_2 = R_S_2(recon.reshape(nb,ne,1,ny,nz))
            
            R1_map_2 = R1_map_2 * mask_img
            R2_map_2 = R2_map_2* mask_img
            S_map_2 = S_map_2 * mask_img   
            # R1_map_2 =  norm_r1(R1_map_2) 
            R1_map_2 = torch.stack((R1_map_2.real,R1_map_2.imag),dim=1)
            R2_map_2 = torch.stack((R2_map_2.real,R2_map_2.imag),dim=1)
            S_map_2 = torch.stack((S_map_2.real,S_map_2.imag),dim=1)
            
            
            optim.zero_grad()


            # R1_map = torch.stack((R1_map.real,R1_map.imag),dim=1)
            # R2_map = torch.stack((R2_map.real,R2_map.imag),dim=1)
            # S_map = torch.stack((S_map.real,S_map.imag),dim=1)
            
            
            loss_mae = F.l1_loss(recon, label)
            loss_ssim = SSIM(abs(recon), abs(label))

            loss1 =  loss_mae   + 0.5*(1-loss_ssim)
            # loss3 = criterion(R1_map,R1_label)  + criterion(S_map,S_label)+ criterion(R2_map,R2_label)
            loss3 = F.l1_loss(R1_map,R1_label)  + F.l1_loss(S_map,S_label)+ F.l1_loss(R2_map,R2_label)
            loss = loss1 + 1e-6 * loss3
            # print(step,':',loss.item())
            if torch.isnan(loss).any():
                savemat('data_nan_2.mat',{'S_map':S_map.detach().cpu().numpy(),'R1_map':R1_map.detach().cpu().numpy(),'R2_map':R2_map.detach().cpu().numpy()})
                print(step)
                continue

            train_loss_epoch.append(loss.item())
            mdi_loss_epoch.append(loss3.item())

            # for name, param in model.named_parameters():
            #     if param.requires_grad and param.grad is None:
            #         print(f"参数 {name} 未参与损失计算！")
            loss.backward()
            optim.step()
            # writecfl('/data0/senjia/recon/img',recon_coils.to('cpu').detach().numpy())
            
            if (step + 1) % 1 == 0:
                print('Epoch', epoch , '/', epoches, 'Step', step + 1, '/', each_epoch_steps,
                      'loss =', loss.item(), 'ssim = ', loss_ssim.item(),
                      'loss_mae = ', loss_mae.item(),
                      'loss_mdi=',loss3.item(),
                      # 'loss_S = ', S_loss.item(),
                      'time', time.time() - t0,
                      # 'S_mae = ', S_loss.item(),
                      )

        # scheduler.step()
        avgLoss = np.mean(train_loss_epoch)
        mdiloss = np.mean(mdi_loss_epoch)
        print('epoch', epoch , 'trnLoss:', avgLoss)
        savemat('RS.mat',{'S_map':S_map.detach().cpu().numpy(),'R1_map':R1_map.detach().cpu().numpy(),'R2_map':R2_map.detach().cpu().numpy()})
                
        total_loss.append(avgLoss)
        lr = optim.param_groups[0]['lr']
        
        # if (epoch ) % 2 == 1: 
        #     model.eval()
        #     with torch.no_grad():
        #         test_loss_epoch = []
        #         for step, sample in enumerate(test_loader):
        #             # if step > 27 and step< 310:
        #             if True:
        #                 t0 = time.time()
        #                 loss_iter = 0

        #                 ksp, sens,mask_img = sample
        #                 ksp, sens,mask_img = ksp.to(device),  sens.to(device),mask_img.to(device)

        #                 nb, nc, ne, ny, nz = ksp.size()
        #                 cg = ksp * mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        #                 # normalization
        #                 # label = ifft2c(ksp)  # [nb nc ne nx ny]
        #                 label = ifft2c(cg)  # [nb nc ne nx ny]
        #                 image = torch.stack([torch.sum(label[:, :, ne, :, :] * torch.conj(sens), dim=1) for ne in range(ne)], dim=1)
        #                 image = sos(label)  # [nb ne nx ny]
        #                 scale_factor, _ = torch.max(abs(image).view(1, -1), dim=1)
                        
        #                 scale_factor, _ = torch.max(image.view(1, -1), dim=1)
        #                 k0 = torch.div(ksp, scale_factor.view(1, 1, 1, 1, 1))
        #                 ku = torch.div(cg, scale_factor.view(1, 1, 1, 1, 1))

        #                 label_coils = ifft2c(k0)
        #                 label = torch.stack([torch.sum(label_coils[:, :, ne, :, :] * torch.conj(sens), dim=1) for ne in range(ne)], dim=1)
                
        #                 k0_u = k0 * mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        #                 img_coils = ifft2c(k0_u)
        #                 img = torch.stack([torch.sum(img_coils[:, :, ne, :, :] * torch.conj(sens), dim=1) for ne in range(ne)], dim=1)
                        
                        
        #                 # x0 = ku * (1 - mask).unsqueeze(0).unsqueeze(0).unsqueeze(0) 
        #                 masks = mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        #                 masks = torch.tile(masks,(nb,1,1,1,1)).real
        #                 recon,S_map,R1_map,R2_map= model(k0_u, sens,  masks, img,mask_img)  # [nb nc ne nx ny]
        #                 # recon = torch.stack([torch.sum(recon_coils[:, :, ne, :, :] * torch.conj(sens), dim=1) for ne in range(ne)], dim=1)
                
        #                 # recon = sos(recon_coils)  # [nb ne nx ny]
        #                 loss_mae = F.l1_loss(recon, label)
        #                 loss_ssim = SSIM(recon, label)
        #                 R1_label,R2_label,S_label = R_S_2(label.reshape(nb,ne,1,ny,nz))
            
        #                 R1_label = R1_label * mask_img
        #                 R2_label = R2_label* mask_img
        #                 S_label = S_label * mask_img
        #                 # R1_label = norm_r1(R1_label)
                        
        #                 R1_label = torch.stack((R1_label.real,R1_label.imag),dim=1)
        #                 R2_label = torch.stack((R2_label.real,R2_label.imag),dim=1)
        #                 S_label = torch.stack((S_label.real,S_label.imag),dim=1)
        #                 R1_map = torch.stack((R1_map.real,R1_map.imag),dim=1)
        #                 R2_map = torch.stack((R2_map.real,R2_map.imag),dim=1)
        #                 S_map = torch.stack((S_map.real,S_map.imag),dim=1)
        #                 loss1 =  loss_mae  + 0.5*(1-loss_ssim)
        #                 loss3 = criterion(R1_map,R1_label)  + criterion(S_map,S_label)+ criterion(R2_map,R2_label)
        #                 # loss3 = criterion(R1_map,R1_label)
        #                 loss = loss1 + 0.03 * loss3
        #                 test_loss_epoch.append(loss.item())
        #     test_Loss = np.mean(test_loss_epoch)
        #     with open(results_file, "a") as f:
        #         train_info = f"[epoch: {epoch}]\n" \
        #                     f"train_loss: {avgLoss:.4f}\n" \
        #                     f"test_loss: {test_Loss:.4f}\n"\
        #                     f"learning rate: {lr:.4f}\n"\

        #         f.write(train_info + "\n\n")
        # else:
        with open(results_file, "a") as f:
            train_info = f"[epoch: {epoch}]\n" \
                        f"train_loss: {avgLoss:.4f}\n" \
                        f"mdi_loss: {mdiloss:.4f}\n" \
                        f"learning rate: {lr:.4f}\n"\

            f.write(train_info + "\n\n")

    #     # save model
        if (epoch) % 2 == 1:                      
            torch.save(model.state_dict(),
                       "%s/net_params_%d.pkl" % (ckpt_path, epoch))  # save only the parameters


    # t0_end = time.time()
    # print('Total training time:', t0_end - t0_start)
