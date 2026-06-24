
# from skimage.measure.simple_metrics import compare_psnr
import numpy as np
import math
import torch
from skimage.metrics import structural_similarity


def mse(x, y):
    return np.mean(np.abs(x - y)**2)

def dice_coef(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    #output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    #target = target.view(-1).data.cpu().numpy()

    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)


def psnr(x, y):
    '''
    Measures the PSNR of recon w.r.t x.
    Image must be of either integer (0, 256) or float value (0,1)
    :param x: [m,n]
    :param y: [m,n]
    :return:
    '''
    assert x.shape == y.shape
    assert x.dtype == y.dtype or np.issubdtype(x.dtype, np.float) \
        and np.issubdtype(y.dtype, np.float)
    if x.dtype == np.uint8:
        max_intensity = 256
    else:
        max_intensity = 1

    mse = np.sum((x - y) ** 2).astype(float) / x.size
    return 20 * np.log10(max_intensity) - 10 * np.log10(mse)

#
# def complex_psnr(x, y, peak='max'):
#     '''
#     x: reference image
#     y: reconstructed image
#     peak: normalised or max
#
#     Notice that ``abs'' squares
#     Be careful with the order, since peak intensity is taken from the reference
#     image (taking from reconstruction yields a different value).
#
#     '''
#     mse = np.mean(np.abs(x - y)**2)
#     # print(mse)
#     # print(np.max(np.abs(x))**2)
#     if peak == 'max':
#         # print(np.max(np.abs(x)))
#         return 10*np.log10(np.max(np.abs(x))**2/mse)
#     else:
#         return 10*np.log10(1./mse)


def complex_psnr(x, y, peak='normalized'):
    '''
    x: reference image
    y: reconstructed image
    peak: normalised or max

    Notice that ``abs'' squares
    Be careful with the order, since peak intensity is taken from the reference
    image (taking from reconstruction yields a different value).

    '''
    # mse = np.mean(np.abs(np.abs(x)- np.abs(y))**2)
    mse = np.mean(np.abs(x-y) ** 2)
    if peak == 'max':
        return 10*np.log10(np.max(np.abs(x))**2/mse)
    else:
        return 10*np.log10(1./mse)


def real_psnr(x, y, peak='normalized'):
    '''
    x: reference image
    y: reconstructed image
    peak: normalised or max

    Notice that ``abs'' squares
    Be careful with the order, since peak intensity is taken from the reference
    image (taking from reconstruction yields a different value).

    '''
    mse = np.mean(np.abs(np.abs(x)- np.abs(y))**2)
    # mse = np.mean(np.abs(x-y) ** 2)
    if peak == 'max':
        return 10*np.log10(np.max(np.abs(x))**2/mse)
    else:
        return 10*np.log10(1./mse)

def for_psnr(b_ini,b_pred, b_label):

    ini = b_ini.cpu().data.numpy()
    pred = b_pred.cpu().data.numpy()
    label = b_label.cpu().data.numpy()
    psnr_pred = 0
    psnr_ini = 0

    psnr_pred_real = 0
    psnr_ini_real = 0
    i = 0


    for step1, img in enumerate(pred):


        image_ini = ini[step1][0] + ini[step1][1] * 1j
        image_pred = img[0] + img[1] * 1j
        image_label = label[step1][0] + label[step1][1] * 1j

        image_ini_abs = np.abs(image_ini)
        image_pred_abs = np.abs(image_pred)
        image_label_abs = np.abs(image_label)

        image_ini = image_ini/np.max(image_ini_abs)
        image_pred = image_pred/np.max(image_pred_abs)
        image_label = image_label/np.max(image_label_abs)

        psnr_ini += complex_psnr(image_label, image_ini, peak='max')
        psnr_pred += complex_psnr(image_label, image_pred, peak='max')
        psnr_ini_real += real_psnr(image_label, image_ini, peak='max')
        psnr_pred_real += real_psnr(image_label, image_pred, peak='max')

    return psnr_ini, psnr_pred,psnr_ini_real,psnr_pred_real


#
def complex_nrmse(x,y):
    '''
       x: reference image
       y: reconstructed image
       peak: normalised or max

       Notice that ``abs'' squares
       Be careful with the order, since peak intensity is taken from the reference
       image (taking from reconstruction yields a different value).

    '''
    denom = np.sqrt(np.mean((x*x),dtype=np.float64))
    mse = np.mean(np.abs(x - y) ** 2)
    out = np.sqrt(mse)/denom
    return out

def for_ssim(b_ini,b_pred, b_label):
    ini = b_ini.cpu().data.numpy()
    pred = b_pred.cpu().data.numpy()
    label = b_label.cpu().data.numpy()
    psnr_pred = 0
    psnr_ini = 0
    i = 0

    for step1, img in enumerate(pred):
        image_ini_real = ini[step1][0]
        image_ini_imag = ini[step1][1]
        image_pred_real = img[0]
        image_pred_imag = img[1]
        image_label_real = label[step1][0]
        image_label_imag = label[step1][1]
        image_label = image_label_real+image_label_imag*1j
        image_pred = image_pred_real + image_pred_imag * 1j
        image_ini = image_ini_real + image_ini_imag * 1j

        image_ini_abs = np.abs(image_ini)
        image_pred_abs = np.abs(image_pred)
        image_label_abs = np.abs(image_label)

        image_ini = np.abs(image_ini/np.max(image_ini_abs))
        image_pred = np.abs(image_pred/np.max(image_pred_abs))
        image_label = np.abs(image_label/np.max(image_label_abs))

        max_val = image_label.max()

        i += 1
        psnr_ini += structural_similarity(image_label,image_ini, data_range = max_val)
        psnr_pred +=  structural_similarity(image_label,image_pred, data_range = max_val)
    return psnr_ini, psnr_pred


def for_nrmse(b_ini,b_pred, b_label):
    ini = b_ini.cpu().data.numpy()
    pred = b_pred.cpu().data.numpy()
    label = b_label.cpu().data.numpy()
    psnr_pred = 0
    psnr_ini = 0
    i = 0

    for step1, img in enumerate(pred):
        image_ini_real = ini[step1][0]
        image_ini_imag = ini[step1][1]
        image_pred_real = img[0]
        image_pred_imag = img[1]
        image_label_real = label[step1][0]
        image_label_imag = label[step1][1]
        image_label = image_label_real + image_label_imag * 1j
        image_pred = image_pred_real + image_pred_imag * 1j
        image_ini = image_ini_real + image_ini_imag * 1j

        image_ini_abs = np.abs(image_ini)
        image_pred_abs = np.abs(image_pred)
        image_label_abs = np.abs(image_label)

        image_ini = np.abs(image_ini/np.max(image_ini_abs))
        image_pred = np.abs(image_pred/np.max(image_pred_abs))
        image_label = np.abs(image_label/np.max(image_label_abs))


        i += 1
        psnr_ini += complex_nrmse(image_label, image_ini)
        psnr_pred += complex_nrmse(image_label, image_pred)
    return psnr_ini, psnr_pred



def cal_psnr(b_ini, b_pred, b_label):
    ini = b_ini.cpu().data.numpy().squeeze()
    pred = b_pred.cpu().data.numpy().squeeze()
    label = b_label.cpu().data.numpy().squeeze()
    psnr_ini = complex_psnr(label, ini, peak='max')
    psnr_pred = complex_psnr(label, pred, peak='max')
    return psnr_ini, psnr_pred

def cal_nrmse(b_ini, b_pred, b_label):
    ini = b_ini.cpu().data.numpy().squeeze()
    pred = b_pred.cpu().data.numpy().squeeze()
    label = b_label.cpu().data.numpy().squeeze()
    psnr_ini = complex_nrmse(label, ini)
    psnr_pred = complex_nrmse(label, pred)
    return psnr_ini, psnr_pred

def cal_ssim(b_ini, b_pred, b_label):
    ini = b_ini.cpu().data.numpy().squeeze()
    pred = b_pred.cpu().data.numpy().squeeze()
    label = b_label.cpu().data.numpy().squeeze()
    psnr_ini = compare_ssim(label, ini, data_range=label.max())
    psnr_pred = compare_ssim(label, pred, data_range=label.max())
    return psnr_ini, psnr_pred

def learning_rate_org(init, epoch):
    optim_factor = 0
    if epoch > 160:
        optim_factor = 3
    elif epoch > 120:
        optim_factor = 2
    elif epoch > 60:
        optim_factor = 1
    return init * math.pow(0.2, optim_factor)

def learning_rate(init, epoch):
    optim_factor = 0
    if epoch >= 80:
        optim_factor = 4
    elif epoch >= 60:
        optim_factor = 3
    elif epoch >= 40:
        optim_factor = 2
    else:
        optim_factor = 1
    return init * math.pow(0.8, optim_factor)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s