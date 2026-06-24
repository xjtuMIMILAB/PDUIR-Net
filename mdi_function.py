import torch
import scipy.io as sio

# T2
def calculate_T2(S2, Nc, Na, N2, delta_TE):
    """
    Calculate T2star_MDI and R2star_MDI from a 5D S2 tensor.

    Parameters:
    S2 (torch.Tensor): 5D tensor with dimensions (x, y, z, coil, echo, flip angle)
    Nc (int): Number of coils
    Na (int): Number of flip angles
    N2 (int): Number of echoes
    delta_TE (float): Echo spacing in seconds

    Returns:
    tuple: (T2star_MDI_abs, R2star_MDI_abs)
    """
    # Initialize a and b as complex and real numbers respectively
    
    a = torch.zeros_like(S2[...,0,0,0], dtype=torch.complex64)
    b = torch.zeros_like(S2[...,0,0,0], dtype=torch.float32)
    
    for iter_coil in range(Nc):
        for iter_fa in range(Na):
            for iter_echo in range(N2 - 1):
                a += torch.conj(S2[..., iter_coil, iter_echo, iter_fa]) * S2[..., iter_coil, iter_echo + 1, iter_fa]
                b += torch.abs(S2[..., iter_coil, iter_echo, iter_fa]) ** 2

    delta_S = a / (b + 1e-8)  # signal ratio between neighboring echoes

    T2star_MDI = -delta_TE / (torch.log(torch.abs(delta_S)) + 1e-8)
    T2star_MDI_abs = torch.abs(T2star_MDI)

    R2star_MDI = -torch.log(torch.abs(delta_S)) / (delta_TE * 1e-3)
    R2star_MDI_abs = torch.abs(R2star_MDI)

    return T2star_MDI_abs, R2star_MDI_abs


# T1 via TR1
def calculate_T1_TR1(S1, Nc, a1, a2, n, TR1):
    A, B = 0, 0
    for coil in range(Nc):
        A += torch.conj(S1[..., coil, 0, 1]) * S1[..., coil, 0, 0]
        B += torch.abs(S1[..., coil, 0, 1]) ** 2

    R1 = A / (B + 1e-8) * (torch.sin(a2) / torch.sin(a1))
    a = n * (n + 1) * torch.cos(a1) * torch.cos(a2) * (R1 * torch.cos(a1) - torch.cos(a2))
    b = (n + 1) * ((n + torch.cos(a1)) * torch.cos(a2) ** 2 - (n + torch.cos(a2)) * torch.cos(a1) ** 2 * R1) + \
        n * (R1 * torch.sin(a1) ** 2 * torch.cos(a2) - torch.cos(a1) * torch.sin(a2) ** 2)
    c = torch.sin(a2) ** 2 * (n + torch.cos(a1)) - R1 * torch.sin(a1) ** 2 * (n + torch.cos(a2))

    k = (-b - torch.sqrt(b ** 2 - 4 * a * c)) / (2 * a + 1e-8)
    return TR1 / (torch.abs(k) + 1e-8)


# T1 via TR2
def calculate_T1_TR2(S2, Nc, N2, a1, a2, n, TR1):
    A, B = 0, 0
    for coil in range(Nc):
        for TE in range(N2):
            A += torch.conj(S2[..., coil, TE, 1]) * S2[..., coil, TE, 0]
            B += torch.abs(S2[..., coil, TE, 1]) ** 2
    R2 = A / (B + 1e-8) 
    R2 = R2 * (torch.sin(a2) / torch.sin(a1))
    a = n * (n + 1) * torch.cos(a1) * torch.cos(a2) * (torch.cos(a1) * R2 - torch.cos(a2))
    b = (n + 1) * ((1 + n * torch.cos(a1)) * torch.cos(a2) ** 2 - (1 + n * torch.cos(a2)) * torch.cos(a1) ** 2 * R2) + \
        n * (R2 * torch.sin(a1) ** 2 * torch.cos(a2) - torch.cos(a1) * torch.sin(a2) ** 2)
    c = torch.sin(a2) ** 2 * (1 + n * torch.cos(a1)) - R2 * torch.sin(a1) ** 2 * (1 + n * torch.cos(a2))

    k = (-b - torch.sqrt(b ** 2 - 4 * a * c)) / (2 * a + 1e-8)
    return TR1 / (torch.abs(k) + 1e-8)


# Augmented T1-weighted image
def calculate_augmented_T1(S1, S2, Nc, N1, N2):
    a, b = 0, 0
    for coil in range(Nc):
        for TE in range(N1):
            a += torch.conj(S1[..., coil, TE, 0]) * S1[..., coil, TE, 1]
            b += torch.abs(S1[..., coil, TE, 0]) ** 2
        for TE in range(N2):
            a += torch.conj(S2[..., coil, TE, 0]) * S2[..., coil, TE, 1]
            b += torch.abs(S2[..., coil, TE, 0]) ** 2

    return torch.abs(a) / (b + 1e-8)

def calculate_R(S1,S2,Nc,N2):
    A, B = 0, 0
    for coil in range(Nc):
        A += torch.conj(S1[..., coil, 0, 1]) * S1[..., coil, 0, 0]
        B += torch.abs(S1[..., coil, 0, 1]) ** 2

    R1 = A / (B + 1e-8)
    
    A, B = 0, 0
    for coil in range(Nc):
        for TE in range(N2):
            A += torch.conj(S2[..., coil, TE, 1]) * S2[..., coil, TE, 0]
            B += torch.abs(S2[..., coil, TE, 1]) ** 2
    R2 = A / (B + 1e-8) 

    return R1,R2

def calculate_S(S2,Nc,Na,N2):
    a = torch.zeros_like(S2[...,0,0,0], dtype=torch.complex64)
    b = torch.zeros_like(S2[...,0,0,0], dtype=torch.float32)
    
    for iter_coil in range(Nc):
        for iter_fa in range(Na):
            for iter_echo in range(N2 - 1):
                a += torch.conj(S2[..., iter_coil, iter_echo, iter_fa]) * S2[..., iter_coil, iter_echo + 1, iter_fa]
                b += torch.abs(S2[..., iter_coil, iter_echo, iter_fa]) ** 2

    delta_S = a / (b + 1e-8)  # signal ratio between neighboring echoes
    return delta_S

def calculate_S_only(S2,Nc,Na,N2):
    a = torch.zeros_like(S2[...,0,0], dtype=torch.complex64)
    b = torch.zeros_like(S2[...,0,0], dtype=torch.float32)
    
    for iter_coil in range(Nc):
        for iter_fa in range(Na):
            for iter_echo in range(N2 - 1):
                a += torch.conj(S2[..., iter_coil, iter_echo]) * S2[..., iter_coil, iter_echo + 1]
                b += torch.abs(S2[..., iter_coil, iter_echo]) ** 2

    delta_S = a / (b + 1e-8)  # signal ratio between neighboring echoes
    return delta_S
    
    
def T1_T2(img):
    TR1 = torch.tensor(7.9700).cuda()  # TR 时间
    TR2 = torch.tensor(38.2300).cuda()  # TR 时间
    N1 = torch.tensor(1)  # 回波数量
    N2 = torch.tensor(5)  # 回波数量
    a1 = torch.tensor(0.0698)  # 翻转角
    a2 = torch.tensor(0.2793)  # 翻转角
    delta_TE = torch.tensor(6.4400)  # Echo spacing
    Na = torch.tensor(2)  # 翻转角数量
    Necho = (N1 + N2) * Na     # 总回波数量
    # 初始化变量
    Nx, Ny, Nz, Nc, Necho = torch.tensor(336), torch.tensor(288), torch.tensor(84), torch.tensor(12), torch.tensor(12)  # 示例维度
    
    S1 = img[..., :, [0, N1 + N2]]  # TR1 信号
    S2 = torch.stack((img[..., :, list(range(N1, N1 + N2))] , img[..., :, list(range(N2 + N1 + N1, Na * (N2 + N1)))]),axis=-1)
    nb,ny,nz,nc,ne = img.shape
    # 重新调整形状
    S1 = S1.reshape(nb, Ny, Nz, Nc, N1, Na)  # TR1 信号重新整形
    S2 = S2.reshape(nb, Ny, Nz, Nc, N2, Na)  # TR2 信号重新整形
    
    eps = 1e-8  # 数值稳定的小常数
    n = TR2 / TR1
    T1_TR1 = calculate_T1_TR1(S1, Nc, a1, a2, n, TR1)
    T1_TR2 = calculate_T1_TR2(S2, Nc, N2, a1, a2, n, TR1)
    T1_all = torch.stack((T1_TR1, T1_TR2), dim=-1)
    T2star_MDI_abs, R2star_MDI_abs = calculate_T2(S2, Nc, Na, N2, delta_TE) 
    return T1_all.squeeze(), T2star_MDI_abs.squeeze()

def R_S(img):
    nb, Necho, ch, Ny, Nz = img.shape  # 示例维度
    img = img.permute(0,3,4,2,1)
    img = ((img[:,:,:,0,:] + 1j * img[:,:,:,1,:])).reshape(nb,Ny,Nz,1,Necho)
    N1 = torch.tensor(1)  # 回波数量
    N2 = torch.tensor(5)  # 回波数量
    Na = torch.tensor(2)  # 翻转角数量
    Nc = torch.tensor(1) # coil number
    # img.size: nb 288 84 1 12
    

    S1 = img[..., :, [0, N1 + N2]]  # TR1 信号
    S2 = torch.stack((img[..., :, list(range(N1, N1 + N2))] , img[..., :, list(range(N2 + N1 + N1, Na * (N2 + N1)))]),dim=-1)

    # sio.savemat('img.mat',{'S1':S1.detach().cpu().numpy(),'S2':S2.detach().cpu().numpy(),'img':img.detach().cpu().numpy()})
    # 初始化变量
    
    # 重新调整形状
    S1 = S1.reshape(nb, Ny, Nz, Nc, N1, Na)  # TR1 信号重新整形
    S2 = S2.reshape(nb, Ny, Nz, Nc, N2, Na)  # TR2 信号重新整形
    # sio.savemat('img_2.mat',{'S1':S1.detach().cpu().numpy(),'S2':S2.detach().cpu().numpy()})
    R1,R2 = calculate_R(S1,S2,Nc,N2)
    S = calculate_S(S2,Nc,Na,N2)
    # sio.savemat('S_R.mat',{'S':S.detach().cpu().numpy(),'R1':R1.detach().cpu().numpy(),'R2':R2.detach().cpu().numpy()})
    
    return R1,R2,S


def R_S_2(img):
    nb, Necho, Nc, Ny, Nz = img.shape  # 示例维度
    img = img.permute(0,3,4,2,1)
    # img = ((img[:,:,:,0,:] + 1j * img[:,:,:,1,:])).reshape(nb,Ny,Nz,1,Necho)
    N1 = torch.tensor(1)  # 回波数量
    N2 = torch.tensor(5)  # 回波数量
    Na = torch.tensor(2)  # 翻转角数量
    # Nc = torch.tensor(1) # coil number
    # img.size: nb 288 84 1 12
    

    S1 = img[..., :, [0, N1 + N2]]  # TR1 信号
    S2 = torch.stack((img[..., :, list(range(N1, N1 + N2))] , img[..., :, list(range(N2 + N1 + N1, Na * (N2 + N1)))]),dim=-1)

    # sio.savemat('img.mat',{'S1':S1.detach().cpu().numpy(),'S2':S2.detach().cpu().numpy(),'img':img.detach().cpu().numpy()})
    # 初始化变量
    
    # 重新调整形状
    S1 = S1.reshape(nb, Ny, Nz, Nc, N1, Na)  # TR1 信号重新整形
    S2 = S2.reshape(nb, Ny, Nz, Nc, N2, Na)  # TR2 信号重新整形
    # sio.savemat('img_2.mat',{'S1':S1.detach().cpu().numpy(),'S2':S2.detach().cpu().numpy()})
    R1,R2 = calculate_R(S1,S2,Nc,N2)
    S = calculate_S(S2,Nc,Na,N2)
    # sio.savemat('S_R.mat',{'S':S.detach().cpu().numpy(),'R1':R1.detach().cpu().numpy(),'R2':R2.detach().cpu().numpy()})
    
    return R1,R2,S