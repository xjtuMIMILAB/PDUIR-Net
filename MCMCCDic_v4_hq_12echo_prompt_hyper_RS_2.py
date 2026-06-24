import common
import torch.nn as nn
import torch.nn.functional as F
import torch
import fastmri
import math
import mdi_function as mdi
"""
model framework:
(1) Init using D^H
for i in range(T):
    (2) Update Block
    (3) Mid_reconstruct with DCM
    (4) ReUpdate Block
(5) Final_reconstruct
"""

"""
v4 Update:
ReUpdate Block:
    1) reupdate C & U with 'x_cur_rec-x_zf' instead of 'x_cur_rec'
    tips: fix bugs such as reconstruct with QU instead of DU !!!
To Do:
    2) try to split C & U with low_frequency & high_frequency using new mask
    3) nolinear Conv Dictionary using sin & cos
"""


def norm_r1(R1_map,max_magnitude=3.0):
    magnitude = torch.abs(R1_map)
    scale = torch.where(magnitude > max_magnitude, 
                       max_magnitude / magnitude, 
                       torch.ones_like(magnitude))
    return R1_map * scale


class generate(nn.Module):
        def __init__(self,input_channels) -> None:
            super(generate,self).__init__()      
            self.embedding = nn.Linear(24,input_channels[0])
            self.mlp1 = nn.Linear(input_channels[0],input_channels[1])
            self.mlp2 = nn.Linear(input_channels[1],input_channels[2])
        def code_embedding(self,code):
            freqs = (torch.exp(-math.log(100) * torch.arange(start=0,end=12,dtype =torch.float32))/12).cuda()
            # code = np.array(code, dtype=np.float32)
            # code =torch.tensor([code]).cuda()
            args = code[:,None].float() *freqs[None,:]
            embedding = torch.cat([torch.cos(args),torch.sin(args)],dim=-1)
            return embedding    

        def forward(self,hot_code):
            code = self.code_embedding(hot_code)
            input_param1 = self.embedding(code)
            input_param2 = self.mlp1(input_param1)
            input_param3 = self.mlp2(input_param2)
            input_param = {'input_param1':input_param1,
                        'input_param2':input_param2,
                        'input_param3':input_param3,
                        
            }
            return input_param

def get_all_conv(net, conv_list = []):

    for name, layer in net._modules.items():
        if not isinstance(layer, nn.Conv2d):
            get_all_conv(layer, conv_list)
        elif isinstance(layer, nn.Conv2d):
           # it's a Conv layer. Register a hook
            conv_list.append(layer)

    for name, layer in net._modules.items():
        if not isinstance(layer, nn.ConvTranspose2d):
            get_all_conv(layer, conv_list)
        elif isinstance(layer, nn.ConvTranspose2d):
           # it's a Conv layer. Register a hook
            conv_list.append(layer)

    return conv_list


def relu(x, lambd):
    lambd = nn.functional.relu(lambd)
    return nn.functional.relu(x - lambd.to(x.device))


class adjoint_conv_op(nn.Module):
    # The adjoint of a conv module.
    def __init__(self, conv_op):
        super().__init__()
        in_channels = conv_op.out_channels
        out_channels = conv_op.in_channels
        kernel_size = conv_op.kernel_size
        padding = kernel_size[0] // 2

        # transpose convolution 
        self.transpose_conv = nn.ConvTranspose2d(in_channels, out_channels,  kernel_size=kernel_size, padding= padding, bias= False)
        
        # tie the weights of transpose convolution with convolution 
        self.transpose_conv.weight = conv_op.weight

    def forward(self, x):
        return self.transpose_conv(x)


class up_block(nn.Module):
    """
    A module that contains:
    (1) an up-sampling operation (implemented by bilinear interpolation or upsampling)
    (2) convolution operations
    """
        
    def __init__(self, kernel_size, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # the up-sampling operation
        self.up = nn.ConvTranspose2d(in_channels , in_channels-32, kernel_size=2, stride=2, bias= False)
        
        # the 2d convolution operation
        self.conv = nn.Conv2d((in_channels-32)*2, out_channels, kernel_size=kernel_size, padding= kernel_size // 2, bias= False)
 
    def forward(self, x1, x2):
        # print(x1.shape)
        x1 = self.up(x1)
        # print(x1.shape)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # input is CHW
        x = torch.cat([x2, x1], dim=1)
        # print(x.shape)
        return self.conv(x)
    

class adjoint_up_block(nn.Module):
    # adjoint of up_block module
    
    def __init__(self, up_block_model):
        super().__init__()
        
        # to construct the adjoint model, one should exclude additive biases and use transposed conv for upsampling.
        
        in_channels = up_block_model.out_channels
        out_channels = up_block_model.in_channels
        
        self.adjoint_conv_op = adjoint_conv_op(up_block_model.conv)
        self.adjoint_up =  nn.Conv2d(in_channels , in_channels // 2, kernel_size=2, stride=2, bias= False)
        self.adjoint_up.weight = up_block_model.up.weight
        
        
    def forward(self, x):
        x = self.adjoint_conv_op(x)
        # input is CHW
        x2 = x[:, :int(x.shape[1]/2), :, :]
        x1 = x[:, int(x.shape[1]/2):, :, :]
        x1 = self.adjoint_up(x1)
        return (x1, x2)


class out_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(out_conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias= False)
    def forward(self, x):
        return self.conv(x)    
    

class adjoint_out_conv(nn.Module):
    def __init__(self, out_conv_model):
        super().__init__()
        in_channels = out_conv_model.out_channels
        out_channels = out_conv_model.in_channels

        self.adjoint_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, bias= False)
        self.adjoint_conv.weight = out_conv_model.conv.weight

    def forward(self, x):
        return self.adjoint_conv(x)
    
    
class dictionary_model(nn.Module):
    def __init__(self,  kernel_size, hidden_layer_width_list, n_classes):
        super(dictionary_model, self).__init__()
        
        self.hidden_layer_width_list = hidden_layer_width_list
        
        in_out_list = [ [hidden_layer_width_list[i], hidden_layer_width_list[i+1]] for i in  range(len(hidden_layer_width_list) -1) ]

        self.num_hidden_layers = len(in_out_list)
        self.generate_model = generate([64,96,128])
        self.n_classes = n_classes

        # the initial convolution on the bottleneck layer
        self.bottleneck_conv = nn.Conv2d(hidden_layer_width_list[0], hidden_layer_width_list[0], kernel_size=kernel_size, padding= kernel_size // 2, bias= False)

        self.syn_up_list = []

        for layer_idx in range(self.num_hidden_layers):
            new_up_block = up_block(kernel_size, *in_out_list[layer_idx])
            self.syn_up_list.append(new_up_block)           
        
        self.syn_up_list = nn.Sequential( *self.syn_up_list )
        
        self.syn_outc = out_conv(hidden_layer_width_list[-1], n_classes)
    def _adjust_channels(self, x_prev, param):
        # 将生成的参数与 x_prev 的通道数相乘以调整通道数
        # 假设 param 的形状是 (batch_size, channels)
        # 我们需要将其扩展到与 x_prev 的形状匹配 (batch_size, channels, height, width)
        param_expanded = param.unsqueeze(-1).unsqueeze(-1)  # (batch_size, channels, 1, 1)
        x_prev_adjusted = x_prev * param_expanded  # 逐通道相乘
        return x_prev_adjusted
    def forward(self, x_list, hot_code):
        nb = x_list[0].shape[0]
        hot_code = hot_code.repeat(nb, 1)
        input_param = self.generate_model(hot_code)
        generate_param1 = input_param['input_param1'].squeeze(1)
        generate_param2 = input_param['input_param2'].squeeze(1)
        generate_param3 = input_param['input_param3'].squeeze(1)
        num_res_levels = len(x_list)
                
#         x_prev = x_list[0]
        x_prev = self.bottleneck_conv(x_list[0])
        
        adjusted_x_prev = self._adjust_channels(x_prev, generate_param3)
        
        for i in range(1, num_res_levels):
            x = x_list[i] 
            syn_up = self.syn_up_list[i-1]
            x_prev = syn_up(adjusted_x_prev, x)
            if i == 1:
                adjusted_x_prev = self._adjust_channels(x_prev, generate_param2)
            elif i == 2:
                adjusted_x_prev = self._adjust_channels(x_prev, generate_param1)
        syn_output = self.syn_outc(adjusted_x_prev)
        return syn_output


class adjoint_dictionary_model(nn.Module):
    def __init__(self, dictionary_model):
        super().__init__()
        
        
        self.adjoint_syn_outc = adjoint_out_conv(dictionary_model.syn_outc)
        self.adjoint_syn_bottleneck_conv = adjoint_conv_op(dictionary_model.bottleneck_conv)        
        
        self.adjoint_syn_up_list = []
        
        self.num_hidden_layers = dictionary_model.num_hidden_layers
        self.generate_model = generate([64,96,128])
        for layer_idx in range(dictionary_model.num_hidden_layers): 
            self.adjoint_syn_up_list.append(adjoint_up_block(dictionary_model.syn_up_list[layer_idx] ) )
            
    def _adjust_channels(self, y, param):
            # 将生成的参数与 y 的通道数相乘以调整通道数
            # 假设 param 的形状是 (batch_size, channels)
            # 我们需要将其扩展到与 y 的形状匹配 (batch_size, channels, height, width)
            param_expanded = param.unsqueeze(-1).unsqueeze(-1)  # (batch_size, channels, 1, 1)
            y_adjusted = y * param_expanded  # 逐通道相乘
            return y_adjusted
    def forward(self, y, hot_code):
        # 使用 generate 模型生成参数
        nb = y.shape[0]
        hot_code = hot_code.repeat(nb, 1)
        input_param = self.generate_model(hot_code)
        generate_param1 = input_param['input_param1'].squeeze(1)
        generate_param2 = input_param['input_param2'].squeeze(1)
        generate_param3 = input_param['input_param3'].squeeze(1)
        
        
        y = self.adjoint_syn_outc(y)
        y = self._adjust_channels(y, generate_param1)
        x_list = []
        if self.num_hidden_layers >= 1:
            y = self._adjust_channels(y, generate_param1)
        for layer_idx in range(self.num_hidden_layers-1, -1, -1):  
            adjoint_syn_up = self.adjoint_syn_up_list[layer_idx]   # 下采样
            y, x = adjoint_syn_up(y)
            if layer_idx == 1:
                y = self._adjust_channels(y, generate_param2)
            elif layer_idx == 2:
                y = self._adjust_channels(y, generate_param3)
            x_list.append(x)
        y = self.adjoint_syn_bottleneck_conv(y)            
        x_list.append(y)
        x_list.reverse()
        return x_list 


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup_c(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, n_resblocks):
        super(ResidualGroup_c, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        # res += x
        return res
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, prompt_dim, prompt_len, prompt_size, kernel_size, n_resblocks):
        super(ResidualGroup, self).__init__()
        self.prompt_block = PromptBlock(prompt_dim=prompt_dim, prompt_len=prompt_len, prompt_size=prompt_size, lin_dim=n_feat, learnable_input_prompt=True)
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat+prompt_dim, kernel_size, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat+prompt_dim, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        prompt = self.prompt_block(x)
        x_cat = torch.cat([x, prompt], dim=1)
        res = self.body(x_cat)
        # res += x
        return res

# Prompt Block(PromptMR)
class PromptBlock(nn.Module):
    def __init__(self, prompt_dim=24, prompt_len=5, prompt_size=64, lin_dim=64, learnable_input_prompt = True):
        super(PromptBlock, self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(
            1, prompt_len, prompt_dim, prompt_size, prompt_size), requires_grad=learnable_input_prompt)
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.dec_conv3x3 = nn.Conv2d(
            prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):

        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1))
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)
        prompt_param = self.prompt_param.unsqueeze(0).repeat(B, 1, 1, 1, 1, 1).squeeze(1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * prompt_param
        prompt = torch.sum(prompt, dim=1)

        prompt = F.interpolate(prompt, (H, W), mode="bilinear")
        prompt = self.dec_conv3x3(prompt)

        return prompt
   
    
class dictionary_model_c(nn.Module):
    def __init__(self,  kernel_size, hidden_layer_width_list, n_classes):
        super(dictionary_model_c, self).__init__()
        
        self.hidden_layer_width_list = hidden_layer_width_list
        
        in_out_list = [ [hidden_layer_width_list[i], hidden_layer_width_list[i+1]] for i in  range(len(hidden_layer_width_list) -1) ]

        self.num_hidden_layers = len(in_out_list)
        
        self.n_classes = n_classes

        # the initial convolution on the bottleneck layer
        self.bottleneck_conv = nn.Conv2d(hidden_layer_width_list[0], hidden_layer_width_list[0], kernel_size=kernel_size, padding= kernel_size // 2, bias= False)

        self.syn_up_list = []

        for layer_idx in range(self.num_hidden_layers):
            new_up_block = up_block(kernel_size, *in_out_list[layer_idx])
            self.syn_up_list.append(new_up_block)           
        
        self.syn_up_list = nn.Sequential( *self.syn_up_list )
        
        self.syn_outc = out_conv(hidden_layer_width_list[-1], n_classes)

    def forward(self, x_list):

        # x_list is ordered from wide-channel to thin-channel.
        num_res_levels = len(x_list)
                
#         x_prev = x_list[0]
        x_prev = self.bottleneck_conv(x_list[0])
    
        for i in range(1, num_res_levels):
            x = x_list[i] 
            syn_up = self.syn_up_list[i-1]
            x_prev = syn_up(x_prev, x)
            
        syn_output = self.syn_outc(x_prev)
        return syn_output


class adjoint_dictionary_model_c(nn.Module):
    def __init__(self, dictionary_model_c):
        super().__init__()
        
        
        self.adjoint_syn_outc = adjoint_out_conv(dictionary_model_c.syn_outc)
        self.adjoint_syn_bottleneck_conv = adjoint_conv_op(dictionary_model_c.bottleneck_conv)        
        
        self.adjoint_syn_up_list = []
        
        self.num_hidden_layers = dictionary_model_c.num_hidden_layers
        
        for layer_idx in range(dictionary_model_c.num_hidden_layers): 
            self.adjoint_syn_up_list.append(adjoint_up_block(dictionary_model_c.syn_up_list[layer_idx] ) )
            

    def forward(self, y):
        y = self.adjoint_syn_outc(y)
        x_list = []
        
        for layer_idx in range(self.num_hidden_layers-1, -1, -1):  
            adjoint_syn_up = self.adjoint_syn_up_list[layer_idx]   # 下采样
            y, x = adjoint_syn_up(y)
            x_list.append(x)
        y = self.adjoint_syn_bottleneck_conv(y)            
        x_list.append(y)
        x_list.reverse()
        return x_list 
    
def code(i,length=12):
    zero_vector = torch.zeros(length).cuda()
    zero_vector[i-1] = 1
    return zero_vector

from unet import Unet
class ista_unet(nn.Module):
    def __init__(self, kernel_size=3, hidden_layer_width_list=[256,128,64],prompt_dim=[72, 48, 24], prompt_len=[5, 5, 5],
                 prompt_size=[16, 32, 64], n_classes=64, ista_num_steps=4, lasso_lambda_scalar=0.01):

        super(ista_unet, self).__init__()
        
        self.r1 = nn.ModuleList([Unet(in_chans=2 ,out_chans=2) for _ in range(ista_num_steps)])
        self.r2 = nn.ModuleList([Unet(in_chans=2 ,out_chans=2) for _ in range(ista_num_steps)])
        self.s = nn.ModuleList([Unet(in_chans=2 ,out_chans=2) for _ in range(ista_num_steps)])
        self.mu1 = nn.Parameter(torch.tensor([1e-4]), requires_grad=True)
        self.mu2 = nn.Parameter(torch.tensor([1e-4]), requires_grad=True)
        self.mu3 = nn.Parameter(torch.tensor([1e-4]), requires_grad=True)
        self.mu4 = nn.Parameter(torch.tensor([1e-4]), requires_grad=True)
        
        
        # self.RSZnet = nn.ModuleList([update_RS_Z() for _ in range(ista_num_steps)])
        # self.RSZ_net = update_RS()
        self.n_classes = n_classes
        self.ista_num_steps = ista_num_steps
        self.lasso_lambda_scalar = lasso_lambda_scalar
        self.hidden_layer_width_list = hidden_layer_width_list
        self.num_layers = len(hidden_layer_width_list)
        self.prompt_dim = prompt_dim
        self.prompt_len = prompt_len
        self.prompt_size = prompt_size
        self.lambda_1 = nn.Parameter(torch.tensor(0.1))
        self.beta_1 =  nn.Parameter(torch.tensor(0.1))
        # list to image parameters  ----->  T1_u
        # D_i^u
        self.encoder_dictionary_u = dictionary_model(kernel_size, hidden_layer_width_list, n_classes)
        [torch.nn.init.kaiming_uniform_(conv.weight, mode = 'fan_in', nonlinearity='linear') for conv in get_all_conv(self.encoder_dictionary_u)]
        # image to list parameters
        self.precond_encoder_dictionary_u = dictionary_model(kernel_size, hidden_layer_width_list, n_classes)
        self.precond_encoder_dictionary_u.load_state_dict(self.encoder_dictionary_u.state_dict())  # initialize with the same atoms
        self.adjoint_encoder_dictionary_u = adjoint_dictionary_model(self.precond_encoder_dictionary_u)  
        # list to image parameters ; for reconstruction 
        
        # Q_i^u
        
        self.decoder_dictionary_u = dictionary_model(kernel_size, hidden_layer_width_list, n_classes)
        self.decoder_dictionary_u.load_state_dict(self.encoder_dictionary_u.state_dict()) # initialize with the same atoms
        
        # image to list parameters ; for reconstruction
        self.precond_decoder_dictionary_u = dictionary_model(kernel_size, hidden_layer_width_list, n_classes)
        self.precond_decoder_dictionary_u.load_state_dict(self.decoder_dictionary_u.state_dict())  # initialize with the same atoms
        self.adjoint_decoder_dictionary_u = adjoint_dictionary_model(self.precond_decoder_dictionary_u)

        
        # list to image parameters  ----->  c
        # one_hot 
        # D^c
        self.encoder_dictionary_c = dictionary_model_c(kernel_size, hidden_layer_width_list, n_classes)
        [torch.nn.init.kaiming_uniform_(conv.weight, mode = 'fan_in', nonlinearity='linear') for conv in get_all_conv(self.encoder_dictionary_c)];
        self.encoder_dictionary_c2 = dictionary_model(kernel_size, hidden_layer_width_list, n_classes)
        [torch.nn.init.kaiming_uniform_(conv.weight, mode = 'fan_in', nonlinearity='linear') for conv in get_all_conv(self.encoder_dictionary_c2)]
        
        # image to list parameters
        self.precond_encoder_dictionary_c = dictionary_model_c(kernel_size, hidden_layer_width_list, n_classes)
        self.precond_encoder_dictionary_c.load_state_dict(self.encoder_dictionary_c.state_dict())  # initialize with the same atoms
        self.adjoint_encoder_dictionary_c = adjoint_dictionary_model_c(self.precond_encoder_dictionary_c)  
        # list to image parameters ; for reconstruction 
        
        # gaiyixia
        self.decoder_dictionary_c = dictionary_model_c(kernel_size, hidden_layer_width_list, n_classes) # 2*3
        self.decoder_dictionary_c.load_state_dict(self.encoder_dictionary_c.state_dict()) # initialize with the same atoms
        
        self.decoder_dictionary_c2 = dictionary_model(kernel_size, hidden_layer_width_list, n_classes) 
        self.decoder_dictionary_c2.load_state_dict(self.decoder_dictionary_c2.state_dict()) # initialize with the same atoms
        
        
        
        # image to list parameters ; for reconstruction 
        self.precond_decoder_dictionary_c = dictionary_model_c(kernel_size, hidden_layer_width_list, n_classes)
        self.precond_decoder_dictionary_c.load_state_dict(self.decoder_dictionary_c.state_dict())  # initialize with the same atoms
        self.adjoint_decoder_dictionary_c = adjoint_dictionary_model_c(self.precond_decoder_dictionary_c)
      

        with torch.no_grad():
            L_T1_u = self.power_iteration_conv_model(self.encoder_dictionary_c, num_simulations = 20)
        self.ista_stepsize_iter_list_T1_u = [nn.Parameter(torch.ones(1)/L_T1_u) for i in range(ista_num_steps)]
        _lasso_lambda_iter_list_T1_u = [[torch.nn.Parameter(lasso_lambda_scalar * torch.ones(1, width, 1, 1) ) for width in hidden_layer_width_list] for i in range(ista_num_steps)]         
        self.lasso_lambda_iter_list_T1_u =  [item for sublist in _lasso_lambda_iter_list_T1_u for item in sublist]

        with torch.no_grad():
            L_T2_u = self.power_iteration_conv_model(self.encoder_dictionary_c, num_simulations = 20)
        self.ista_stepsize_iter_list_T2_u = [nn.Parameter(torch.ones(1)/L_T2_u) for i in range(ista_num_steps)]
        _lasso_lambda_iter_list_T2_u = [[torch.nn.Parameter(lasso_lambda_scalar * torch.ones(1, width, 1, 1) ) for width in hidden_layer_width_list] for i in range(ista_num_steps)]         
        self.lasso_lambda_iter_list_T2_u =  [item for sublist in _lasso_lambda_iter_list_T2_u for item in sublist]
        with torch.no_grad():
            L_T3_u = self.power_iteration_conv_model(self.encoder_dictionary_c, num_simulations = 20)
        self.ista_stepsize_iter_list_T3_u = [nn.Parameter(torch.ones(1)/L_T3_u) for i in range(ista_num_steps)]
        _lasso_lambda_iter_list_T3_u = [[torch.nn.Parameter(lasso_lambda_scalar * torch.ones(1, width, 1, 1) ) for width in hidden_layer_width_list] for i in range(ista_num_steps)]         
        self.lasso_lambda_iter_list_T3_u =  [item for sublist in _lasso_lambda_iter_list_T3_u for item in sublist]

        with torch.no_grad():
            L_T4_u = self.power_iteration_conv_model(self.encoder_dictionary_c, num_simulations = 20)
        self.ista_stepsize_iter_list_T4_u = [nn.Parameter(torch.ones(1)/L_T4_u) for i in range(ista_num_steps)]
        _lasso_lambda_iter_list_T4_u = [[torch.nn.Parameter(lasso_lambda_scalar * torch.ones(1, width, 1, 1) ) for width in hidden_layer_width_list] for i in range(ista_num_steps)]         
        self.lasso_lambda_iter_list_T4_u =  [item for sublist in _lasso_lambda_iter_list_T4_u for item in sublist]
        with torch.no_grad():
            L_T5_u = self.power_iteration_conv_model(self.encoder_dictionary_c, num_simulations = 20)
        self.ista_stepsize_iter_list_T5_u = [nn.Parameter(torch.ones(1)/L_T5_u) for i in range(ista_num_steps)]
        _lasso_lambda_iter_list_T5_u = [[torch.nn.Parameter(lasso_lambda_scalar * torch.ones(1, width, 1, 1) ) for width in hidden_layer_width_list] for i in range(ista_num_steps)]         
        self.lasso_lambda_iter_list_T5_u =  [item for sublist in _lasso_lambda_iter_list_T5_u for item in sublist]

        with torch.no_grad():
            L_T6_u = self.power_iteration_conv_model(self.encoder_dictionary_c, num_simulations = 20)
        self.ista_stepsize_iter_list_T6_u = [nn.Parameter(torch.ones(1)/L_T6_u) for i in range(ista_num_steps)]
        _lasso_lambda_iter_list_T6_u = [[torch.nn.Parameter(lasso_lambda_scalar * torch.ones(1, width, 1, 1) ) for width in hidden_layer_width_list] for i in range(ista_num_steps)]         
        self.lasso_lambda_iter_list_T6_u =  [item for sublist in _lasso_lambda_iter_list_T6_u for item in sublist]        
        
        with torch.no_grad():
            L_T7_u = self.power_iteration_conv_model(self.encoder_dictionary_c, num_simulations = 20)
        self.ista_stepsize_iter_list_T7_u = [nn.Parameter(torch.ones(1)/L_T7_u) for i in range(ista_num_steps)]
        _lasso_lambda_iter_list_T7_u = [[torch.nn.Parameter(lasso_lambda_scalar * torch.ones(1, width, 1, 1) ) for width in hidden_layer_width_list] for i in range(ista_num_steps)]         
        self.lasso_lambda_iter_list_T7_u =  [item for sublist in _lasso_lambda_iter_list_T7_u for item in sublist]

        with torch.no_grad():
            L_T8_u = self.power_iteration_conv_model(self.encoder_dictionary_c, num_simulations = 20)
        self.ista_stepsize_iter_list_T8_u = [nn.Parameter(torch.ones(1)/L_T8_u) for i in range(ista_num_steps)]
        _lasso_lambda_iter_list_T8_u = [[torch.nn.Parameter(lasso_lambda_scalar * torch.ones(1, width, 1, 1) ) for width in hidden_layer_width_list] for i in range(ista_num_steps)]         
        self.lasso_lambda_iter_list_T8_u =  [item for sublist in _lasso_lambda_iter_list_T8_u for item in sublist]

        with torch.no_grad():
            L_T9_u = self.power_iteration_conv_model(self.encoder_dictionary_c, num_simulations = 20)
        self.ista_stepsize_iter_list_T9_u = [nn.Parameter(torch.ones(1)/L_T9_u) for i in range(ista_num_steps)]
        _lasso_lambda_iter_list_T9_u = [[torch.nn.Parameter(lasso_lambda_scalar * torch.ones(1, width, 1, 1) ) for width in hidden_layer_width_list] for i in range(ista_num_steps)]         
        self.lasso_lambda_iter_list_T9_u =  [item for sublist in _lasso_lambda_iter_list_T9_u for item in sublist]

        with torch.no_grad():
            L_T10_u = self.power_iteration_conv_model(self.encoder_dictionary_c, num_simulations = 20)
        self.ista_stepsize_iter_list_T10_u = [nn.Parameter(torch.ones(1)/L_T10_u) for i in range(ista_num_steps)]
        _lasso_lambda_iter_list_T10_u = [[torch.nn.Parameter(lasso_lambda_scalar * torch.ones(1, width, 1, 1) ) for width in hidden_layer_width_list] for i in range(ista_num_steps)]         
        self.lasso_lambda_iter_list_T10_u =  [item for sublist in _lasso_lambda_iter_list_T10_u for item in sublist]

        with torch.no_grad():
            L_T11_u = self.power_iteration_conv_model(self.encoder_dictionary_c, num_simulations = 20)
        self.ista_stepsize_iter_list_T11_u = [nn.Parameter(torch.ones(1)/L_T11_u) for i in range(ista_num_steps)]
        _lasso_lambda_iter_list_T11_u = [[torch.nn.Parameter(lasso_lambda_scalar * torch.ones(1, width, 1, 1) ) for width in hidden_layer_width_list] for i in range(ista_num_steps)]         
        self.lasso_lambda_iter_list_T11_u =  [item for sublist in _lasso_lambda_iter_list_T11_u for item in sublist]

        with torch.no_grad():
            L_T12_u = self.power_iteration_conv_model(self.encoder_dictionary_c, num_simulations = 20)
        self.ista_stepsize_iter_list_T12_u = [nn.Parameter(torch.ones(1)/L_T12_u) for i in range(ista_num_steps)]
        _lasso_lambda_iter_list_T12_u = [[torch.nn.Parameter(lasso_lambda_scalar * torch.ones(1, width, 1, 1) ) for width in hidden_layer_width_list] for i in range(ista_num_steps)]         
        self.lasso_lambda_iter_list_T12_u =  [item for sublist in _lasso_lambda_iter_list_T12_u for item in sublist]
        
        with torch.no_grad():
            L_c = self.power_iteration_conv_model(self.encoder_dictionary_c, num_simulations = 20)
        self.ista_stepsize_iter_list_c = [nn.Parameter(torch.ones(1)/L_c) for i in range(ista_num_steps)]
        _lasso_lambda_iter_list_c = [[torch.nn.Parameter(lasso_lambda_scalar * torch.ones(1, width, 1, 1) ) for width in hidden_layer_width_list] for i in range(ista_num_steps)]         
        self.lasso_lambda_iter_list_c =  [item for sublist in _lasso_lambda_iter_list_c for item in sublist]
    
        # v3 RG

        # self.resnet_T1_u = nn.ModuleList([ResidualGroup(conv=common.default_conv, n_feat=self.hidden_layer_width_list[0], kernel_size=3, n_resblocks=1), \
        #                                   ResidualGroup(conv=common.default_conv, n_feat=self.hidden_layer_width_list[1], kernel_size=3, n_resblocks=3), \
        #                                   ResidualGroup(conv=common.default_conv, n_feat=self.hidden_layer_width_list[2], kernel_size=3, n_resblocks=5)])

        self.resnet_u = nn.ModuleList([ResidualGroup(conv=common.default_conv, n_feat=self.hidden_layer_width_list[0], prompt_dim=self.prompt_dim[0], 
                                                       prompt_len=self.prompt_len[0], prompt_size=self.prompt_size[0], kernel_size=3, n_resblocks=1), \
                                         ResidualGroup(conv=common.default_conv, n_feat=self.hidden_layer_width_list[1], prompt_dim=self.prompt_dim[1], 
                                                       prompt_len=self.prompt_len[1], prompt_size=self.prompt_size[1], kernel_size=3, n_resblocks=3), \
                                         ResidualGroup(conv=common.default_conv, n_feat=self.hidden_layer_width_list[2], prompt_dim=self.prompt_dim[2], 
                                                       prompt_len=self.prompt_len[2], prompt_size=self.prompt_size[2], kernel_size=3, n_resblocks=5)])

        self.resnet_c = nn.ModuleList([ResidualGroup_c(conv=common.default_conv, n_feat=self.hidden_layer_width_list[0], kernel_size=3, n_resblocks=1), \
                                       ResidualGroup_c(conv=common.default_conv, n_feat=self.hidden_layer_width_list[1], kernel_size=3, n_resblocks=3), \
                                       ResidualGroup_c(conv=common.default_conv, n_feat=self.hidden_layer_width_list[2], kernel_size=3, n_resblocks=5)])


        self.layer_in_T1 = nn.Conv2d(2, n_classes, 3, 1, 1); self.layer_in_T1_res = nn.Conv2d(2, n_classes, 3, 1, 1)
        self.layer_in_T2 = nn.Conv2d(2, n_classes, 3, 1, 1); self.layer_in_T2_res = nn.Conv2d(2, n_classes, 3, 1, 1)
        self.layer_in_T3 = nn.Conv2d(2, n_classes, 3, 1, 1); self.layer_in_T3_res = nn.Conv2d(2, n_classes, 3, 1, 1)
        self.layer_in_T4 = nn.Conv2d(2, n_classes, 3, 1, 1); self.layer_in_T4_res = nn.Conv2d(2, n_classes, 3, 1, 1)
        self.layer_in_T5 = nn.Conv2d(2, n_classes, 3, 1, 1); self.layer_in_T5_res = nn.Conv2d(2, n_classes, 3, 1, 1)
        self.layer_in_T6 = nn.Conv2d(2, n_classes, 3, 1, 1); self.layer_in_T6_res = nn.Conv2d(2, n_classes, 3, 1, 1)
        self.layer_in_T7 = nn.Conv2d(2, n_classes, 3, 1, 1); self.layer_in_T7_res = nn.Conv2d(2, n_classes, 3, 1, 1)
        self.layer_in_T8 = nn.Conv2d(2, n_classes, 3, 1, 1); self.layer_in_T8_res = nn.Conv2d(2, n_classes, 3, 1, 1)
        self.layer_in_T9 = nn.Conv2d(2, n_classes, 3, 1, 1); self.layer_in_T9_res = nn.Conv2d(2, n_classes, 3, 1, 1)
        self.layer_in_T10 = nn.Conv2d(2, n_classes, 3, 1, 1); self.layer_in_T10_res = nn.Conv2d(2, n_classes, 3, 1, 1)
        self.layer_in_T11 = nn.Conv2d(2, n_classes, 3, 1, 1); self.layer_in_T11_res = nn.Conv2d(2, n_classes, 3, 1, 1)
        self.layer_in_T12 = nn.Conv2d(2, n_classes, 3, 1, 1); self.layer_in_T12_res = nn.Conv2d(2, n_classes, 3, 1, 1)
        
        
        self.layer_in_c = nn.Conv2d(24, n_classes, 3, 1, 1)
        self.relu = nn.ReLU()
        
        self.compress_c = nn.Conv2d(n_classes*12, n_classes, 3, 1, 1)

        self.rec_T1_u = nn.Conv2d(n_classes, 2, 3, 1, 1)
        self.rec_T2_u = nn.Conv2d(n_classes, 2, 3, 1, 1)
        self.rec_T3_u = nn.Conv2d(n_classes, 2, 3, 1, 1)
        self.rec_T4_u = nn.Conv2d(n_classes, 2, 3, 1, 1)
        self.rec_T5_u = nn.Conv2d(n_classes, 2, 3, 1, 1)
        self.rec_T6_u = nn.Conv2d(n_classes, 2, 3, 1, 1)
        self.rec_T7_u = nn.Conv2d(n_classes, 2, 3, 1, 1)
        self.rec_T8_u = nn.Conv2d(n_classes, 2, 3, 1, 1)
        self.rec_T9_u = nn.Conv2d(n_classes, 2, 3, 1, 1)
        self.rec_T10_u = nn.Conv2d(n_classes, 2, 3, 1, 1)
        self.rec_T11_u = nn.Conv2d(n_classes, 2, 3, 1, 1)
        self.rec_T12_u = nn.Conv2d(n_classes, 2, 3, 1, 1)
        self.rec_c = nn.Conv2d(n_classes, 24, 3, 1, 1) # 6 -> 12
        
        self.HFF_fuse_T1_u = nn.Conv2d(n_classes*(ista_num_steps-1), n_classes, 1, 1)
        self.HFF_fuse_T2_u = nn.Conv2d(n_classes*(ista_num_steps-1), n_classes, 1, 1)
        self.HFF_fuse_T3_u = nn.Conv2d(n_classes*(ista_num_steps-1), n_classes, 1, 1)
        self.HFF_fuse_T4_u = nn.Conv2d(n_classes*(ista_num_steps-1), n_classes, 1, 1)
        self.HFF_fuse_T5_u = nn.Conv2d(n_classes*(ista_num_steps-1), n_classes, 1, 1)
        self.HFF_fuse_T6_u = nn.Conv2d(n_classes*(ista_num_steps-1), n_classes, 1, 1)
        self.HFF_fuse_T7_u = nn.Conv2d(n_classes*(ista_num_steps-1), n_classes, 1, 1)
        self.HFF_fuse_T8_u = nn.Conv2d(n_classes*(ista_num_steps-1), n_classes, 1, 1)
        self.HFF_fuse_T9_u = nn.Conv2d(n_classes*(ista_num_steps-1), n_classes, 1, 1)
        self.HFF_fuse_T10_u = nn.Conv2d(n_classes*(ista_num_steps-1), n_classes, 1, 1)
        self.HFF_fuse_T11_u = nn.Conv2d(n_classes*(ista_num_steps-1), n_classes, 1, 1)
        self.HFF_fuse_T12_u = nn.Conv2d(n_classes*(ista_num_steps-1), n_classes, 1, 1)

    def init_feature(self, T1_img, T2_img, T3_img, T4_img, T5_img, T6_img,T7_img, T8_img, T9_img, T10_img, T11_img, T12_img):
        # cat F & T1 & T2 img, gen common_img
        common_img = torch.cat([ T1_img, T2_img, T3_img, T4_img, T5_img, T6_img,
                                T7_img, T8_img, T9_img, T10_img, T11_img, T12_img], 1)
        # cal u_in & c_in
        T1_in = self.layer_in_T1(T1_img)
        T2_in = self.layer_in_T2(T2_img)
        T3_in = self.layer_in_T3(T3_img)
        T4_in = self.layer_in_T4(T4_img)
        T5_in = self.layer_in_T5(T5_img)
        T6_in = self.layer_in_T6(T6_img)
        T7_in = self.layer_in_T7(T7_img)
        T8_in = self.layer_in_T8(T8_img)
        T9_in = self.layer_in_T9(T9_img)
        T10_in = self.layer_in_T10(T10_img)
        T11_in = self.layer_in_T11(T11_img)
        T12_in = self.layer_in_T12(T12_img)
        c_in = self.layer_in_c(common_img)
        # init F & T1 & T2 list
        adj_err_list_T1_u  = self.adjoint_encoder_dictionary_u(T1_in,code(1))
        adj_err_list_T2_u  = self.adjoint_encoder_dictionary_u(T2_in,code(2))
        adj_err_list_T3_u  = self.adjoint_encoder_dictionary_u(T3_in,code(3))
        adj_err_list_T4_u  = self.adjoint_encoder_dictionary_u(T4_in,code(4))
        adj_err_list_T5_u  = self.adjoint_encoder_dictionary_u(T5_in,code(5))
        adj_err_list_T6_u  = self.adjoint_encoder_dictionary_u(T6_in,code(6))
        adj_err_list_T7_u  = self.adjoint_encoder_dictionary_u(T7_in,code(7))
        adj_err_list_T8_u  = self.adjoint_encoder_dictionary_u(T8_in,code(8))
        adj_err_list_T9_u  = self.adjoint_encoder_dictionary_u(T9_in,code(9))
        adj_err_list_T10_u  = self.adjoint_encoder_dictionary_u(T10_in,code(10))
        adj_err_list_T11_u  = self.adjoint_encoder_dictionary_u(T11_in,code(11))
        adj_err_list_T12_u  = self.adjoint_encoder_dictionary_u(T12_in,code(12))
        adj_err_list_c  = self.adjoint_encoder_dictionary_c(c_in)

        T1_u_list = []
        T2_u_list = []
        T3_u_list = []
        T4_u_list = []
        T5_u_list = []
        T6_u_list = []
        T7_u_list = []
        T8_u_list = []
        T9_u_list = []
        T10_u_list = []
        T11_u_list = []
        T12_u_list = []
        c_list = []
        for i in range(self.num_layers):
            lambd_T1_u = self.ista_stepsize_iter_list_T1_u[0] * self.lasso_lambda_iter_list_T1_u[i]
            T1_u_list_i = relu(self.ista_stepsize_iter_list_T1_u[0].to(T1_img.device) * adj_err_list_T1_u[i], lambd = lambd_T1_u.to(T1_img.device))
            T1_u_list.append(T1_u_list_i)
            lambd_T2_u = self.ista_stepsize_iter_list_T2_u[0] * self.lasso_lambda_iter_list_T2_u[i]
            T2_u_list_i = relu(self.ista_stepsize_iter_list_T2_u[0].to(T2_img.device) * adj_err_list_T2_u[i], lambd = lambd_T2_u.to(T2_img.device))
            T2_u_list.append(T2_u_list_i)
            lambd_T3_u = self.ista_stepsize_iter_list_T3_u[0] * self.lasso_lambda_iter_list_T3_u[i]
            T3_u_list_i = relu(self.ista_stepsize_iter_list_T3_u[0].to(T3_img.device) * adj_err_list_T3_u[i], lambd=lambd_T3_u.to(T3_img.device))
            T3_u_list.append(T3_u_list_i)
            lambd_T4_u = self.ista_stepsize_iter_list_T4_u[0] * self.lasso_lambda_iter_list_T4_u[i]
            T4_u_list_i = relu(self.ista_stepsize_iter_list_T4_u[0].to(T4_img.device) * adj_err_list_T4_u[i], lambd=lambd_T4_u.to(T4_img.device))
            T4_u_list.append(T4_u_list_i)
            lambd_T5_u = self.ista_stepsize_iter_list_T5_u[0] * self.lasso_lambda_iter_list_T5_u[i]
            T5_u_list_i = relu(self.ista_stepsize_iter_list_T5_u[0].to(T5_img.device) * adj_err_list_T5_u[i], lambd=lambd_T5_u.to(T5_img.device))
            T5_u_list.append(T5_u_list_i)
            lambd_T6_u = self.ista_stepsize_iter_list_T6_u[0] * self.lasso_lambda_iter_list_T6_u[i]
            T6_u_list_i = relu(self.ista_stepsize_iter_list_T6_u[0].to(T6_img.device) * adj_err_list_T6_u[i], lambd=lambd_T6_u.to(T6_img.device))
            T6_u_list.append(T6_u_list_i)
            # T7
            lambd_T7_u = self.ista_stepsize_iter_list_T7_u[0] * self.lasso_lambda_iter_list_T7_u[i]
            T7_u_list_i = relu(self.ista_stepsize_iter_list_T7_u[0].to(T7_img.device) * adj_err_list_T7_u[i], lambd=lambd_T7_u.to(T7_img.device))
            T7_u_list.append(T7_u_list_i)
            
            # T8
            lambd_T8_u = self.ista_stepsize_iter_list_T8_u[0] * self.lasso_lambda_iter_list_T8_u[i]
            T8_u_list_i = relu(self.ista_stepsize_iter_list_T8_u[0].to(T8_img.device) * adj_err_list_T8_u[i], lambd=lambd_T8_u.to(T8_img.device))
            T8_u_list.append(T8_u_list_i)
            
            # T9
            lambd_T9_u = self.ista_stepsize_iter_list_T9_u[0] * self.lasso_lambda_iter_list_T9_u[i]
            T9_u_list_i = relu(self.ista_stepsize_iter_list_T9_u[0].to(T9_img.device) * adj_err_list_T9_u[i], lambd=lambd_T9_u.to(T9_img.device))
            T9_u_list.append(T9_u_list_i)
            
            # T10
            lambd_T10_u = self.ista_stepsize_iter_list_T10_u[0] * self.lasso_lambda_iter_list_T10_u[i]
            T10_u_list_i = relu(self.ista_stepsize_iter_list_T10_u[0].to(T10_img.device) * adj_err_list_T10_u[i], lambd=lambd_T10_u.to(T10_img.device))
            T10_u_list.append(T10_u_list_i)
            
            # T11
            lambd_T11_u = self.ista_stepsize_iter_list_T11_u[0] * self.lasso_lambda_iter_list_T11_u[i]
            T11_u_list_i = relu(self.ista_stepsize_iter_list_T11_u[0].to(T11_img.device) * adj_err_list_T11_u[i], lambd=lambd_T11_u.to(T11_img.device))
            T11_u_list.append(T11_u_list_i)
            
            # T12
            lambd_T12_u = self.ista_stepsize_iter_list_T12_u[0] * self.lasso_lambda_iter_list_T12_u[i]
            T12_u_list_i = relu(self.ista_stepsize_iter_list_T12_u[0].to(T12_img.device) * adj_err_list_T12_u[i], lambd=lambd_T12_u.to(T12_img.device))
            T12_u_list.append(T12_u_list_i)
            
            
            
            lambd_c = self.ista_stepsize_iter_list_c[0] * self.lasso_lambda_iter_list_c[i]
            c_list_i = relu(self.ista_stepsize_iter_list_c[0].to(common_img.device) * adj_err_list_c[i], lambd = lambd_c.to(common_img.device))
            c_list.append(c_list_i)
        # save F_img*F_D & T1_img*T1_D & T2_img*T2_D

        T1_D_list = adj_err_list_T1_u
        T2_D_list = adj_err_list_T2_u
        T3_D_list = adj_err_list_T3_u
        T4_D_list = adj_err_list_T4_u
        T5_D_list = adj_err_list_T5_u
        T6_D_list = adj_err_list_T6_u
        T7_D_list = adj_err_list_T7_u
        T8_D_list = adj_err_list_T8_u
        T9_D_list = adj_err_list_T9_u
        T10_D_list = adj_err_list_T10_u
        T11_D_list = adj_err_list_T11_u
        T12_D_list = adj_err_list_T12_u

        return T1_in, T2_in,T3_in, T4_in,T5_in, T6_in,T7_in, T8_in,T9_in, T10_in,T11_in, T12_in,\
               T1_u_list, T2_u_list,T3_u_list, T4_u_list,T5_u_list, T6_u_list,T7_u_list, T8_u_list,T9_u_list, T10_u_list,T11_u_list, T12_u_list, c_list,\
               T1_D_list, T2_D_list,T3_D_list, T4_D_list,T5_D_list, T6_D_list,T7_D_list, T8_D_list,T9_D_list, T10_D_list,T11_D_list, T12_D_list

    def update_u_list(self, idx, contrast, D_list, c_list, u_list):
        if contrast == 'T1':
            err = self.encoder_dictionary_u(u_list,code(1)) + self.encoder_dictionary_c2(c_list,code(1))
            adj_err_list = self.adjoint_encoder_dictionary_u(err,code(1))
            ista_stepsize = self.ista_stepsize_iter_list_T1_u[idx]
            for i in range(self.num_layers):
                u_list[i] = u_list[i] - ista_stepsize.to(u_list[i].device) * (adj_err_list[i] - D_list[i])
                u_list[i] = self.resnet_u[i](self.relu(u_list[i]))
        elif contrast == 'T2':
            err = self.encoder_dictionary_u(u_list,code(2)) + self.encoder_dictionary_c2(c_list,code(2))
            adj_err_list = self.adjoint_encoder_dictionary_u(err,code(2))
            ista_stepsize = self.ista_stepsize_iter_list_T2_u[idx]
            for i in range(self.num_layers):
                u_list[i] = u_list[i] - ista_stepsize.to(u_list[i].device) * (adj_err_list[i] - D_list[i])
                u_list[i] = self.resnet_u[i](self.relu(u_list[i]))
        elif contrast == 'T3':
            err = self.encoder_dictionary_u(u_list,code(3)) + self.encoder_dictionary_c2(c_list,code(3))
            adj_err_list = self.adjoint_encoder_dictionary_u(err,code(3))
            ista_stepsize = self.ista_stepsize_iter_list_T3_u[idx]
            for i in range(self.num_layers):
                u_list[i] = u_list[i] - ista_stepsize.to(u_list[i].device) * (adj_err_list[i] - D_list[i])
                u_list[i] = self.resnet_u[i](self.relu(u_list[i]))
        elif contrast == 'T4':
            err = self.encoder_dictionary_u(u_list,code(4)) + self.encoder_dictionary_c2(c_list,code(4))
            adj_err_list = self.adjoint_encoder_dictionary_u(err,code(4))
            ista_stepsize = self.ista_stepsize_iter_list_T4_u[idx]
            for i in range(self.num_layers):
                u_list[i] = u_list[i] - ista_stepsize.to(u_list[i].device) * (adj_err_list[i] - D_list[i])
                u_list[i] = self.resnet_u[i](self.relu(u_list[i]))
        elif contrast == 'T5':
            err = self.encoder_dictionary_u(u_list,code(5)) + self.encoder_dictionary_c2(c_list,code(5))
            adj_err_list = self.adjoint_encoder_dictionary_u(err,code(5))
            ista_stepsize = self.ista_stepsize_iter_list_T5_u[idx]
            for i in range(self.num_layers):
                u_list[i] = u_list[i] - ista_stepsize.to(u_list[i].device) * (adj_err_list[i] - D_list[i])
                u_list[i] = self.resnet_u[i](self.relu(u_list[i]))
        elif contrast == 'T6':
            err = self.encoder_dictionary_u(u_list,code(6)) + self.encoder_dictionary_c2(c_list,code(6))
            adj_err_list = self.adjoint_encoder_dictionary_u(err,code(6))
            ista_stepsize = self.ista_stepsize_iter_list_T6_u[idx]
            for i in range(self.num_layers):
                u_list[i] = u_list[i] - ista_stepsize.to(u_list[i].device) * (adj_err_list[i] - D_list[i])
                u_list[i] = self.resnet_u[i](self.relu(u_list[i]))
        elif contrast == 'T7':
            err = self.encoder_dictionary_u(u_list,code(7)) + self.encoder_dictionary_c2(c_list,code(7))
            adj_err_list = self.adjoint_encoder_dictionary_u(err,code(7))
            ista_stepsize = self.ista_stepsize_iter_list_T7_u[idx]
            for i in range(self.num_layers):
                u_list[i] = u_list[i] - ista_stepsize.to(u_list[i].device) * (adj_err_list[i] - D_list[i])
                u_list[i] = self.resnet_u[i](self.relu(u_list[i]))
        elif contrast == 'T8':
            err = self.encoder_dictionary_u(u_list,code(8)) + self.encoder_dictionary_c2(c_list,code(8))
            adj_err_list = self.adjoint_encoder_dictionary_u(err,code(8))
            ista_stepsize = self.ista_stepsize_iter_list_T8_u[idx]
            for i in range(self.num_layers):
                u_list[i] = u_list[i] - ista_stepsize.to(u_list[i].device) * (adj_err_list[i] - D_list[i])
                u_list[i] = self.resnet_u[i](self.relu(u_list[i]))
        elif contrast == 'T9':
            err = self.encoder_dictionary_u(u_list,code(9)) + self.encoder_dictionary_c2(c_list,code(9))
            adj_err_list = self.adjoint_encoder_dictionary_u(err,code(9))
            ista_stepsize = self.ista_stepsize_iter_list_T9_u[idx]
            for i in range(self.num_layers):
                u_list[i] = u_list[i] - ista_stepsize.to(u_list[i].device) * (adj_err_list[i] - D_list[i])
                u_list[i] = self.resnet_u[i](self.relu(u_list[i]))
        elif contrast == 'T10':
            err = self.encoder_dictionary_u(u_list,code(10)) + self.encoder_dictionary_c2(c_list,code(10))
            adj_err_list = self.adjoint_encoder_dictionary_u(err,code(10))
            ista_stepsize = self.ista_stepsize_iter_list_T10_u[idx]
            for i in range(self.num_layers):
                u_list[i] = u_list[i] - ista_stepsize.to(u_list[i].device) * (adj_err_list[i] - D_list[i])
                u_list[i] = self.resnet_u[i](self.relu(u_list[i]))
        elif contrast == 'T11':
            err = self.encoder_dictionary_u(u_list,code(11)) + self.encoder_dictionary_c2(c_list,code(11))
            adj_err_list = self.adjoint_encoder_dictionary_u(err,code(11))
            ista_stepsize = self.ista_stepsize_iter_list_T11_u[idx]
            for i in range(self.num_layers):
                u_list[i] = u_list[i] - ista_stepsize.to(u_list[i].device) * (adj_err_list[i] - D_list[i])
                u_list[i] = self.resnet_u[i](self.relu(u_list[i]))
        elif contrast == 'T12':
            err = self.encoder_dictionary_u(u_list,code(12)) + self.encoder_dictionary_c2(c_list,code(12))
            adj_err_list = self.adjoint_encoder_dictionary_u(err,code(12))
            ista_stepsize = self.ista_stepsize_iter_list_T12_u[idx]
            for i in range(self.num_layers):
                u_list[i] = u_list[i] - ista_stepsize.to(u_list[i].device) * (adj_err_list[i] - D_list[i])
                u_list[i] = self.resnet_u[i](self.relu(u_list[i]))        
        
        return u_list
    
    def update_u_list_new(self, idx, contrast, Q_list, c_list, u_list):

        if contrast == 'T1':
            err = self.decoder_dictionary_u(u_list,code(1)) + self.decoder_dictionary_c2(c_list,code(1))
            adj_err_list = self.adjoint_decoder_dictionary_u(err,code(1))
            ista_stepsize = self.ista_stepsize_iter_list_T1_u[idx]
            for i in range(self.num_layers):
                u_list[i] = u_list[i] - ista_stepsize.to(u_list[i].device) * (adj_err_list[i] - Q_list[i])
                u_list[i] = self.resnet_u[i](self.relu(u_list[i]))
        elif contrast == 'T2':
            err = self.decoder_dictionary_u(u_list,code(2)) + self.decoder_dictionary_c2(c_list,code(2))
            adj_err_list = self.adjoint_decoder_dictionary_u(err,code(2))
            ista_stepsize = self.ista_stepsize_iter_list_T2_u[idx]
            for i in range(self.num_layers):
                u_list[i] = u_list[i] - ista_stepsize.to(u_list[i].device) * (adj_err_list[i] - Q_list[i])
                u_list[i] = self.resnet_u[i](self.relu(u_list[i]))
        elif contrast == 'T3':
            err = self.decoder_dictionary_u(u_list,code(3)) + self.decoder_dictionary_c2(c_list,code(3))
            adj_err_list = self.adjoint_decoder_dictionary_u(err,code(3))
            ista_stepsize = self.ista_stepsize_iter_list_T3_u[idx]
            for i in range(self.num_layers):
                u_list[i] = u_list[i] - ista_stepsize.to(u_list[i].device) * (adj_err_list[i] - Q_list[i])
                u_list[i] = self.resnet_u[i](self.relu(u_list[i]))
        elif contrast == 'T4':
            err = self.decoder_dictionary_u(u_list,code(4)) + self.decoder_dictionary_c2(c_list,code(4))
            adj_err_list = self.adjoint_decoder_dictionary_u(err,code(4))
            ista_stepsize = self.ista_stepsize_iter_list_T4_u[idx]
            for i in range(self.num_layers):
                u_list[i] = u_list[i] - ista_stepsize.to(u_list[i].device) * (adj_err_list[i] - Q_list[i])
                u_list[i] = self.resnet_u[i](self.relu(u_list[i]))
        elif contrast == 'T5':
            err = self.decoder_dictionary_u(u_list,code(5)) + self.decoder_dictionary_c2(c_list,code(5))
            adj_err_list = self.adjoint_decoder_dictionary_u(err,code(5))
            ista_stepsize = self.ista_stepsize_iter_list_T5_u[idx]
            for i in range(self.num_layers):
                u_list[i] = u_list[i] - ista_stepsize.to(u_list[i].device) * (adj_err_list[i] - Q_list[i])
                u_list[i] = self.resnet_u[i](self.relu(u_list[i]))
        elif contrast == 'T6':
            err = self.decoder_dictionary_u(u_list,code(6)) + self.decoder_dictionary_c2(c_list,code(6))
            adj_err_list = self.adjoint_decoder_dictionary_u(err,code(6))
            ista_stepsize = self.ista_stepsize_iter_list_T6_u[idx]
            for i in range(self.num_layers):
                u_list[i] = u_list[i] - ista_stepsize.to(u_list[i].device) * (adj_err_list[i] - Q_list[i])
                u_list[i] = self.resnet_u[i](self.relu(u_list[i]))
        elif contrast == 'T7':
            err = self.decoder_dictionary_u(u_list,code(7)) + self.decoder_dictionary_c2(c_list,code(7))
            adj_err_list = self.adjoint_decoder_dictionary_u(err,code(7))
            ista_stepsize = self.ista_stepsize_iter_list_T7_u[idx]
            for i in range(self.num_layers):
                u_list[i] = u_list[i] - ista_stepsize.to(u_list[i].device) * (adj_err_list[i] - Q_list[i])
                u_list[i] = self.resnet_u[i](self.relu(u_list[i]))
        elif contrast == 'T8':
            err = self.decoder_dictionary_u(u_list,code(8)) + self.decoder_dictionary_c2(c_list,code(8))
            adj_err_list = self.adjoint_decoder_dictionary_u(err,code(8))
            ista_stepsize = self.ista_stepsize_iter_list_T8_u[idx]
            for i in range(self.num_layers):
                u_list[i] = u_list[i] - ista_stepsize.to(u_list[i].device) * (adj_err_list[i] - Q_list[i])
                u_list[i] = self.resnet_u[i](self.relu(u_list[i]))
        elif contrast == 'T9':
            err = self.decoder_dictionary_u(u_list,code(9)) + self.decoder_dictionary_c2(c_list,code(9))
            adj_err_list = self.adjoint_decoder_dictionary_u(err,code(9))
            ista_stepsize = self.ista_stepsize_iter_list_T9_u[idx]
            for i in range(self.num_layers):
                u_list[i] = u_list[i] - ista_stepsize.to(u_list[i].device) * (adj_err_list[i] - Q_list[i])
                u_list[i] = self.resnet_u[i](self.relu(u_list[i]))
        elif contrast == 'T10':
            err = self.decoder_dictionary_u(u_list,code(10)) + self.decoder_dictionary_c2(c_list,code(10))
            adj_err_list = self.adjoint_decoder_dictionary_u(err,code(10))
            ista_stepsize = self.ista_stepsize_iter_list_T10_u[idx]
            for i in range(self.num_layers):
                u_list[i] = u_list[i] - ista_stepsize.to(u_list[i].device) * (adj_err_list[i] - Q_list[i])
                u_list[i] = self.resnet_u[i](self.relu(u_list[i]))
        elif contrast == 'T11':
            err = self.decoder_dictionary_u(u_list,code(11))+ self.decoder_dictionary_c2(c_list,code(11))
            adj_err_list = self.adjoint_decoder_dictionary_u(err,code(11))
            ista_stepsize = self.ista_stepsize_iter_list_T11_u[idx]
            for i in range(self.num_layers):
                u_list[i] = u_list[i] - ista_stepsize.to(u_list[i].device) * (adj_err_list[i] - Q_list[i])
                u_list[i] = self.resnet_u[i](self.relu(u_list[i]))
        elif contrast == 'T12':
            err = self.decoder_dictionary_u(u_list,code(12)) + self.decoder_dictionary_c2(c_list,code(12))
            adj_err_list = self.adjoint_decoder_dictionary_u(err,code(12))
            ista_stepsize = self.ista_stepsize_iter_list_T12_u[idx]
            for i in range(self.num_layers):
                u_list[i] = u_list[i] - ista_stepsize.to(u_list[i].device) * (adj_err_list[i] - Q_list[i])
                u_list[i] = self.resnet_u[i](self.relu(u_list[i]))
        return u_list
    
    def update_c_list(self, idx, Dc_res, c_list):
        err_c = self.encoder_dictionary_c(c_list) - Dc_res
        adj_err_list_c  = self.adjoint_encoder_dictionary_c(err_c)
        ista_stepsize_c = self.ista_stepsize_iter_list_c[idx]
        for i in range(self.num_layers):
            c_list[i] = c_list[i] - ista_stepsize_c.to(c_list[i].device) * adj_err_list_c[i]
            c_list[i] = self.resnet_c[i](self.relu(c_list[i]))

        return c_list
    
    def update_c_list_new(self, idx, Qc_res, c_list):
        err_c = self.decoder_dictionary_c(c_list) - Qc_res
        adj_err_list_c  = self.adjoint_decoder_dictionary_c(err_c)
        ista_stepsize_c = self.ista_stepsize_iter_list_c[idx]
        for i in range(self.num_layers):
            c_list[i] = c_list[i] - ista_stepsize_c.to(c_list[i].device) * adj_err_list_c[i]
            c_list[i] = self.resnet_c[i](self.relu(c_list[i]))

        return c_list
    
    # def DCM(self, rho, img_ori, k_lq, sense, mask,z_img):
    #     bs, w, h, _ = img_ori.shape
    #     z_img = z_img.permute(0,2,3,1)
    #     img_ori = img_ori.reshape(bs, 1, w, h, _)
    #     z_img = z_img.reshape(bs, 1, w, h, _)
    #     if rho == -1:
    #         img_coil_ori = fastmri.complex_mul(sense, img_ori)
    #         k_rec = mask*k_lq + (1-mask)*fastmri.fft2c(img_coil_ori)
    #     elif rho > 0:
    #         img_coil_ori = fastmri.complex_mul(sense, img_ori)
    #         z_coil_ori = fastmri.complex_mul(sense, z_img)
    #         k_rec = (mask*k_lq + self.lambda_1 * fastmri.fft2c(img_coil_ori) + self.beta_1 * fastmri.fft2c(z_coil_ori) )/(mask + self.lambda_1 + self.beta_1)
    #     img_coil_rec = fastmri.ifft2c(k_rec) # type: ignore
    #     img_rec = fastmri.complex_mul(img_coil_rec, fastmri.complex_conj(sense)).sum(dim=1, keepdim=False)

    #     return img_rec
    
    def DCM(self, rho, img_ori, k_lq, sense, mask):
        bs, w, h, _ = img_ori.shape
        img_ori = img_ori.reshape(bs, 1, w, h, _)
        if rho == -1:
            img_coil_ori = fastmri.complex_mul(sense, img_ori)
            k_rec = mask*k_lq + (1-mask)*fastmri.fft2c(img_coil_ori)
        elif rho > 0:
            img_coil_ori = fastmri.complex_mul(sense, img_ori)
            k_rec = (mask*k_lq + rho*fastmri.fft2c(img_coil_ori))/(mask + rho)
        img_coil_rec = fastmri.ifft2c(k_rec)
        img_rec = fastmri.complex_mul(img_coil_rec, fastmri.complex_conj(sense)).sum(dim=1, keepdim=False)

        return img_rec
    
    def split_by_frequency(self, rate, img_rec, img_lq):
        if img_lq is not None:
            img_res = img_rec-img_lq
        else:
            img_res = img_rec
        bs, w, h, _ = img_res.shape
        lf_len = int(math.sqrt(w * h * rate))
        assert (lf_len <= w and lf_len <= h)
        mask_lf = torch.zeros(1, w, h, 1)
        mask_lf[0, int((w-lf_len)/2):int((w+lf_len)/2), int((h-lf_len)/2):int((h+lf_len)/2), 0] = 1

        img_lf = fastmri.ifft2c(mask_lf*fastmri.fft2c(img_res)) # fft2c or fft2c+sense
        img_hf = fastmri.ifft2c((1-mask_lf)*fastmri.fft2c(img_res))

        return img_lf, img_hf

    def update_Z(self,img,Z,mask_img,idx):
        if not Z.requires_grad:
            Z.requires_grad = True
        img = img.permute(0,1,4,2,3)
        # var1, var2, var3, var4, var5, var6,var7, var8, var9, var10, var11, var12 = [chunk.squeeze(1) for chunk in torch.chunk(Z, chunks=12, dim=1)]  
        Z = Z.permute(0,1,4,2,3)
        R1,R2,S = mdi.R_S(Z)
        
        R1 = R1 * mask_img
        R2 = R2 * mask_img
        S = S * mask_img 
        # R1 = norm_r1(R1)
        R1 = torch.stack((R1.real,R1.imag),dim=1)
        R2 = torch.stack((R2.real,R2.imag),dim=1)
        S = torch.stack((S.real,S.imag),dim=1)
        R1_map = self.r1[idx](R1) + R1
        S_map = self.s[idx](S) + S
        R2_map = self.r2[idx](R2) + R2
        # R1_map,R2_map,S_map  = self.RSZ_net(R1,R2,S)
        
        L_R1 = torch.sum((R1-R1_map)**2)
        L_R2 = torch.sum((R2-R2_map)**2)
        L_S = torch.sum((S-S_map)**2)
        
        grad_1 = torch.autograd.grad(outputs=L_R1, inputs=Z,  # type: ignore
                                    grad_outputs=torch.ones_like(L_R1), retain_graph=True)[0] # type: ignore
        grad_2 = torch.autograd.grad(outputs=L_R2, inputs=Z,  # type: ignore
                                    grad_outputs=torch.ones_like(L_R2), retain_graph=True)[0] # type: ignore
        grad_3 = torch.autograd.grad(outputs=L_S, inputs=Z, 
                                    grad_outputs=torch.ones_like(L_S), retain_graph=True)[0]
        
        grad_1 = torch.nan_to_num(grad_1, nan=0.0)
        grad_2 = torch.nan_to_num(grad_2, nan=0.0)
        grad_3 = torch.nan_to_num(grad_3, nan=0.0)
        grads = self.mu1 * grad_1+ self.mu2 * grad_2 + self.mu3 * grad_3 # + self.mu4 * (Z-img)
        Z = Z - grads
        
        return Z,R1_map,R2_map,S_map
    def update_img(self,img,R1_map,R2_map,S_map):
        T1_img,T2_img,T3_img,T4_img,T5_img,T6_img,T7_img,T8_img,T9_img,T10_img,T11_img,T12_img\
            = [chunk.squeeze(1) for chunk in torch.chunk(img, chunks=12, dim=1)]
        T1_img =  (1 - self.mu2)* T1_img + self.mu2  * T7_img * R1_map
        T2_img =  (1 - self.mu2)* T2_img + self.mu2  * T8_img * R2_map
        
        T3_img = (1 - self.mu1 - self.mu2) * T3_img + self.mu1 * T2_img * S_map + self.mu2 * T9_img * R2_map
        T4_img = (1 - self.mu1 - self.mu2) * T4_img + self.mu1 * T3_img * S_map + self.mu2 * T10_img * R2_map
        T5_img = (1 - self.mu1 - self.mu2) * T5_img + self.mu1 * T4_img * S_map + self.mu2 * T11_img * R2_map
        T6_img = (1 - self.mu1 - self.mu2) * T6_img + self.mu1 * T5_img * S_map + self.mu2 * T12_img * R2_map
         
        T9_img  = (1 - self.mu1 ) * T9_img  + self.mu1 * T8_img *  S_map 
        T10_img = (1 - self.mu1 ) * T10_img + self.mu1 * T9_img  * S_map 
        T11_img = (1 - self.mu1 ) * T11_img + self.mu1 * T10_img * S_map
        T12_img = (1 - self.mu1 ) * T12_img + self.mu1 * T11_img * S_map
        img = torch.stack((T1_img, T2_img,T3_img, T4_img,T5_img, T6_img,\
            T7_img, T8_img,T9_img, T10_img,T11_img, T12_img),dim=1) 
        return img
        
        
        
    def update_Z_2(self,img,mask_img,idx):
        img = img.permute(0,1,4,2,3)
        R1,R2,S = mdi.R_S(img)
        
        R1 = R1 * mask_img
        R2 = R2 * mask_img
        S = S * mask_img 
        # R1 = norm_r1(R1)
        
        R1 = torch.stack((R1.real,R1.imag),dim=1)
        R2 = torch.stack((R2.real,R2.imag),dim=1)
        S = torch.stack((S.real,S.imag),dim=1)
        R1_map = self.r1[idx](R1) + R1
        S_map = self.s[idx](S) + S
        R2_map = self.r2[idx](R2) + R2
        img = self.update_img(img,R1_map,R2_map,S_map)
        return img
    def forward(self,
                    T1_img_lq, T1_k_lq, T1_sense, T1_mask, 
                    T2_img_lq, T2_k_lq, T2_sense, T2_mask,
                    T3_img_lq, T3_k_lq, T3_sense, T3_mask, 
                    T4_img_lq, T4_k_lq, T4_sense, T4_mask,
                    T5_img_lq, T5_k_lq, T5_sense, T5_mask, 
                    T6_img_lq, T6_k_lq, T6_sense, T6_mask,
                    T7_img_lq, T7_k_lq, T7_sense, T7_mask, 
                    T8_img_lq, T8_k_lq, T8_sense, T8_mask,
                    T9_img_lq, T9_k_lq, T9_sense, T9_mask, 
                    T10_img_lq, T10_k_lq, T10_sense, T10_mask,
                    T11_img_lq, T11_k_lq, T11_sense, T11_mask, 
                    T12_img_lq, T12_k_lq, T12_sense, T12_mask,
                    mask_img
                    ):
        HFF_T1_u = []
        HFF_T2_u = []
        HFF_T3_u = []
        HFF_T4_u = []
        HFF_T5_u = []
        HFF_T6_u = []
        HFF_T7_u = []
        HFF_T8_u = []
        HFF_T9_u = []
        HFF_T10_u = []
        HFF_T11_u = []
        HFF_T12_u = []

        ## initialize
        T1_lq, T2_lq,T3_lq, T4_lq,T5_lq, T6_lq,T7_lq, T8_lq,T9_lq, T10_lq,T11_lq, T12_lq,\
        T1_u_list, T2_u_list,T3_u_list, T4_u_list,T5_u_list, T6_u_list,T7_u_list, T8_u_list,T9_u_list, T10_u_list,T11_u_list, T12_u_list, c_list,\
        T1_D_list, T2_D_list,T3_D_list, T4_D_list,T5_D_list, T6_D_list,T7_D_list, T8_D_list,T9_D_list, T10_D_list,T11_D_list, T12_D_list = self.init_feature( 
            T1_img_lq, T2_img_lq,T3_img_lq, T4_img_lq,T5_img_lq, T6_img_lq,T7_img_lq, T8_img_lq,T9_img_lq, T10_img_lq,T11_img_lq, T12_img_lq)

        # update F_u_list, T1_u_list, T2_u_list
        
        T1_u_list = self.update_u_list(1, 'T1', T1_D_list, c_list, T1_u_list)
        T2_u_list = self.update_u_list(1, 'T2', T2_D_list, c_list, T2_u_list)
        T3_u_list = self.update_u_list(1, 'T3', T3_D_list, c_list, T3_u_list)
        T4_u_list = self.update_u_list(1, 'T4', T4_D_list, c_list, T4_u_list)
        T5_u_list = self.update_u_list(1, 'T5', T5_D_list, c_list, T5_u_list)
        T6_u_list = self.update_u_list(1, 'T6', T6_D_list, c_list, T6_u_list)
        T7_u_list = self.update_u_list(1, 'T7', T7_D_list, c_list, T7_u_list)
        T8_u_list = self.update_u_list(1, 'T8', T8_D_list, c_list, T8_u_list)
        T9_u_list = self.update_u_list(1, 'T9', T9_D_list, c_list, T9_u_list)
        T10_u_list = self.update_u_list(1, 'T10', T10_D_list, c_list, T10_u_list)
        T11_u_list = self.update_u_list(1, 'T11', T11_D_list, c_list, T11_u_list)
        T12_u_list = self.update_u_list(1, 'T12', T12_D_list, c_list, T12_u_list)
        # cal F_u, F_c , ...
        # UNDONE: decoder or encoder!!!
        T1_Du_cur = self.encoder_dictionary_u(T1_u_list,code(1))
        T1_Dc_cur = T1_lq - T1_Du_cur
        
        T1_Qu_cur = self.decoder_dictionary_u(T1_u_list,code(1))
        
        
        T2_Du_cur = self.encoder_dictionary_u(T2_u_list,code(2))
        T2_Dc_cur = T2_lq - T2_Du_cur
        T2_Qu_cur = self.decoder_dictionary_u(T2_u_list,code(2))
        
        T3_Du_cur = self.encoder_dictionary_u(T3_u_list,code(3))
        T3_Dc_cur = T3_lq - T3_Du_cur
        T3_Qu_cur = self.decoder_dictionary_u(T3_u_list,code(3))
        T4_Du_cur = self.encoder_dictionary_u(T4_u_list,code(4))
        T4_Dc_cur = T4_lq - T4_Du_cur
        T4_Qu_cur = self.decoder_dictionary_u(T4_u_list,code(4))
        T5_Du_cur = self.encoder_dictionary_u(T5_u_list,code(5))
        T5_Dc_cur = T5_lq - T5_Du_cur
        T5_Qu_cur = self.decoder_dictionary_u(T5_u_list,code(5))
        T6_Du_cur = self.encoder_dictionary_u(T6_u_list,code(6))
        T6_Dc_cur = T6_lq - T6_Du_cur
        T6_Qu_cur = self.decoder_dictionary_u(T6_u_list,code(6))
        T7_Du_cur = self.encoder_dictionary_u(T7_u_list,code(7))
        T7_Dc_cur = T7_lq - T7_Du_cur
        T7_Qu_cur = self.decoder_dictionary_u(T7_u_list,code(7))
        T8_Du_cur = self.encoder_dictionary_u(T8_u_list,code(8))
        T8_Dc_cur = T8_lq - T8_Du_cur
        T8_Qu_cur = self.decoder_dictionary_u(T8_u_list,code(8))
        T9_Du_cur = self.encoder_dictionary_u(T9_u_list,code(9))
        T9_Dc_cur = T9_lq - T9_Du_cur
        T9_Qu_cur = self.decoder_dictionary_u(T9_u_list,code(9))
        T10_Du_cur = self.encoder_dictionary_u(T10_u_list,code(10))
        T10_Dc_cur = T10_lq - T10_Du_cur
        T10_Qu_cur = self.decoder_dictionary_u(T10_u_list,code(10))
        T11_Du_cur = self.encoder_dictionary_u(T11_u_list,code(11))
        T11_Dc_cur = T11_lq - T11_Du_cur
        T11_Qu_cur = self.decoder_dictionary_u(T11_u_list,code(11))
        
        T12_Du_cur = self.encoder_dictionary_u(T12_u_list,code(12))
        T12_Dc_cur = T12_lq - T12_Du_cur
        T12_Qu_cur = self.decoder_dictionary_u(T12_u_list,code(12))
        # update c_list
        Dc_res = self.compress_c(torch.cat([T1_Dc_cur, T2_Dc_cur,T3_Dc_cur, T4_Dc_cur,T5_Dc_cur, T6_Dc_cur,
                                            T7_Dc_cur, T8_Dc_cur,T9_Dc_cur, T10_Dc_cur,T11_Dc_cur, T12_Dc_cur], 1))
        c_list = self.update_c_list(1, Dc_res, c_list)

        # MCMCCDic_v2 reconstruct and update in each step
        # reconstruction with mask & sense map
        T1_img_Qu_cur = self.rec_T1_u(T1_Qu_cur)
        T2_img_Qu_cur = self.rec_T2_u(T2_Qu_cur)
        T3_img_Qu_cur = self.rec_T3_u(T3_Qu_cur)
        T4_img_Qu_cur = self.rec_T4_u(T4_Qu_cur)
        T5_img_Qu_cur = self.rec_T5_u(T5_Qu_cur)
        T6_img_Qu_cur = self.rec_T6_u(T6_Qu_cur)
        T7_img_Qu_cur = self.rec_T7_u(T7_Qu_cur)
        T8_img_Qu_cur = self.rec_T8_u(T8_Qu_cur)
        T9_img_Qu_cur = self.rec_T9_u(T9_Qu_cur)
        T10_img_Qu_cur = self.rec_T10_u(T10_Qu_cur)
        T11_img_Qu_cur = self.rec_T11_u(T11_Qu_cur)
        T12_img_Qu_cur = self.rec_T12_u(T12_Qu_cur)
        Qc_cur = self.decoder_dictionary_c(c_list)
        img_Qc_cur = self.rec_c(Qc_cur)

        T1_img_cur = img_Qc_cur[:,0:2,:,:] + T1_img_Qu_cur + T1_img_lq
        T2_img_cur = img_Qc_cur[:,2:4,:,:] + T2_img_Qu_cur  + T2_img_lq
        T3_img_cur = img_Qc_cur[:,4:6,:,:] + T3_img_Qu_cur + T3_img_lq
        T4_img_cur = img_Qc_cur[:,6:8,:,:] + T4_img_Qu_cur  + T4_img_lq
        T5_img_cur = img_Qc_cur[:,8:10,:,:] + T5_img_Qu_cur + T5_img_lq
        T6_img_cur = img_Qc_cur[:,10:12,:,:] + T6_img_Qu_cur  + T6_img_lq
        T7_img_cur = img_Qc_cur[:,12:14,:,:] + T7_img_Qu_cur + T7_img_lq
        T8_img_cur = img_Qc_cur[:,14:16,:,:] + T8_img_Qu_cur  + T8_img_lq
        T9_img_cur = img_Qc_cur[:,16:18,:,:] + T9_img_Qu_cur + T9_img_lq
        T10_img_cur = img_Qc_cur[:,18:20,:,:] + T10_img_Qu_cur  + T10_img_lq
        T11_img_cur = img_Qc_cur[:,20:22,:,:] + T11_img_Qu_cur + T11_img_lq
        T12_img_cur = img_Qc_cur[:,22:24,:,:] + T12_img_Qu_cur  + T12_img_lq

        T1_img_rec_cur = self.DCM(-1, T1_img_cur.permute(0, 2, 3, 1), T1_k_lq, T1_sense, T1_mask)
        T2_img_rec_cur = self.DCM(-1, T2_img_cur.permute(0, 2, 3, 1), T2_k_lq, T2_sense, T2_mask)
        T3_img_rec_cur = self.DCM(-1, T3_img_cur.permute(0, 2, 3, 1), T3_k_lq, T3_sense, T3_mask)
        T4_img_rec_cur = self.DCM(-1, T4_img_cur.permute(0, 2, 3, 1), T4_k_lq, T4_sense, T4_mask)
        T5_img_rec_cur = self.DCM(-1, T5_img_cur.permute(0, 2, 3, 1), T5_k_lq, T5_sense, T5_mask)
        T6_img_rec_cur = self.DCM(-1, T6_img_cur.permute(0, 2, 3, 1), T6_k_lq, T6_sense, T6_mask)
        T7_img_rec_cur = self.DCM(-1, T7_img_cur.permute(0, 2, 3, 1), T7_k_lq, T7_sense, T7_mask)
        T8_img_rec_cur = self.DCM(-1, T8_img_cur.permute(0, 2, 3, 1), T8_k_lq, T8_sense, T8_mask)
        T9_img_rec_cur = self.DCM(-1, T9_img_cur.permute(0, 2, 3, 1), T9_k_lq, T9_sense, T9_mask)
        T10_img_rec_cur = self.DCM(-1, T10_img_cur.permute(0, 2, 3, 1), T10_k_lq, T10_sense, T10_mask)
        T11_img_rec_cur = self.DCM(-1, T11_img_cur.permute(0, 2, 3, 1), T11_k_lq, T11_sense, T11_mask)
        T12_img_rec_cur = self.DCM(-1, T12_img_cur.permute(0, 2, 3, 1), T12_k_lq, T12_sense, T12_mask)
    
        img = torch.stack((T1_img_rec_cur, T2_img_rec_cur,T3_img_rec_cur, T4_img_rec_cur,T5_img_rec_cur, T6_img_rec_cur,\
            T7_img_rec_cur, T8_img_rec_cur,T9_img_rec_cur, T10_img_rec_cur,T11_img_rec_cur, T12_img_rec_cur),dim=1) # X
        
        # img = self.update_Z_2(img,mask_img,0)
        T1_img_rec_cur,T2_img_rec_cur,T3_img_rec_cur,T4_img_rec_cur,T5_img_rec_cur,T6_img_rec_cur,T7_img_rec_cur,T8_img_rec_cur,T9_img_rec_cur,T10_img_rec_cur,T11_img_rec_cur,T12_img_rec_cur = [chunk.squeeze(1) for chunk in torch.chunk(img, chunks=12, dim=1)]
        
        # MCMCCDic_v4 update 1) only
        # again update F_u_list,...
        T1_hq_cur = self.layer_in_T1_res(T1_img_rec_cur.permute(0, 3, 1, 2)-T1_img_lq)
        T2_hq_cur = self.layer_in_T2_res(T2_img_rec_cur.permute(0, 3, 1, 2)-T2_img_lq)
        T3_hq_cur = self.layer_in_T3_res(T3_img_rec_cur.permute(0, 3, 1, 2)-T3_img_lq)
        T4_hq_cur = self.layer_in_T4_res(T4_img_rec_cur.permute(0, 3, 1, 2)-T4_img_lq)
        T5_hq_cur = self.layer_in_T5_res(T5_img_rec_cur.permute(0, 3, 1, 2)-T5_img_lq)
        T6_hq_cur = self.layer_in_T6_res(T6_img_rec_cur.permute(0, 3, 1, 2)-T6_img_lq)
        T7_hq_cur = self.layer_in_T7_res(T7_img_rec_cur.permute(0, 3, 1, 2)-T7_img_lq)
        T8_hq_cur = self.layer_in_T8_res(T8_img_rec_cur.permute(0, 3, 1, 2)- T8_img_lq)
        T9_hq_cur = self.layer_in_T9_res(T9_img_rec_cur.permute(0, 3, 1, 2)- T9_img_lq)
        T10_hq_cur = self.layer_in_T10_res(T10_img_rec_cur.permute(0, 3, 1, 2) - T10_img_lq)
        T11_hq_cur = self.layer_in_T11_res(T11_img_rec_cur.permute(0, 3, 1, 2) - T11_img_lq)
        T12_hq_cur = self.layer_in_T12_res(T12_img_rec_cur.permute(0, 3, 1, 2) - T12_img_lq)

        
        
        T1_Q_list_cur  = self.adjoint_decoder_dictionary_u(T1_hq_cur,code(1))
        T2_Q_list_cur  = self.adjoint_decoder_dictionary_u(T2_hq_cur,code(2))
        T3_Q_list_cur  = self.adjoint_decoder_dictionary_u(T3_hq_cur,code(3))
        T4_Q_list_cur  = self.adjoint_decoder_dictionary_u(T4_hq_cur,code(4))
        T5_Q_list_cur  = self.adjoint_decoder_dictionary_u(T5_hq_cur,code(5))
        T6_Q_list_cur  = self.adjoint_decoder_dictionary_u(T6_hq_cur,code(6))
        T7_Q_list_cur = self.adjoint_decoder_dictionary_u(T7_hq_cur,code(7))
        T8_Q_list_cur = self.adjoint_decoder_dictionary_u(T8_hq_cur,code(8))
        T9_Q_list_cur = self.adjoint_decoder_dictionary_u(T9_hq_cur,code(9))
        T10_Q_list_cur = self.adjoint_decoder_dictionary_u(T10_hq_cur,code(10))
        T11_Q_list_cur = self.adjoint_decoder_dictionary_u(T11_hq_cur,code(11))
        T12_Q_list_cur = self.adjoint_decoder_dictionary_u(T12_hq_cur,code(12))

        T1_u_list = self.update_u_list_new(1, 'T1', T1_Q_list_cur, c_list, T1_u_list)
        T2_u_list = self.update_u_list_new(1, 'T2', T2_Q_list_cur, c_list, T2_u_list)
        T3_u_list = self.update_u_list_new(1, 'T3', T3_Q_list_cur, c_list, T3_u_list)
        T4_u_list = self.update_u_list_new(1, 'T4', T4_Q_list_cur, c_list, T4_u_list)
        T5_u_list = self.update_u_list_new(1, 'T5', T5_Q_list_cur, c_list, T5_u_list)
        T6_u_list = self.update_u_list_new(1, 'T6', T6_Q_list_cur, c_list, T6_u_list)
        T7_u_list = self.update_u_list_new(1, 'T7', T7_Q_list_cur, c_list, T7_u_list)
        T8_u_list = self.update_u_list_new(1, 'T8', T8_Q_list_cur, c_list, T8_u_list)
        T9_u_list = self.update_u_list_new(1, 'T9', T9_Q_list_cur, c_list, T9_u_list)
        T10_u_list = self.update_u_list_new(1, 'T10', T10_Q_list_cur, c_list, T10_u_list)
        T11_u_list = self.update_u_list_new(1, 'T11', T11_Q_list_cur, c_list, T11_u_list)
        T12_u_list = self.update_u_list_new(1, 'T12', T12_Q_list_cur, c_list, T12_u_list)
        T1_Qu_cur = self.decoder_dictionary_u(T1_u_list,code(1))
        T1_Qc_cur = T1_hq_cur - T1_Qu_cur
        HFF_T1_u.append(T1_Qu_cur)
        T2_Qu_cur = self.decoder_dictionary_u(T2_u_list,code(2))
        T2_Qc_cur = T2_hq_cur - T2_Qu_cur
        HFF_T2_u.append(T2_Qu_cur)
        T3_Qu_cur = self.decoder_dictionary_u(T3_u_list,code(3))
        T3_Qc_cur = T3_hq_cur - T3_Qu_cur
        HFF_T3_u.append(T3_Qu_cur)
        T4_Qu_cur = self.decoder_dictionary_u(T4_u_list,code(4))
        T4_Qc_cur = T4_hq_cur - T4_Qu_cur
        HFF_T4_u.append(T4_Qu_cur)
        T5_Qu_cur = self.decoder_dictionary_u(T5_u_list,code(5))
        T5_Qc_cur = T5_hq_cur - T5_Qu_cur
        HFF_T5_u.append(T5_Qu_cur)
        T6_Qu_cur = self.decoder_dictionary_u(T6_u_list,code(6))
        T6_Qc_cur = T6_hq_cur - T6_Qu_cur
        HFF_T6_u.append(T6_Qu_cur)
        T7_Qu_cur = self.decoder_dictionary_u(T7_u_list,code(7))
        T7_Qc_cur = T7_hq_cur - T7_Qu_cur
        HFF_T7_u.append(T7_Qu_cur)
        T8_Qu_cur = self.decoder_dictionary_u(T8_u_list,code(8))
        T8_Qc_cur = T8_hq_cur - T8_Qu_cur
        HFF_T8_u.append(T8_Qu_cur)
        T9_Qu_cur = self.decoder_dictionary_u(T9_u_list,code(9))
        T9_Qc_cur = T9_hq_cur - T9_Qu_cur
        HFF_T9_u.append(T9_Qu_cur)
        T10_Qu_cur = self.decoder_dictionary_u(T10_u_list,code(10))
        T10_Qc_cur = T10_hq_cur - T10_Qu_cur
        HFF_T10_u.append(T10_Qu_cur)
        T11_Qu_cur = self.decoder_dictionary_u(T11_u_list,code(11))
        T11_Qc_cur = T11_hq_cur - T11_Qu_cur
        HFF_T11_u.append(T11_Qu_cur)
        T12_Qu_cur = self.decoder_dictionary_u(T12_u_list,code(12))
        T12_Qc_cur = T12_hq_cur - T12_Qu_cur
        HFF_T12_u.append(T12_Qu_cur)
        
        
        # again update c_list
        Qc_res = self.compress_c(torch.cat([T1_Qc_cur, T2_Qc_cur,T3_Qc_cur, T4_Qc_cur,T5_Qc_cur, T6_Qc_cur,
                                            T7_Qc_cur, T8_Qc_cur,T9_Qc_cur, T10_Qc_cur,T11_Qc_cur, T12_Qc_cur], 1))
        c_list = self.update_c_list_new(1, Qc_res, c_list)

        # starting from the 2nd iteration
        for idx in range(2, self.ista_num_steps):
            # MCMCCDic_v2 reconstruct and update in each step
            # reconstruction with mask & sense map
            T1_img_Qu_cur = self.rec_T1_u(T1_Qu_cur)
            T2_img_Qu_cur = self.rec_T2_u(T2_Qu_cur)
            T3_img_Qu_cur = self.rec_T3_u(T3_Qu_cur)
            T4_img_Qu_cur = self.rec_T4_u(T4_Qu_cur)
            T5_img_Qu_cur = self.rec_T5_u(T5_Qu_cur)
            T6_img_Qu_cur = self.rec_T6_u(T6_Qu_cur)
            T7_img_Qu_cur = self.rec_T7_u(T7_Qu_cur)
            T8_img_Qu_cur = self.rec_T8_u(T8_Qu_cur)
            T9_img_Qu_cur = self.rec_T9_u(T9_Qu_cur)
            T10_img_Qu_cur = self.rec_T10_u(T10_Qu_cur)
            T11_img_Qu_cur = self.rec_T11_u(T11_Qu_cur)
            T12_img_Qu_cur = self.rec_T12_u(T12_Qu_cur)        
            
            Qc_cur = self.decoder_dictionary_c(c_list)
            img_Qc_cur = self.rec_c(Qc_cur)
            
            T1_img_cur = img_Qc_cur[:,0:2,:,:] + T1_img_Qu_cur + T1_img_lq
            T2_img_cur = img_Qc_cur[:,2:4,:,:] + T2_img_Qu_cur + T2_img_lq
            T3_img_cur = img_Qc_cur[:,4:6,:,:] + T3_img_Qu_cur + T3_img_lq
            T4_img_cur = img_Qc_cur[:,6:8,:,:] + T4_img_Qu_cur + T4_img_lq
            T5_img_cur = img_Qc_cur[:,8:10,:,:] + T5_img_Qu_cur + T5_img_lq
            T6_img_cur = img_Qc_cur[:,10:12,:,:] + T6_img_Qu_cur + T6_img_lq
            T7_img_cur = img_Qc_cur[:, 12:14, :, :] + T7_img_Qu_cur + T7_img_lq
            T8_img_cur = img_Qc_cur[:, 14:16, :, :] + T8_img_Qu_cur + T8_img_lq
            T9_img_cur = img_Qc_cur[:, 16:18, :, :] + T9_img_Qu_cur + T9_img_lq
            T10_img_cur = img_Qc_cur[:, 18:20, :, :] + T10_img_Qu_cur + T10_img_lq
            T11_img_cur = img_Qc_cur[:, 20:22, :, :] + T11_img_Qu_cur + T11_img_lq
            T12_img_cur = img_Qc_cur[:, 22:24, :, :] + T12_img_Qu_cur + T12_img_lq
            # T1_img_rec_cur = self.DCM(1, T1_img_cur.permute(0, 2, 3, 1), T1_k_lq, T1_sense, T1_mask,Z_1)
            # T2_img_rec_cur = self.DCM(1, T2_img_cur.permute(0, 2, 3, 1), T2_k_lq, T2_sense, T2_mask,Z_2)
            # T3_img_rec_cur = self.DCM(1, T3_img_cur.permute(0, 2, 3, 1), T3_k_lq, T3_sense, T3_mask,Z_3) # type: ignore
            # T4_img_rec_cur = self.DCM(1, T4_img_cur.permute(0, 2, 3, 1), T4_k_lq, T4_sense, T4_mask,Z_4)
            # T5_img_rec_cur = self.DCM(1, T5_img_cur.permute(0, 2, 3, 1), T5_k_lq, T5_sense, T5_mask,Z_5)
            # T6_img_rec_cur = self.DCM(1, T6_img_cur.permute(0, 2, 3, 1), T6_k_lq, T6_sense, T6_mask,Z_6)
            # T7_img_rec_cur = self.DCM(1, T7_img_cur.permute(0, 2, 3, 1), T7_k_lq, T7_sense, T7_mask,Z_7)
            # T8_img_rec_cur = self.DCM(1, T8_img_cur.permute(0, 2, 3, 1), T8_k_lq, T8_sense, T8_mask,Z_8)
            # T9_img_rec_cur = self.DCM(1, T9_img_cur.permute(0, 2, 3, 1), T9_k_lq, T9_sense, T9_mask,Z_9)
            # T10_img_rec_cur = self.DCM(1, T10_img_cur.permute(0, 2, 3, 1), T10_k_lq, T10_sense, T10_mask,Z_10)
            # T11_img_rec_cur = self.DCM(1, T11_img_cur.permute(0, 2, 3, 1), T11_k_lq, T11_sense, T11_mask,Z_11)
            # T12_img_rec_cur = self.DCM(1, T12_img_cur.permute(0, 2, 3, 1), T12_k_lq, T12_sense, T12_mask,Z_12)
            T1_img_rec_cur = self.DCM(-1, T1_img_cur.permute(0, 2, 3, 1), T1_k_lq, T1_sense, T1_mask)
            T2_img_rec_cur = self.DCM(-1, T2_img_cur.permute(0, 2, 3, 1), T2_k_lq, T2_sense, T2_mask)
            T3_img_rec_cur = self.DCM(-1, T3_img_cur.permute(0, 2, 3, 1), T3_k_lq, T3_sense, T3_mask)
            T4_img_rec_cur = self.DCM(-1, T4_img_cur.permute(0, 2, 3, 1), T4_k_lq, T4_sense, T4_mask)
            T5_img_rec_cur = self.DCM(-1, T5_img_cur.permute(0, 2, 3, 1), T5_k_lq, T5_sense, T5_mask)
            T6_img_rec_cur = self.DCM(-1, T6_img_cur.permute(0, 2, 3, 1), T6_k_lq, T6_sense, T6_mask)
            T7_img_rec_cur = self.DCM(-1, T7_img_cur.permute(0, 2, 3, 1), T7_k_lq, T7_sense, T7_mask)
            T8_img_rec_cur = self.DCM(-1, T8_img_cur.permute(0, 2, 3, 1), T8_k_lq, T8_sense, T8_mask)
            T9_img_rec_cur = self.DCM(-1, T9_img_cur.permute(0, 2, 3, 1), T9_k_lq, T9_sense, T9_mask)
            T10_img_rec_cur = self.DCM(-1, T10_img_cur.permute(0, 2, 3, 1), T10_k_lq, T10_sense, T10_mask)
            T11_img_rec_cur = self.DCM(-1, T11_img_cur.permute(0, 2, 3, 1), T11_k_lq, T11_sense, T11_mask)
            T12_img_rec_cur = self.DCM(-1, T12_img_cur.permute(0, 2, 3, 1), T12_k_lq, T12_sense, T12_mask)
            img = torch.stack((T1_img_rec_cur, T2_img_rec_cur,T3_img_rec_cur, T4_img_rec_cur,T5_img_rec_cur, T6_img_rec_cur,\
            T7_img_rec_cur, T8_img_rec_cur,T9_img_rec_cur, T10_img_rec_cur,T11_img_rec_cur, T12_img_rec_cur),dim=1) # X
            
            img = self.update_Z_2(img,mask_img,idx-1)
    
            T1_img_rec_cur,T2_img_rec_cur,T3_img_rec_cur,T4_img_rec_cur,T5_img_rec_cur,T6_img_rec_cur,T7_img_rec_cur,T8_img_rec_cur,T9_img_rec_cur,T10_img_rec_cur,T11_img_rec_cur,T12_img_rec_cur = [chunk.squeeze(1) for chunk in torch.chunk(img, chunks=12, dim=1)]
        
            
            # MCMCCDic_v4 update 1) only
            # again update F_u_list,...

            T1_hq_cur = self.layer_in_T1_res(T1_img_rec_cur - T1_img_lq)
            T2_hq_cur = self.layer_in_T2_res(T2_img_rec_cur - T2_img_lq)
            T3_hq_cur = self.layer_in_T3_res(T3_img_rec_cur- T3_img_lq)
            T4_hq_cur = self.layer_in_T4_res(T4_img_rec_cur- T4_img_lq)
            T5_hq_cur = self.layer_in_T5_res(T5_img_rec_cur- T5_img_lq)
            T6_hq_cur = self.layer_in_T6_res(T6_img_rec_cur- T6_img_lq)
            T7_hq_cur = self.layer_in_T7_res(T7_img_rec_cur- T7_img_lq)
            T8_hq_cur = self.layer_in_T8_res(T8_img_rec_cur- T8_img_lq)
            T9_hq_cur = self.layer_in_T9_res(T9_img_rec_cur - T9_img_lq)
            T10_hq_cur = self.layer_in_T10_res(T10_img_rec_cur- T10_img_lq)
            T11_hq_cur = self.layer_in_T11_res(T11_img_rec_cur- T11_img_lq)
            T12_hq_cur = self.layer_in_T12_res(T12_img_rec_cur- T12_img_lq)
            
            # T1_hq_cur = self.layer_in_T1_res(T1_img_rec_cur.permute(0, 3, 1, 2) - T1_img_lq)
            # T2_hq_cur = self.layer_in_T2_res(T2_img_rec_cur.permute(0, 3, 1, 2) - T2_img_lq)
            # T3_hq_cur = self.layer_in_T3_res(T3_img_rec_cur.permute(0, 3, 1, 2) - T3_img_lq)
            # T4_hq_cur = self.layer_in_T4_res(T4_img_rec_cur.permute(0, 3, 1, 2) - T4_img_lq)
            # T5_hq_cur = self.layer_in_T5_res(T5_img_rec_cur.permute(0, 3, 1, 2) - T5_img_lq)
            # T6_hq_cur = self.layer_in_T6_res(T6_img_rec_cur.permute(0, 3, 1, 2) - T6_img_lq)
            # T7_hq_cur = self.layer_in_T7_res(T7_img_rec_cur.permute(0, 3, 1, 2) - T7_img_lq)
            # T8_hq_cur = self.layer_in_T8_res(T8_img_rec_cur.permute(0, 3, 1, 2) - T8_img_lq)
            # T9_hq_cur = self.layer_in_T9_res(T9_img_rec_cur.permute(0, 3, 1, 2) - T9_img_lq)
            # T10_hq_cur = self.layer_in_T10_res(T10_img_rec_cur.permute(0, 3, 1, 2) - T10_img_lq)
            # T11_hq_cur = self.layer_in_T11_res(T11_img_rec_cur.permute(0, 3, 1, 2) - T11_img_lq)
            # T12_hq_cur = self.layer_in_T12_res(T12_img_rec_cur.permute(0, 3, 1, 2) - T12_img_lq)
            
            
            T1_Q_list_cur = self.adjoint_decoder_dictionary_u(T1_hq_cur,code(1))
            T2_Q_list_cur = self.adjoint_decoder_dictionary_u(T2_hq_cur,code(2))
            T3_Q_list_cur = self.adjoint_decoder_dictionary_u(T3_hq_cur,code(3))
            T4_Q_list_cur = self.adjoint_decoder_dictionary_u(T4_hq_cur,code(4))
            T5_Q_list_cur = self.adjoint_decoder_dictionary_u(T5_hq_cur,code(5))
            T6_Q_list_cur = self.adjoint_decoder_dictionary_u(T6_hq_cur,code(6))
            T7_Q_list_cur = self.adjoint_decoder_dictionary_u(T7_hq_cur,code(7))
            T8_Q_list_cur = self.adjoint_decoder_dictionary_u(T8_hq_cur,code(8))
            T9_Q_list_cur = self.adjoint_decoder_dictionary_u(T9_hq_cur,code(9))
            T10_Q_list_cur = self.adjoint_decoder_dictionary_u(T10_hq_cur,code(10))
            T11_Q_list_cur = self.adjoint_decoder_dictionary_u(T11_hq_cur,code(11))
            T12_Q_list_cur = self.adjoint_decoder_dictionary_u(T12_hq_cur,code(12))

            T1_u_list = self.update_u_list_new(idx, 'T1', T1_Q_list_cur, c_list, T1_u_list)
            T2_u_list = self.update_u_list_new(idx, 'T2', T2_Q_list_cur, c_list, T2_u_list)
            T3_u_list = self.update_u_list_new(idx, 'T3', T3_Q_list_cur, c_list, T3_u_list)
            T4_u_list = self.update_u_list_new(idx, 'T4', T4_Q_list_cur, c_list, T4_u_list)
            T5_u_list = self.update_u_list_new(idx, 'T5', T5_Q_list_cur, c_list, T5_u_list)
            T6_u_list = self.update_u_list_new(idx, 'T6', T6_Q_list_cur, c_list, T6_u_list)
            T7_u_list = self.update_u_list_new(idx, 'T7', T7_Q_list_cur, c_list, T7_u_list)
            T8_u_list = self.update_u_list_new(idx, 'T8', T8_Q_list_cur, c_list, T8_u_list)
            T9_u_list = self.update_u_list_new(idx, 'T9', T9_Q_list_cur, c_list, T9_u_list)
            T10_u_list = self.update_u_list_new(idx, 'T10', T10_Q_list_cur, c_list, T10_u_list)
            T11_u_list = self.update_u_list_new(idx, 'T11', T11_Q_list_cur, c_list, T11_u_list)
            T12_u_list = self.update_u_list_new(idx, 'T12', T12_Q_list_cur, c_list, T12_u_list)

            T1_Qu_cur = self.decoder_dictionary_u(T1_u_list,code(1))
            T1_Qc_cur = T1_hq_cur - T1_Qu_cur
            HFF_T1_u.append(T1_Qu_cur)
            T2_Qu_cur = self.decoder_dictionary_u(T2_u_list,code(2))
            T2_Qc_cur = T2_hq_cur - T2_Qu_cur
            HFF_T2_u.append(T2_Qu_cur)
            T3_Qu_cur = self.decoder_dictionary_u(T3_u_list,code(3))
            T3_Qc_cur = T3_hq_cur - T3_Qu_cur
            HFF_T3_u.append(T3_Qu_cur)
            T4_Qu_cur = self.decoder_dictionary_u(T4_u_list,code(4))
            T4_Qc_cur = T4_hq_cur - T4_Qu_cur
            HFF_T4_u.append(T4_Qu_cur)
            T5_Qu_cur = self.decoder_dictionary_u(T5_u_list,code(5))
            T5_Qc_cur = T5_hq_cur - T5_Qu_cur
            HFF_T5_u.append(T5_Qu_cur)
            T6_Qu_cur = self.decoder_dictionary_u(T6_u_list,code(6))
            T6_Qc_cur = T6_hq_cur - T6_Qu_cur
            HFF_T6_u.append(T6_Qu_cur)
            T7_Qu_cur = self.decoder_dictionary_u(T7_u_list,code(7))
            T7_Qc_cur = T7_hq_cur - T7_Qu_cur
            HFF_T7_u.append(T7_Qu_cur)
            T8_Qu_cur = self.decoder_dictionary_u(T8_u_list,code(8))
            T8_Qc_cur = T8_hq_cur - T8_Qu_cur
            HFF_T8_u.append(T8_Qu_cur)

            T9_Qu_cur = self.decoder_dictionary_u(T9_u_list,code(9))
            T9_Qc_cur = T9_hq_cur - T9_Qu_cur
            HFF_T9_u.append(T9_Qu_cur)

            T10_Qu_cur = self.decoder_dictionary_u(T10_u_list,code(10))
            T10_Qc_cur = T10_hq_cur - T10_Qu_cur
            HFF_T10_u.append(T10_Qu_cur)

            T11_Qu_cur = self.decoder_dictionary_u(T11_u_list,code(11))
            T11_Qc_cur = T11_hq_cur - T11_Qu_cur
            HFF_T11_u.append(T11_Qu_cur)

            T12_Qu_cur = self.decoder_dictionary_u(T12_u_list,code(12))
            T12_Qc_cur = T12_hq_cur - T12_Qu_cur
            HFF_T12_u.append(T12_Qu_cur)
                        
            
            
            # again update c_list
            Qc_res = self.compress_c(torch.cat([T1_Qc_cur, T2_Qc_cur,T3_Qc_cur, T4_Qc_cur,T5_Qc_cur, T6_Qc_cur,
                                                T7_Qc_cur, T8_Qc_cur,T9_Qc_cur, T10_Qc_cur,T11_Qc_cur, T12_Qc_cur], 1))
            c_list = self.update_c_list_new(idx, Qc_res, c_list)
        
        ## reconstruction, channel = 64
        T1_Qu = self.HFF_fuse_T1_u(torch.cat(HFF_T1_u, 1))
        T2_Qu = self.HFF_fuse_T2_u(torch.cat(HFF_T2_u, 1))
        T3_Qu = self.HFF_fuse_T3_u(torch.cat(HFF_T3_u, 1))
        T4_Qu = self.HFF_fuse_T4_u(torch.cat(HFF_T4_u, 1))
        T5_Qu = self.HFF_fuse_T5_u(torch.cat(HFF_T5_u, 1))
        T6_Qu = self.HFF_fuse_T6_u(torch.cat(HFF_T6_u, 1))
        T7_Qu = self.HFF_fuse_T7_u(torch.cat(HFF_T7_u, 1))
        T8_Qu = self.HFF_fuse_T8_u(torch.cat(HFF_T8_u, 1))
        T9_Qu = self.HFF_fuse_T9_u(torch.cat(HFF_T9_u, 1))
        T10_Qu = self.HFF_fuse_T10_u(torch.cat(HFF_T10_u, 1))
        T11_Qu = self.HFF_fuse_T11_u(torch.cat(HFF_T11_u, 1))
        T12_Qu = self.HFF_fuse_T12_u(torch.cat(HFF_T12_u, 1))
        Qc = self.decoder_dictionary_c(c_list)
       
        T1_img_Qu = self.rec_T1_u(T1_Qu)
        T2_img_Qu = self.rec_T2_u(T2_Qu)
        T3_img_Qu = self.rec_T3_u(T3_Qu)
        T4_img_Qu = self.rec_T4_u(T4_Qu)
        T5_img_Qu = self.rec_T5_u(T5_Qu)
        T6_img_Qu = self.rec_T6_u(T6_Qu)
        T7_img_Qu = self.rec_T7_u(T7_Qu)
        T8_img_Qu = self.rec_T8_u(T8_Qu)
        T9_img_Qu = self.rec_T9_u(T9_Qu)
        T10_img_Qu = self.rec_T10_u(T10_Qu)
        T11_img_Qu = self.rec_T11_u(T11_Qu)
        T12_img_Qu = self.rec_T12_u(T12_Qu)
        img_Qc = self.rec_c(Qc)

        T1_img_rec = img_Qc[:, 0:2, :, :] + T1_img_Qu + T1_img_lq
        T2_img_rec = img_Qc[:, 2:4, :, :] + T2_img_Qu + T2_img_lq
        T3_img_rec = img_Qc[:, 4:6, :, :] + T3_img_Qu + T3_img_lq
        T4_img_rec = img_Qc[:, 6:8, :, :] + T4_img_Qu + T4_img_lq
        T5_img_rec = img_Qc[:, 8:10, :, :] + T5_img_Qu + T5_img_lq
        T6_img_rec = img_Qc[:, 10:12, :, :] + T6_img_Qu + T6_img_lq
        T7_img_rec = img_Qc[:, 12:14, :, :] + T7_img_Qu + T7_img_lq
        T8_img_rec = img_Qc[:, 14:16, :, :] + T8_img_Qu + T8_img_lq
        T9_img_rec = img_Qc[:, 16:18, :, :] + T9_img_Qu + T9_img_lq
        T10_img_rec = img_Qc[:, 18:20, :, :] + T10_img_Qu + T10_img_lq
        T11_img_rec = img_Qc[:, 20:22, :, :] + T11_img_Qu + T11_img_lq
        T12_img_rec = img_Qc[:, 22:24, :, :] + T12_img_Qu + T12_img_lq
        img = torch.stack((T1_img_rec, T2_img_rec,T3_img_rec, T4_img_rec,T5_img_rec, T6_img_rec,\
            T7_img_rec, T8_img_rec,T9_img_rec, T10_img_rec,T11_img_rec, T12_img_rec),dim=1) # X
        # Z = img.permute(0,1,3,4,2).permute(0,1,4,2,3)
        Z= img
        R1_map_2,R2_map_2,S_map_2 = mdi.R_S(Z)
        R1_map_2 = R1_map_2 * mask_img
        R2_map_2 = R2_map_2 * mask_img
        S_map_2 = S_map_2 * mask_img   
        # R1_map_2 =  norm_r1(R1_map_2) 
        R1_map = torch.stack((R1_map_2.real,R1_map_2.imag),dim=1)
        R2_map = torch.stack((R2_map_2.real,R2_map_2.imag),dim=1)
        S_map = torch.stack((S_map_2.real,S_map_2.imag),dim=1)
        # R1_map = self.r1[-1](R1_map) + R1_map
        # S_map = self.s[-1](S_map) + S_map
        # R2_map = self.r2[-1](R2_map) + R2_map
        # R1_map,R2_map,S_map  = self.RSZ_net(R1_map,R2_map,S_map)
        # R1_map,R2_map,S_map =self.RSZnet[self.ista_num_steps-1](mask_img,img.permute(0,1,3,4,2),Z)
        
        R1_map = R1_map[:,0,:,:].squeeze(1) + 1j * R1_map[:,1,:,:].squeeze(1) 
        R2_map = R2_map[:,0,:,:].squeeze(1) + 1j * R2_map[:,1,:,:].squeeze(1) 
        S_map = S_map[:,0,:,:].squeeze(1) + 1j * S_map[:,1,:,:].squeeze(1) 

        return T1_img_rec, T2_img_rec,T3_img_rec, T4_img_rec,T5_img_rec, T6_img_rec,T7_img_rec, T8_img_rec,T9_img_rec, T10_img_rec,T11_img_rec, T12_img_rec,S_map,R1_map,R2_map
    def initialize_sparse_codes(self, x, rand_bool = False):
        code_list = []

        num_samples =  x.shape[0]    
        input_spatial_dim_1 = x.shape[2]
        input_spatial_dim_2 = x.shape[3]

        if rand_bool:
            initializer = torch.rand
        else:
            initializer = torch.zeros

        for i in range(self.num_layers):
            feature_map_dim_1 = int(input_spatial_dim_1/  (2 ** i) )
            feature_map_dim_2 = int(input_spatial_dim_2/  (2 ** i) )
            code_tensor = initializer(num_samples, self.hidden_layer_width_list[self.num_layers-i-1],  feature_map_dim_1, feature_map_dim_2 )
            code_list.append(code_tensor)

        code_list.reverse() # order the code from low-spatial-dim to high-spatial-dim.
        return code_list

    def power_iteration_conv_model(self, conv_model, num_simulations: int):

        eigen_vec_list = self.initialize_sparse_codes(x = torch.zeros(1, 3, 64, 64), rand_bool = True)

        adjoint_conv_model = adjoint_dictionary_model_c(conv_model)

        for _ in range(num_simulations):
            # calculate the matrix-by-vector product Ab
            eigen_vec_list = adjoint_conv_model(conv_model(eigen_vec_list))
            # calculate the norm
            flatten_x_norm = torch.norm(torch.cat([x.flatten() for x in eigen_vec_list ]) )
            # re-normalize the vector
            eigen_vec_list = [x/ flatten_x_norm for x in eigen_vec_list] 

        eigen_vecs_flatten = torch.cat([x.flatten() for x in eigen_vec_list])

        linear_trans_eigen_vecs_list = adjoint_conv_model(conv_model(eigen_vec_list ))

        linear_trans_eigen_vecs_list_flatten = torch.cat([x.flatten() for x in linear_trans_eigen_vecs_list] )

        numerator = torch.dot(eigen_vecs_flatten, linear_trans_eigen_vecs_list_flatten)

        denominator = torch.dot(eigen_vecs_flatten, eigen_vecs_flatten)

        eigenvalue = numerator / denominator
        return eigenvalue
    


class MCMCCDic_v4_hq(nn.Module):
    def __init__(self):
        super(MCMCCDic_v4_hq, self).__init__()

        # self.in_channel = 1
        self.channel_fea = 64
        self.num_stages = 4 # 4
        self.predict_ista = ista_unet(kernel_size=3, hidden_layer_width_list=[128, 96, 64],prompt_dim=[72, 48, 24], prompt_len=[5, 5, 5],
                                      prompt_size=[16, 32, 64], n_classes=self.channel_fea, ista_num_steps=self.num_stages)
        # self.decoder = decoder(self.in_channel, self.channel_fea)

    # def forward(self, F_img_lq, F_k_lq, F_sense, F_mask, 
    #                 T1_img_lq, T1_k_lq, T1_sense, T1_mask, 
    #                 T2_img_lq, T2_k_lq, T2_sense, T2_mask):
    def com2real(self,data):
        real =data.real
        imag = data.imag
        return torch.stack((real,imag),-1)
    def real2com(self,data):
        real =data[...,0].squeeze(-1)
        imag = data[...,1].squeeze(-1)
        return real+ 1j * imag
    def forward(self, ksp , sens , mask,img,mask_img):
        
        # img: [bs, 256, 176, 2]
        # kspace: [bs, 32, 256, 176, 2]
        # sense: [bs, 32, 256, 176, 2]
        # mask: [bs, 1, 256, 176, 1]; mask*kspace
        nb,nc,ne,ny,nz = ksp.shape
        sens = self.com2real(sens)
        T1_img_lq = self.com2real(img[:,0,:,:].squeeze(2))
        T2_img_lq = self.com2real(img[:,1,:,:].squeeze(2))
        T3_img_lq = self.com2real(img[:,2,:,:].squeeze(2))
        T4_img_lq = self.com2real(img[:,3,:,:].squeeze(2))
        T5_img_lq = self.com2real(img[:,4,:,:].squeeze(2))
        T6_img_lq = self.com2real(img[:,5,:,:].squeeze(2))
        T7_img_lq = self.com2real(img[:,6,:,:].squeeze(2))
        T8_img_lq = self.com2real(img[:,7,:,:].squeeze(2))
        T9_img_lq = self.com2real(img[:,8,:,:].squeeze(2))
        T10_img_lq = self.com2real(img[:,9,:,:].squeeze(2))
        T11_img_lq = self.com2real(img[:,10,:,:].squeeze(2))
        T12_img_lq = self.com2real(img[:,11,:,:].squeeze(2))
        
        T1_img_lq = T1_img_lq.permute(0, 3, 1, 2)
        T2_img_lq = T2_img_lq.permute(0, 3, 1, 2)
        T3_img_lq = T3_img_lq.permute(0, 3, 1, 2)
        T4_img_lq = T4_img_lq.permute(0, 3, 1, 2)
        T5_img_lq = T5_img_lq.permute(0, 3, 1, 2)
        T6_img_lq = T6_img_lq.permute(0, 3, 1, 2)
        T7_img_lq = T7_img_lq.permute(0, 3, 1, 2)
        T8_img_lq = T8_img_lq.permute(0, 3, 1, 2)
        T9_img_lq = T9_img_lq.permute(0, 3, 1, 2)
        T10_img_lq = T10_img_lq.permute(0, 3, 1, 2)
        T11_img_lq = T11_img_lq.permute(0, 3, 1, 2)
        T12_img_lq = T12_img_lq.permute(0, 3, 1, 2)
        
        T1_k_lq = self.com2real(ksp[:,:,0,:,:].squeeze(2))
        T2_k_lq = self.com2real(ksp[:,:,1,:,:].squeeze(2))
        T3_k_lq = self.com2real(ksp[:,:,2,:,:].squeeze(2))
        T4_k_lq = self.com2real(ksp[:,:,3,:,:].squeeze(2))
        T5_k_lq = self.com2real(ksp[:,:,4,:,:].squeeze(2))
        T6_k_lq = self.com2real(ksp[:,:,5,:,:].squeeze(2))
        T7_k_lq = self.com2real(ksp[:,:,6,:,:].squeeze(2))
        T8_k_lq = self.com2real(ksp[:,:,7,:,:].squeeze(2))
        T9_k_lq = self.com2real(ksp[:,:,8,:,:].squeeze(2))
        T10_k_lq = self.com2real(ksp[:,:,9,:,:].squeeze(2))
        T11_k_lq = self.com2real(ksp[:,:,10,:,:].squeeze(2))
        T12_k_lq = self.com2real(ksp[:,:,11,:,:].squeeze(2))
        
        
        T1_img_rec, T2_img_rec,T3_img_rec, T4_img_rec,T5_img_rec, T6_img_rec,\
            T7_img_rec, T8_img_rec,T9_img_rec, T10_img_rec,T11_img_rec, T12_img_rec,\
            S_map,R1_map,R2_map  = self.predict_ista(
                                                              T1_img_lq, T1_k_lq, sens, mask, 
                                                              T2_img_lq, T2_k_lq, sens, mask,
                                                              T3_img_lq, T3_k_lq, sens, mask, 
                                                              T4_img_lq, T4_k_lq, sens, mask,
                                                              T5_img_lq, T5_k_lq, sens, mask, 
                                                              T6_img_lq, T6_k_lq, sens, mask,
                                                              T7_img_lq, T7_k_lq, sens, mask, 
                                                              T8_img_lq, T8_k_lq, sens, mask,
                                                              T9_img_lq, T9_k_lq, sens, mask, 
                                                              T10_img_lq, T10_k_lq, sens, mask,
                                                              T11_img_lq, T11_k_lq, sens, mask, 
                                                              T12_img_lq, T12_k_lq, sens, mask,
                                                              mask_img
                                                              )
         
        # T1_img_rec, T2_img_rec,T3_img_rec, T4_img_rec,T5_img_rec, T6_img_rec,T7_img_rec, T8_img_rec,T9_img_rec, T10_img_rec,T11_img_rec, T12_img_rec = self.predict_ista(
        #                                                       T1_img_lq, T1_k_lq, sens, mask, 
        #                                                       T2_img_lq, T2_k_lq, sens, mask,
        #                                                       T3_img_lq, T3_k_lq, sens, mask, 
        #                                                       T4_img_lq, T4_k_lq, sens, mask,
        #                                                       T5_img_lq, T5_k_lq, sens, mask, 
        #                                                       T6_img_lq, T6_k_lq, sens, mask,
        #                                                       T7_img_lq, T7_k_lq, sens, mask, 
        #                                                       T8_img_lq, T8_k_lq, sens, mask,
        #                                                       T9_img_lq, T9_k_lq, sens, mask, 
        #                                                       T10_img_lq, T10_k_lq, sens, mask,
        #                                                       T11_img_lq, T11_k_lq, sens, mask, 
        #                                                       T12_img_lq, T12_k_lq, sens, mask,
        #                                                       mask_img
        #                                                       )
        T1_img_rec = self.real2com(T1_img_rec.permute(0, 2, 3, 1))
        T2_img_rec = self.real2com(T2_img_rec.permute(0, 2, 3, 1))
        T3_img_rec = self.real2com(T3_img_rec.permute(0, 2, 3, 1))
        T4_img_rec = self.real2com(T4_img_rec.permute(0, 2, 3, 1))
        T5_img_rec = self.real2com(T5_img_rec.permute(0, 2, 3, 1))
        T6_img_rec = self.real2com(T6_img_rec.permute(0, 2, 3, 1))
        T7_img_rec = self.real2com(T7_img_rec.permute(0, 2, 3, 1))
        T8_img_rec = self.real2com(T8_img_rec.permute(0, 2, 3, 1))
        T9_img_rec = self.real2com(T9_img_rec.permute(0, 2, 3, 1))
        T10_img_rec = self.real2com(T10_img_rec.permute(0, 2, 3, 1))
        T11_img_rec = self.real2com(T11_img_rec.permute(0, 2, 3, 1))
        T12_img_rec = self.real2com(T12_img_rec.permute(0, 2, 3, 1))
        img = torch.stack((T1_img_rec,T2_img_rec,T3_img_rec,T4_img_rec,T5_img_rec,T6_img_rec,
                           T7_img_rec,T8_img_rec,T9_img_rec,T10_img_rec,T11_img_rec,T12_img_rec),1)
        return img ,S_map,R1_map,R2_map