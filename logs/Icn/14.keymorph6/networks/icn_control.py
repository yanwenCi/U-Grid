import torch.nn as nn
import torch
import torch.nn.functional as nnf
import src.model.layers as layers
from src.model.networks.VoxelMorph import Stage
import torchvision.models as models
from src.model.networks.local import ResidualBlock
import numpy as np
from src.model.functions import get_reference_grid3d

class ICNet(nn.Module):
    '''implicit correspondence network'''
    def __init__(self, config):
        super(ICNet, self).__init__()
        self.config = config
        self.img_enc = ImageEncoder(config)
        self.control_point_net = ControlPointNet(config)
        
        self.grid_transformer = GridTrasformer(config)

    def forward(self, x):
        '''
        grid --> [batch, 3, h, w, z]
        '''
        enc_feature = self.img_enc(x)
        # enc_feature2 = self.img_enc(x[:, 1:2, ...])
        adapted_feature, grid = self.control_point_net(enc_feature, x)
        gdf = self.grid_transformer(torch.cat((adapted_feature, grid), dim=1))  # [b, c, N], N=(h*w*z)
        # shape = [gdf.shape[0], gdf.shape[1],]+ list(self.config.grid_size)
        shape = [gdf.shape[0], gdf.shape[1],] + list(enc_feature.size()[2:])
        return gdf.reshape(shape), grid.reshape(shape) # Grid displacement field

class ControlPointNet_fail(nn.Module):
    #output grid size: [batch,grid_number, 3]
    def __init__(self, config):
        super(ControlPointNet_fail, self).__init__()
        self.config = config
        
        nc = [8*(2**i) for i in range(3)]#[8, 64, 512]
        self.layer0 = ResidualBlock(2, nc[0], stride=1)
        self.layer1 = ResidualBlock(nc[0], nc[1], stride=1)
        self.layer2 = ResidualBlock(nc[1], nc[2], stride=1)
        self.layer3 = nn.Sequential(nn.Conv3d(nc[2], 1, kernel_size=1, stride=1, padding=0, bias=True), nn.Sigmoid())
    
    
    def unravel_index(self, indices, shape):
        coords = []
        for dim_size in reversed(shape):
            coords.append(indices % dim_size)
            indices = indices // dim_size
        corrds=torch.stack(list(reversed(coords)), dim=-1)
        return corrds


    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        feats = self.layer3(x) 
        reshaped_feature_map = feats.view(feats.size(0), feats.size(1), -1)
        grid = get_reference_grid3d(x).view(x.size(0), 3, -1) # [batch, 3, grid_number]
        # Find the indices of the top 10 largest values for eac h batch element
        top_indices = torch.topk(reshaped_feature_map, k=1000, dim=-1).indices # [batch, 1, k]

        grid_select = []
        for i in range(top_indices.shape[0]):
            grid_select.append(grid[i, :, top_indices[i, 0,...]])
        grid = torch.stack(grid_select, dim=0) 
        return grid # [batch, 3, k]
 
class ControlPointNet_cat(nn.Module):
    #output grid size: [batch,grid_number, 3]
    def __init__(self, config):
        super(ControlPointNet_cat, self).__init__()
        self.config = config
        
        nc = [256, 256]#[8, 64, 512]
        self.nonlin = nn.Softmax(dim=1)
        # self.layer0 = nn.Sequential(ResidualBlock(nc[0], nc[1], stride=1), nn.Softmax(dim=1))
        # self.layer1 = ResidualBlock(nc[1], nc[2], stride=1)
        # self.layer3 = nn.Sequential(nn.Conv3d(nc[2], 1, kernel_size=1, stride=1, padding=0, bias=True), nn.Sigmoid())
        self.pool = nn.MaxPool3d(4, stride=4, return_indices=True)
        self.unpool = nn.MaxUnpool3d(4, 4)
        


    def forward(self, inp, inp2, img):
        self.adppool = nn.AdaptiveAvgPool3d(inp.shape[2:])
        control_p=[]
        b,c,w,h,d = inp.shape
        for x in [inp, inp2]:
            x = self.nonlin(x)
            feats_top, index_top = self.pool(x)
            control_p.append(self.unpool(feats_top, index_top))
        control_p = torch.cat(control_p, dim=1)
        #print("Is dense_tensor sparse?", torch.sum(control_p==0)/torch.numel(control_p))
       
        img = self.adppool(img)
        feats_cat = torch.cat([inp, inp2], dim=1)
        control_p = control_p * feats_cat
        control_p = torch.cat([control_p, img], dim=1)
        # import matplotlib.pyplot as plt
        # plt.subplot(1,2,1)
        # plt.imshow(control_p[0,0,..., 10].cpu().detach().numpy())
        # plt.subplot(1,2,2)
        # plt.imshow(inp[0,0,..., 10].cpu().detach().numpy())
        # plt.savefig("control_p.png")
        grid = get_reference_grid3d(inp).view(b, 3, -1) # [batch, 3, grid_number]
        control_p = control_p.view(b, c*2+2, -1) # [b, c, grid_number]
        
        # print(enc_out.shape, grid.shape)

        #reshaped_feature_map = feats.view(feats.size(0), feats.size(1), -1)
        #grid = get_reference_grid3d(feats_top).view(inp.size(0), 3, -1) # [batch, 3, grid_number]
        # Find the indices of the top 10 largest values for eac h batch element
        # top_indices = torch.topk(reshaped_feature_map, k=1000, dim=-1).indices # [batch, 1, k]
        # assert len(torch.stack([top_indices.squeeze(dim=1)]*3, dim=1).shape) == len(grid.shape)
        # grid_select = torch.gather(grid, 2, torch.stack([top_indices.squeeze(dim=1)]*3, dim=1))  # [batch, 3, k]
        # # Extract corresponding top values from the input tensor
        # xx = xx.view(xx.size(0), xx.size(1), -1)
        # top_value = xx.gather(2, torch.stack([top_indices.squeeze(dim=1)]*xx.size(1), dim=1))  # [batch, channels, k]
        return control_p, grid # [batch, 3, k]

class ControlPointNet(nn.Module):
    #output grid size: [batch,grid_number, 3]
    def __init__(self, config):
        super(ControlPointNet, self).__init__()
        self.config = config
        
        #nc = [256, 256]#[8, 64, 512]
        self.nonlin = nn.Softmax(dim=1)
        # self.layer0 = nn.Sequential(ResidualBlock(nc[0], nc[1], stride=1), nn.Softmax(dim=1))
        # self.layer1 = ResidualBlock(nc[1], nc[2], stride=1)
        # self.layer3 = nn.Sequential(nn.Conv3d(nc[2], 1, kernel_size=1, stride=1, padding=0, bias=True), nn.Sigmoid())
        self.maxpool = nn.MaxPool3d(4, stride=4, return_indices=True)
        #self.avgpool = nn.AvgPool3d(4, stride=4, return_indices=True)
        self.unpool = nn.MaxUnpool3d(4, 4)
        


    def forward(self, inp, img):
        self.adppool = nn.AdaptiveAvgPool3d(inp.shape[2:])
        control_p=[]
        b,c,w,h,d = inp.shape
       
        x = self.nonlin(inp)
        feats_top, index_top = self.maxpool(x)
        control_p = self.unpool(feats_top, index_top)
       
        img = self.adppool(img)
       
        control_p = control_p * inp
        control_p = torch.cat([control_p, img], dim=1)
        # import matplotlib.pyplot as plt
        # plt.subplot(1,2,1)
        # plt.imshow(control_p[0,0,..., 10].cpu().detach().numpy())
        # plt.subplot(1,2,2)
        # plt.imshow(inp[0,0,..., 10].cpu().detach().numpy())
        # plt.savefig("control_p.png")
        grid = get_reference_grid3d(inp).view(b, 3, -1) # [batch, 3, grid_number]
        control_p = control_p.view(b, c+2, -1) # [b, c, grid_number]
        
        # print(enc_out.shape, grid.shape)

        #reshaped_feature_map = feats.view(feats.size(0), feats.size(1), -1)
        #grid = get_reference_grid3d(feats_top).view(inp.size(0), 3, -1) # [batch, 3, grid_number]
        # Find the indices of the top 10 largest values for eac h batch element
        # top_indices = torch.topk(reshaped_feature_map, k=1000, dim=-1).indices # [batch, 1, k]
        # assert len(torch.stack([top_indices.squeeze(dim=1)]*3, dim=1).shape) == len(grid.shape)
        # grid_select = torch.gather(grid, 2, torch.stack([top_indices.squeeze(dim=1)]*3, dim=1))  # [batch, 3, k]
        # # Extract corresponding top values from the input tensor
        # xx = xx.view(xx.size(0), xx.size(1), -1)
        # top_value = xx.gather(2, torch.stack([top_indices.squeeze(dim=1)]*xx.size(1), dim=1))  # [batch, channels, k]
        return control_p, grid # [batch, 3, k]


class ImageEncoder(nn.Module):
    def __init__(self, config):
        super(ImageEncoder, self).__init__()
        self.input_shape = config.input_shape

        nc = [16,32,64, 128]#[8*(2**i) for i in range(5)]#
        self.downsample_block0 = layers.DownsampleBlock(inc=2, outc=nc[0])
        self.downsample_block1 = layers.DownsampleBlock(inc=nc[0], outc=nc[1])
        self.downsample_block2 = layers.DownsampleBlock(inc=nc[1]+2, outc=nc[2], down = False)
        self.downsample_block3 = layers.DownsampleBlock(inc=nc[2]+2, outc=nc[3], down = False)
        #self.downsample_block4 = layers.DownsampleBlock(inc=nc[3]+2, outc=nc[4], down = False)
        #self.adpt_pool = nn.AdaptiveAvgPool3d((2, 2, 2))
        #self.fclayer = nn.Sequential(nn.Linear(6144, 2048), nn.Tanh())
        self.maxpool = nn.AvgPool3d(kernel_size=4, stride=4)    
       
    def forward(self, x):
        f_down0, _ = self.downsample_block0(x)
        f_down1, _ = self.downsample_block1(f_down0)
        f_down1 = torch.cat([f_down1, self.maxpool(x)], dim=1)
        f_down2, _ = self.downsample_block2(f_down1)
        f_down2 = torch.cat([f_down2, self.maxpool(x)], dim=1)
        f_down3, _ = self.downsample_block3(f_down2)
        #f_down3 = torch.cat([f_down3, self.maxpool(x)], dim=1)
        #f_down4, _ = self.downsample_block4(f_down3)
        #out = self.adpt_pool(f_down4)
        #out=out.reshape([out.shape[0], -1])
        #out = self.fclayer(f_down4.reshape([f_down4.shape[0], -1]))
        return  f_down3 # squeeze but preserve the batch dimension.


class ImageEncoderAffine(nn.Module):
    def __init__(self, config):
        super(ImageEncoderAffine, self).__init__()
        self.input_shape = config.input_shape
        #nc = [16, 32, 32, 64]
        nc = [8, 16, 32, 64, 128]
        pre_nc = 2
        self.encoder = nn.ModuleList()
        for i in range(len(nc)):
            self.encoder.append(Stage(in_channels=pre_nc, out_channels=nc[i], stride=2, dropout=True, bnorm=True))
            pre_nc = nc[i]   
        self.encoder.append(nn.AdaptiveAvgPool3d((2,2,2)))

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        return x.reshape([x.shape[0], -1])  # squeeze but preserve the batch dimension.





class Adapter(nn.Module):
    '''a network module to adapte 3d tensors to 1d tensors '''
    def __init__(self, config):
        super(Adapter, self).__init__()
        self.grid_size = config.grid_size
        self.h, self.w, self.z = self.grid_size

    def forward(self, enc_out, grid):
        '''
        enc_out --> [b, L] --> [b, h*w*z, L]
        grid    --> [b, 3, h, w, z] --> [b, 3, h*w*z]n--> [b, h*w*z, 3]
        '''
        # grid: #[batch, 3, grid_number]
        #enc_out = torch.stack([enc_out]*self.h*self.w*self.z, dim=1) # enc_out:[ b, grid_number, c]
        grid = get_reference_grid3d(enc_out).view(enc_out.size(0), 3, -1) # [batch, 3, grid_number]
        enc_out = enc_out.view(enc_out.size(0), enc_out.size(1), -1) # [b, c, grid_number]
        grid = grid.view(grid.size(0), grid.size(1), -1) # [batch, 3, grid_number]
        # print(enc_out.shape, grid.shape)
        grid_feats = torch.cat([enc_out, grid], dim=1)
        return grid_feats # [batch, feature_len+3, grid_number]



class GridTrasformer(nn.Module):
    '''transform the grid via image feature'''
    def __init__(self, config):
        super(GridTrasformer, self).__init__()
        #nc = [240*8, 512, 256, 128, 64, 3]
        nc = [128+2, 3]#256, 128, 64, 32, 3]
        self.conv5 = nn.Conv1d(in_channels=nc[0]+3, out_channels=nc[1], kernel_size=1)
        # self.actv1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # self.conv2 = nn.Conv1d(in_channels=nc[1], out_channels=nc[2], kernel_size=1)
        # self.actv2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # self.conv3 = nn.Conv1d(in_channels=nc[2], out_channels=nc[3], kernel_size=1)
        # self.actv3 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # self.conv4 = nn.Conv1d(in_channels=nc[3], out_channels=nc[4], kernel_size=1)
        # self.actv4 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # self.conv5 = nn.Conv1d(in_channels=nc[4], out_channels=nc[5], kernel_size=1)
        self.actv5 = nn.Tanh()

    def forward(self, x):
        '''
        x --> [b, 1027, 1000]
        out --> [b,3,1000] #[b, 1000, 3]
        '''
        #x = self.actv1(self.conv1(x))
        # x = self.actv2(self.conv2(x))
        # x = self.actv3(self.conv3(x))
        # x = self.actv4(self.conv4(x))
        x = self.actv5(self.conv5(x))
        return x#torch.transpose(x, 1, 2) 
    


class AffineTransform(nn.Module):
    """
    3-D Affine Transformer
    """

    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, src, affine):

        mat = affine#torch.bmm(shear_mat, torch.bmm(scale_mat, torch.bmm(rot_mat_z, torch.matmul(rot_mat_y, rot_mat_x))))
        inv_mat = mat#torch.inverse(mat)
        grid = nnf.affine_grid(mat, src.size(), align_corners=True)
        #inv_grid = nnf.affine_grid(inv_mat, [src.shape[0], 3, src.shape[2], src.shape[3], src.shape[4]], align_corners=True)
        return nnf.grid_sample(src, grid, align_corners=True, mode=self.mode), mat, inv_mat





