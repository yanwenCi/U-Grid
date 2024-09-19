import torch.nn as nn
import torch
import torch.nn.functional as nnf
import src.model.layers as layers
from src.model.networks.VoxelMorph import Stage
import torchvision.models as models


class ICNet(nn.Module):
    '''implicit correspondence network'''
    def __init__(self, config):
        super(ICNet, self).__init__()
        self.config = config
        self.img_enc = ImageEncoder(config)
        self.adapter = Adapter(config)
        self.grid_transformer = GridTrasformer(config)

    def forward(self, x, grid):
        '''
        grid --> [batch, 3, h, w, z]
        '''
        enc_feature = self.img_enc(x)
        adapted_feature = self.adapter(enc_feature, grid)
        gdf = self.grid_transformer(adapted_feature)#.transpose(2,1)  # [b, c, N], N=(h*w*z)
        return gdf.reshape(grid.shape)  # Grid displacement field

class ImageEncoder(nn.Module):
    def __init__(self, config):
        super(ImageEncoder, self).__init__()
        self.input_shape = config.input_shape

        nc = [8*(2**i) for i in range(5)]#[16,32,32,32]#
        self.downsample_block0 = layers.DownsampleBlock(inc=2, outc=nc[0])
        self.downsample_block1 = layers.DownsampleBlock(inc=nc[0], outc=nc[1])
        self.downsample_block2 = layers.DownsampleBlock(inc=nc[1], outc=nc[2])
        self.downsample_block3 = layers.DownsampleBlock(inc=nc[2], outc=nc[3])
        self.downsample_block4 = layers.DownsampleBlock(inc=nc[3], outc=nc[4])
        self.adpt_pool = nn.AdaptiveAvgPool3d((2, 2, 2))
        #self.fclayer = nn.Sequential(nn.Linear(6144, 2048), nn.Tanh())
       
    def forward(self, x):
        f_down0, _ = self.downsample_block0(x)
        f_down1, _ = self.downsample_block1(f_down0)
        f_down2, _ = self.downsample_block2(f_down1)
        f_down3, _ = self.downsample_block3(f_down2)
        f_down4, _ = self.downsample_block4(f_down3)
        out = self.adpt_pool(f_down4)
        out=out.reshape([out.shape[0], -1])
        #out = self.fclayer(f_down4.reshape([f_down4.shape[0], -1]))
        return  out # squeeze but preserve the batch dimension.


class ImageEncoderRes(nn.Module):
    def __init__(self, config):
        super(ImageEncoderRes, self).__init__()
        self.model = models.video.r3d_18(pretrained=True)
        # Freeze the parameters of the pretrained layers
        self.model.stem[0] = nn.Conv3d(2, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        
        # for param in self.model.parameters():
        #     param.requires_grad = False
        #self.fclayer = nn.Sequential(nn.Linear(512, 2048), nn.ReLU())
        self.model.fc = nn.Sequential(nn.Linear(512, 2048), nn.ReLU())
        print(self.model)
        # Enable gradient calculation for the modified last layer
        # for param in self.model.fc.parameters():
        #     param.requires_grad = True

    def forward(self, x):
        out = self.model(x)  
        return out#out.view(out.shape[0], -1) # squeeze but preserve the batch dimension.
        

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



class ImageEncoder4(nn.Module):
    def __init__(self, config):
        super(ImageEncoder4, self).__init__()
        self.input_shape = config.input_shape
        #nc = [16, 32, 32, 64]
        nc = [ 32, 64, 128, 256, 512]
        pre_nc = 2
        self.encoder = nn.ModuleList()
        for i in range(len(nc)):
            self.encoder.append(Stage(in_channels=pre_nc, out_channels=nc[i], stride=2, dropout=True, bnorm=True))
            pre_nc = nc[i]   
        #self.encoder.append(nn.AdaptiveAvgPool3d((2,2,2)))
        #ft_size = round(128/(2**len(nc)))
    
    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        return x.reshape([x.shape[0], -1])  # squeeze but preserve the batch dimension.


class ImageEncoder1(nn.Module):
    def __init__(self, config):
        super(ImageEncoder1, self).__init__()
        self.input_shape = config.input_shape
        #nc = [16, 32, 32, 64]
        nc = [8, 16, 32, 64, 128]
        pre_nc = 2
        self.encoder = nn.ModuleList()
        for i in range(len(nc)):
            self.encoder.append(Stage(in_channels=pre_nc, out_channels=nc[i], stride=2, dropout=True, bnorm=True))
            pre_nc = nc[i]   
        #self.encoder.append(nn.AdaptiveAvgPool3d((2,2,2)))
        #ft_size = round(128/(2**len(nc)))
        self.fclayer = nn.Sequential(nn.Linear(6144, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.Tanh())

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        x = self.fclayer(x.reshape([x.shape[0], -1]))
        return x#x.reshape([x.shape[0], -1])  # squeeze but preserve the batch dimension.

class ImageEncoder0(nn.Module):
    def __init__(self, config):
        super(ImageEncoder0, self).__init__()
        self.input_shape = config.input_shape
        #nc = [16, 32, 32, 64]
        nc = [32, 64, 128, 256]
        pre_nc = 2
        self.encoder = nn.ModuleList()
        self.bottlenecks = nn.ModuleList()
        
        for i in range(len(nc)):
            self.encoder.append(Stage(in_channels=pre_nc, out_channels=nc[i], stride=2, dropout=True, bnorm=True))
            pre_nc = nc[i]   
            self.bottlenecks.append(nn.Sequential(nn.Conv3d(in_channels=nc[i], out_channels=16, kernel_size=1), nn.ReLU(), nn.AdaptiveAvgPool3d((2,2,2))))
        self.avgp = nn.AdaptiveAvgPool3d((2,2,2))
        #self.fclayers = nn.Sequential(nn.Linear(nc[-1]*512, 2048))
    def forward(self, x):
        self.addx = []
        for layer, bottle in zip(self.encoder, self.bottlenecks):
            x = layer(x)
            self.addx.append(bottle(x))
        x = self.avgp(x)
        out1, out2, out3, out4 = [i.view(i.shape[0], -1) for i in self.addx]
        return out1, out2, out3, out4, x.view(x.shape[0], -1)  # squeeze but preserve the batch dimension.

class ImageEncoder3(nn.Module):
    def __init__(self, config):
        super(ImageEncoder3, self).__init__()
        self.input_shape = config.input_shape
        #nc = [16, 32, 32, 64]
        nc = [32, 64, 128, 256]
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
        enc_out --> [b, L] --> [b, c, L]
        grid    --> [b, 3, h, w, z] --> [b, 3, h*w*z]n--> [b, h*w*z, 3]
        '''
       
        enc_out = torch.stack([enc_out]*self.h*self.w*self.z, dim=1)
        grid = torch.reshape(grid, [grid.shape[0], grid.shape[1], -1])
        grid = torch.transpose(grid, 2, 1)
        grid_feats = torch.transpose(torch.cat([enc_out, grid], dim=2), 1, 2)
        return torch.transpose(torch.cat([enc_out, grid], dim=2), 1, 2)  # [batch, feature_len, number]

class Adapter0(nn.Module):
    '''a network module to adapte 3d tensors to 1d tensors '''
    def __init__(self, config):
        super(Adapter0, self).__init__()
        self.grid_size = config.grid_size
        self.h, self.w, self.z = self.grid_size

    def forward(self, enc_outs, grid):
        '''
        enc_out --> [b, L] --> [b, c, L]
        grid    --> [b, 3, h, w, z] --> [b, 3, h*w*z]n--> [b, h*w*z, 3]
        '''
        grid = torch.reshape(grid, [grid.shape[0], grid.shape[1], -1])
        grid = torch.transpose(grid, 2, 1)
        grid_feats = []
        for enc_out in enc_outs[:-1]:
            enc_out = torch.stack([enc_out]*self.h*self.w*self.z, dim=1)            
            grid_feats.append(torch.transpose(enc_out, 1, 2))

        enc_out = torch.stack([enc_outs[-1]]*self.h*self.w*self.z, dim=1)
        grid_feats.append(torch.transpose(torch.cat([enc_out, grid], dim=2), 1, 2))
        
        #print(enc_out.shape, grid.shape)
        return grid_feats  # [batch, feature_len, number]


class GridTrasformer(nn.Module):
    '''transform the grid via image feature'''
    def __init__(self, config):
        super(GridTrasformer, self).__init__()
        #nc = [240*8, 512, 256, 128, 64, 3]
        nc = [1024, 512, 256, 128, 64, 3]
        self.conv1 = nn.Conv1d(in_channels=nc[0]+3, out_channels=nc[1], kernel_size=1)
        self.actv1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv2 = nn.Conv1d(in_channels=nc[1], out_channels=nc[2], kernel_size=1)
        self.actv2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv3 = nn.Conv1d(in_channels=nc[2], out_channels=nc[3], kernel_size=1)
        self.actv3 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv4 = nn.Conv1d(in_channels=nc[3], out_channels=nc[4], kernel_size=1)
        self.actv4 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv5 = nn.Conv1d(in_channels=nc[4], out_channels=nc[5], kernel_size=1)
        self.actv5 = nn.Tanh()

    def forward(self, x):
        '''
        x --> [b, 1027, 1000]
        out --> [b,3,1000] #[b, 1000, 3]
        '''
        x = self.actv1(self.conv1(x))
        x = self.actv2(self.conv2(x))
        x = self.actv3(self.conv3(x))
        x = self.actv4(self.conv4(x))
        x = self.actv5(self.conv5(x))
        return x#torch.transpose(x, 1, 2) 
    
class GridTrasformer0(nn.Module):
    '''transform the grid via image feature'''
    def __init__(self, config, nc):
        super(GridTrasformer0, self).__init__()
        #nc = [240*8, 512, 256, 128, 64, 3]
        self.conv1 = nn.Conv1d(in_channels=256*8+3, out_channels=nc[-1], kernel_size=1)
        self.actv1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv2 = nn.Conv1d(in_channels=nc[-1]+128, out_channels=nc[-2], kernel_size=1)
        self.actv2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv3 = nn.Conv1d(in_channels=nc[-2]+128, out_channels=nc[-3], kernel_size=1)
        self.actv3 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv4 = nn.Conv1d(in_channels=nc[-3]+128, out_channels=nc[-4], kernel_size=1)
        self.actv4 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv5 = nn.Conv1d(in_channels=nc[-4], out_channels=3, kernel_size=1)
        self.actv5 = nn.Tanh()

    def forward(self, enc_feats):
        '''
        x --> [b, 1027, 1000]
        out --> [b,3,1000] #[b, 1000, 3]
        '''
        #print(enc_feats[-1].shape, enc_feats[-2].shape, enc_feats[-3].shape, enc_feats[-4].shape)
        x = self.actv1(self.conv1(enc_feats[-1]))
        x = self.actv2(self.conv2(torch.cat([x, enc_feats[-2]], dim=1)))
        x = self.actv3(self.conv3(torch.cat([x, enc_feats[-3]], dim=1)))
        x = self.actv4(self.conv4(torch.cat([x, enc_feats[-4]], dim=1)))
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

# class GridTrasformer(nn.Module):
#     '''transform the grid via image feature'''
#     def __init__(self, config):
#         super(GridTrasformer, self).__init__()
#         nc = [256, 32, 32, 32, 16, 3]
        
#         self.conv1 = nn.Linear(nc[0]+3, nc[1])
#         self.actv1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

#         self.conv2 = nn.Linear(nc[1], nc[2])
#         self.actv2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

#         self.conv3 = nn.Linear(nc[2], nc[3])
#         self.actv3 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

#         self.conv4 = nn.Linear(nc[3], nc[4])
#         self.actv4 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

#         self.conv5 = nn.Linear(nc[4], nc[5])
#         self.actv5 = nn.Tanh()

#     def forward(self, x):
#         '''
#         x --> [b, 1027, 1000]
#         out --> [b,3,1000] #[b, 1000, 3]
#         '''
#         x = x.transpose(1,2)
#         x = self.actv1(self.conv1(x))
#         x = self.actv2(self.conv2(x))
#         x = self.actv3(self.conv3(x))
#         x = self.actv4(self.conv4(x))
#         x = self.actv5(self.conv5(x))
#         return torch.transpose(x, 1, 2) 

# class GridTrasformer(nn.Module):
#     '''transform the grid via image feature'''
#     def __init__(self, config):
#         super(GridTrasformer, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=1027, out_channels=512, kernel_size=1)
#         self.actv1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

#         self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
#         self.actv2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

#         self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1)
#         self.actv3 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

#         self.conv4 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1)
#         self.actv4 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

#         self.conv5 = nn.Conv1d(in_channels=64, out_channels=3, kernel_size=1)
#         self.actv5 = nn.Tanh()

#     def forward(self, x):
#         '''
#         x --> [b, 1027, 1000]
#         out --> [b,3,1000] #[b, 1000, 3]
#         '''
#         x = self.actv1(self.conv1(x))
#         x = self.actv2(self.conv2(x))
#         x = self.actv3(self.conv3(x))
#         x = self.actv4(self.conv4(x))
#         x = self.actv5(self.conv5(x))

#         return x#torch.transpose(x, 1, 2) 



