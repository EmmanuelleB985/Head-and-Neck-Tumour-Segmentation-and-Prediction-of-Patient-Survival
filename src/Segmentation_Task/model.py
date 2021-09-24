import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from layers import ConvBlock3d, FastSmoothSeNormConv3d, RESseNormConv3d , ChannelSpatialSELayer3D, init_weights, unetConv3_x, unetConv3_k


class NormResSEUNet_3Plus(nn.Module):

    '''
    Inspired from:
    -UNet 3+ architecture with full scale inter and intra-skip connections with ground-truth supervision defined in:
    Huang H. et al.(2020). UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation. 1055-1059. 10.1109/ICASSP40776.2020.9053405. 
    https://arxiv.org/abs/2004.08790
    - 3D normalised squeeze and excitation blocks defined in:
    Iantsen A., Visvikis D., Hatt M. (2021) Squeeze-and-Excitation Normalization for Automated Delineation of Head and Neck Primary Tumors in Combined PET and CT Images. 
    In: Andrearczyk V., Oreiller V., Depeursinge A. (eds) Head and Neck Tumor Segmentation. HECKTOR 2020. Lecture Notes in Computer Science, vol 12603. Springer, Cham.
    https://doi.org/10.1007/978-3-030-67194-5_4
    '''

    def __init__(self, in_channels=2, n_classes=1, feature_scale=4, reduction=2, is_deconv=True, is_batchnorm=True):
        super(NormResSEUNet_3Plus, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [24,48,96,192,384] 

        ## -------------Encoder--------------
        self.block_1_1_left = RESseNormConv3d(self.in_channels, filters[0], reduction, kernel_size=7, stride=1, padding=3)
        self.block_1_2_left = RESseNormConv3d(filters[0], filters[0], reduction, kernel_size=3, stride=1, padding=1)

        self.pool_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block_2_1_left = RESseNormConv3d(filters[0], filters[1], reduction, kernel_size=3, stride=1, padding=1)
        self.block_2_2_left = RESseNormConv3d(filters[1], filters[1], reduction, kernel_size=3, stride=1, padding=1)
        self.block_2_3_left = RESseNormConv3d(filters[1], filters[1], reduction, kernel_size=3, stride=1, padding=1)

        self.pool_2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block_3_1_left = RESseNormConv3d(filters[1], filters[2], reduction, kernel_size=3, stride=1, padding=1)
        self.block_3_2_left = RESseNormConv3d(filters[2], filters[2], reduction, kernel_size=3, stride=1, padding=1)
        self.block_3_3_left = RESseNormConv3d(filters[2], filters[2], reduction, kernel_size=3, stride=1, padding=1)

        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block_4_1_left = RESseNormConv3d(filters[2], filters[3], reduction, kernel_size=3, stride=1, padding=1)
        self.block_4_2_left = RESseNormConv3d(filters[3], filters[3], reduction, kernel_size=3, stride=1, padding=1)
        self.block_4_3_left = RESseNormConv3d(filters[3], filters[3], reduction, kernel_size=3, stride=1, padding=1)

        self.pool_4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block_5_1_left = RESseNormConv3d(filters[3], filters[4], reduction, kernel_size=3, stride=1, padding=1)
        self.block_5_2_left = RESseNormConv3d(filters[4], filters[4], reduction, kernel_size=3, stride=1, padding=1)
        self.block_5_3_left = RESseNormConv3d(filters[4], filters[4], reduction, kernel_size=3, stride=1, padding=1)

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 6
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        self.h1_PT_hd4 = nn.MaxPool3d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = FastSmoothSeNormConv3d(filters[0], self.CatChannels, reduction, kernel_size=3, stride=1, padding=1)


        self.h2_PT_hd4 = nn.MaxPool3d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv =  FastSmoothSeNormConv3d(filters[1], self.CatChannels, reduction, kernel_size=3, stride=1, padding=1)

        self.h3_PT_hd4 = nn.MaxPool3d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = FastSmoothSeNormConv3d(filters[2], self.CatChannels, reduction, kernel_size=3, stride=1, padding=1)


        self.h4_Cat_hd4_conv = FastSmoothSeNormConv3d(filters[3], self.CatChannels, reduction, kernel_size=3, stride=1, padding=1)
  

        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='trilinear')  # 14*14
        self.hd5_UT_hd4_conv = FastSmoothSeNormConv3d(filters[4], self.CatChannels, reduction, kernel_size=3, stride=1, padding=1)
      

        self.conv4d_1 = FastSmoothSeNormConv3d(self.UpChannels, self.UpChannels, reduction, kernel_size=3, stride=1, padding=1)
    

        '''stage 3d'''
        self.h1_PT_hd3 = nn.MaxPool3d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = FastSmoothSeNormConv3d(filters[0], self.CatChannels, reduction, kernel_size=3, stride=1, padding=1)


        self.h2_PT_hd3 = nn.MaxPool3d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = FastSmoothSeNormConv3d(filters[1], self.CatChannels, reduction, kernel_size=3, stride=1, padding=1)


        self.h3_Cat_hd3_conv = FastSmoothSeNormConv3d(filters[2], self.CatChannels, reduction, kernel_size=3, stride=1, padding=1)
  

        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='trilinear')  
        self.hd4_UT_hd3_conv = FastSmoothSeNormConv3d(self.UpChannels, self.CatChannels, reduction, kernel_size=3, stride=1, padding=1)
        

        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='trilinear')  
        self.hd5_UT_hd3_conv = FastSmoothSeNormConv3d(filters[4], self.CatChannels, reduction, kernel_size=3, stride=1, padding=1)
    

        self.conv3d_1 = FastSmoothSeNormConv3d(self.UpChannels, self.UpChannels, reduction, kernel_size=3, stride=1, padding=1)
        

        '''stage 2d '''
        self.h1_PT_hd2 = nn.MaxPool3d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = FastSmoothSeNormConv3d(filters[0], self.CatChannels, reduction, kernel_size=3, stride=1, padding=1)
   
        self.h2_Cat_hd2_conv = FastSmoothSeNormConv3d(filters[1], self.CatChannels, reduction, kernel_size=3, stride=1, padding=1)
     

        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='trilinear')  
        self.hd3_UT_hd2_conv = FastSmoothSeNormConv3d(self.UpChannels, self.CatChannels, reduction, kernel_size=3, stride=1, padding=1)


        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='trilinear')  
        self.hd4_UT_hd2_conv = FastSmoothSeNormConv3d(self.UpChannels, self.CatChannels, reduction, kernel_size=3, stride=1, padding=1)
       

        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='trilinear')  
        self.hd5_UT_hd2_conv = FastSmoothSeNormConv3d(filters[4], self.CatChannels, reduction, kernel_size=3, stride=1, padding=1)


        self.conv2d_1 = FastSmoothSeNormConv3d(self.UpChannels, self.UpChannels, reduction, kernel_size=3, stride=1, padding=1)
        

        '''stage 1d'''
        self.h1_Cat_hd1_conv = FastSmoothSeNormConv3d(filters[0], self.CatChannels, reduction, kernel_size=3, stride=1, padding=1)


        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='trilinear')  
        self.hd2_UT_hd1_conv = FastSmoothSeNormConv3d(self.UpChannels, self.CatChannels, reduction, kernel_size=3, stride=1, padding=1)
      
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='trilinear')  
        self.hd3_UT_hd1_conv = FastSmoothSeNormConv3d(self.UpChannels, self.CatChannels, reduction, kernel_size=3, stride=1, padding=1)
      

        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='trilinear')  
        self.hd4_UT_hd1_conv = FastSmoothSeNormConv3d(self.UpChannels, self.CatChannels, reduction, kernel_size=3, stride=1, padding=1)
     
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='trilinear')  
        self.hd5_UT_hd1_conv = FastSmoothSeNormConv3d(filters[4], self.CatChannels, reduction, kernel_size=3, stride=1, padding=1)
        
        self.conv1d_1 = FastSmoothSeNormConv3d(self.UpChannels, self.UpChannels, reduction, kernel_size=3, stride=1, padding=1)
      
        # -------------Trilinear Upsampling--------------
        self.upscore6 = nn.Upsample(scale_factor=32,mode='trilinear')
        self.upscore5 = nn.Upsample(scale_factor=16,mode='trilinear')
        self.upscore4 = nn.Upsample(scale_factor=8,mode='trilinear')
        self.upscore3 = nn.Upsample(scale_factor=4,mode='trilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='trilinear')


        # DeepSup
        self.outconv1 = nn.Conv3d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv2 = nn.Conv3d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv3 = nn.Conv3d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv4 = nn.Conv3d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv5 = nn.Conv3d(filters[4], n_classes, 3, padding=1)

        self.conv1x1 = nn.Conv3d(self.UpChannels, n_classes, kernel_size=1, stride=1, padding=0)



    def forward(self, x):
        ## -------------Encoder-------------
        h1 = self.block_1_2_left(self.block_1_1_left(x))
        h2 = self.pool_1(h1)
        h2 = self.block_2_3_left(self.block_2_2_left(self.block_2_1_left(h2)))
        h3 = self.pool_2(h2)
        h3 = self.block_3_3_left(self.block_3_2_left(self.block_3_1_left(h3)))
        h4 = self.pool_3(h3)
        h4 = self.block_4_3_left(self.block_4_2_left(self.block_4_1_left(h4)))
        h5 = self.pool_4(h4)
        hd5 = self.block_5_3_left(self.block_5_2_left(self.block_5_1_left(h5)))

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))
        h2_PT_hd4 = self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))
        h3_PT_hd4 = self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))
        h4_Cat_hd4 = self.h4_Cat_hd4_conv(h4)
        hd5_UT_hd4 = self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))
        hd4 = self.conv4d_1(torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1)) 

        h1_PT_hd3 = self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))
        h2_PT_hd3 = self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))
        h3_Cat_hd3 = self.h3_Cat_hd3_conv(h3)
        hd4_UT_hd3 = self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))
        hd5_UT_hd3 = self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))
        hd3 = self.conv3d_1(torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1)) 

        h1_PT_hd2 = self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))
        h2_Cat_hd2 = self.h2_Cat_hd2_conv(h2)
        hd3_UT_hd2 = self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))
        hd4_UT_hd2 = self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))
        hd5_UT_hd2 = self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))
        hd2 = self.conv2d_1(torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1)) 

        h1_Cat_hd1 = self.h1_Cat_hd1_conv(h1)
        hd2_UT_hd1 = self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))
        hd3_UT_hd1 = self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))
        hd4_UT_hd1 = self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))
        hd5_UT_hd1 = self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))
        hd1 = self.conv1d_1(torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1)) 

        d5 = self.outconv5(hd5)
        d5 = self.upscore5(d5) 

        d4 = self.outconv4(hd4)
        d4 = self.upscore4(d4) 

        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3) 

        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2) 

        d1 = self.conv1x1(hd1) 


        return F.sigmoid(d1) 
