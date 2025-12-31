"""
3D neural network model used in the accompanying qPACT study.

This implementation is based on the general design principles of the CS²-Net
architecture introduced by Lei Mou et al. (Medical Image Analysis, 2020),
but is **not** the original CS²-Net implementation.

Provenance:
  - Initial architectural ideas were adapted from CS²-Net–related code
    authored by Lei Mou.
  - The model has been substantially modified and repurposed for 3D
    quantitative photoacoustic computed tomography by Refik Mert Cam
    (PhD candidate, ECE, UIUC).

Notes:
  - The network follows a 3D U-Net–like encoder–decoder structure.
  - Channel/spatial attention mechanisms are retained conceptually.
  - Task-specific decoder heads are defined and used in the training script
    for joint regression and segmentation.

"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

def downsample():
    return nn.MaxPool3d(kernel_size=2, stride=2)


def deconv(in_channels, out_channels):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()


class ResEncoder3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResEncoder3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        #self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        #self.bn2 = nn.BatchNorm3d(out_channels)
        #self.relu = nn.ReLU(inplace=False)
        self.relu = nn.LeakyReLU(0.1, inplace=False)
        self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        #out = self.relu(self.bn1(self.conv1(x)))
        #out = self.relu(self.bn2(self.conv2(out)))
        out += residual
        out = self.relu(out)
        return out


class Decoder3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm3d(out_channels),
            #nn.ReLU(inplace=False),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm3d(out_channels),
            #nn.ReLU(inplace=False)
            nn.LeakyReLU(0.1, inplace=False)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class SpatialAttentionBlock3d(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionBlock3d, self).__init__()
        #self.query = nn.Conv3d(in_channels, in_channels // 16, kernel_size=(1, 3, 1), padding=(0, 1, 0))
        #self.key = nn.Conv3d(in_channels, in_channels // 16, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        #self.judge = nn.Conv3d(in_channels, in_channels // 16, kernel_size=(1, 1, 3), padding=(0, 0, 1))
        self.query = nn.Conv3d(in_channels, in_channels // 8, kernel_size=(1, 3, 1), padding=(0, 1, 0))
        self.key = nn.Conv3d(in_channels, in_channels // 8, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.judge = nn.Conv3d(in_channels, in_channels // 8, kernel_size=(1, 1, 3), padding=(0, 0, 1))
        self.value = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxWxZ )
        :return: affinity value + x
        B: batch size
        C: channels
        H: height
        W: width
        D: slice number (depth)
        """
        B, C, H, W, D = x.size()
        # compress x: [B,C,H,W,Z]-->[B,H*W*Z,C], make a matrix transpose
        proj_query = self.query(x).view(B, -1, W * H * D).permute(0, 2, 1)  # -> [B,W*H*D,C]
        proj_key = self.key(x).view(B, -1, W * H * D)  # -> [B,H*W*D,C]
        proj_judge = self.judge(x).view(B, -1, W * H * D).permute(0, 2, 1)  # -> [B,C,H*W*D]

        affinity1 = torch.matmul(proj_query, proj_key)
        affinity2 = torch.matmul(proj_judge, proj_key)
        affinity = torch.matmul(affinity1, affinity2)
        affinity = self.softmax(affinity)

        proj_value = self.value(x).view(B, -1, H * W * D)  # -> C*N
        weights = torch.matmul(proj_value, affinity)
        weights = weights.view(B, C, H, W, D)
        out = self.gamma * weights + x
        return out


class ChannelAttentionBlock3d(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionBlock3d, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxWxD )
        :return: affinity value + x
        """
        B, C, H, W, D = x.size()
        proj_query = x.view(B, C, -1).permute(0, 2, 1)
        proj_key = x.view(B, C, -1)
        proj_judge = x.view(B, C, -1).permute(0, 2, 1)
        affinity1 = torch.matmul(proj_key, proj_query)
        affinity2 = torch.matmul(proj_key, proj_judge)
        affinity = torch.matmul(affinity1, affinity2)
        affinity_new = torch.max(affinity, -1, keepdim=True)[0].expand_as(affinity) - affinity
        affinity_new = self.softmax(affinity_new)
        proj_value = x.view(B, C, -1)
        weights = torch.matmul(affinity_new, proj_value)
        weights = weights.view(B, C, H, W, D)
        out = self.gamma * weights + x
        return out


class AffinityAttention3d(nn.Module):
    """ Affinity attention module """

    def __init__(self, in_channels):
        super(AffinityAttention3d, self).__init__()
        self.sab = SpatialAttentionBlock3d(in_channels)
        self.cab = ChannelAttentionBlock3d(in_channels)
        # self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x):
        """
        sab: spatial attention block
        cab: channel attention block
        :param x: input tensor
        :return: sab + cab
        """
        sab = self.sab(x)
        cab = self.cab(x)
        out = sab + cab + x
        return out



class CSNet3D(nn.Module):
    def __init__(self, classes, channels):
        """
        :param classes: the object classes number.
        :param channels: the channels of the input image.
        """
        super(CSNet3D, self).__init__()
        self.enc_input = ResEncoder3d(channels, 8)
        self.encoder1 = ResEncoder3d(8, 16)
        self.encoder2 = ResEncoder3d(16, 32)
        self.encoder3 = ResEncoder3d(32, 64)
        self.encoder4 = ResEncoder3d(64, 128)
        self.downsample = downsample()
        self.affinity_attention = AffinityAttention3d(128)
        self.attention_fuse = nn.Conv3d(128 * 2, 128, kernel_size=1)
        
        # Segmentation decoder
        self.decoder4_seg = Decoder3d(128, 64)
        self.decoder3_seg = Decoder3d(64, 32)
        self.decoder2_seg = Decoder3d(32, 16)
        self.decoder1_seg = Decoder3d(16, 8)
        self.deconv4_seg = deconv(128, 64)
        self.deconv3_seg = deconv(64, 32)
        self.deconv2_seg = deconv(32, 16)
        self.deconv1_seg = deconv(16, 8)
        self.final_seg = nn.Conv3d(8, classes, kernel_size=1)

        # Regression decoder
        self.decoder4_reg = Decoder3d(128, 64)
        self.decoder3_reg = Decoder3d(64, 32)
        self.decoder2_reg = Decoder3d(32, 16)
        self.decoder1_reg = Decoder3d(16, 8)
        self.deconv4_reg = deconv(128, 64)
        self.deconv3_reg = deconv(64, 32)
        self.deconv2_reg = deconv(32, 16)
        self.deconv1_reg = deconv(16, 8)
        self.final_reg = nn.Conv3d(8, 1, kernel_size=1)  # Assuming regression output is a single channel

        initialize_weights(self)

    def forward(self, x):
        enc_input = self.enc_input(x)
        down1 = self.downsample(enc_input)

        enc1 = self.encoder1(down1)
        down2 = self.downsample(enc1)

        enc2 = self.encoder2(down2)
        down3 = self.downsample(enc2)

        enc3 = self.encoder3(down3)
        down4 = self.downsample(enc3)

        input_feature = self.encoder4(down4)

        # Do Attention operations here
        attention = self.affinity_attention(input_feature)
        attention_fuse = input_feature + attention

        # Segmentation decoder operations
        up4_seg = self.deconv4_seg(attention_fuse)
        up4_seg = torch.cat((enc3, up4_seg), dim=1)
        dec4_seg = self.decoder4_seg(up4_seg)

        up3_seg = self.deconv3_seg(dec4_seg)
        up3_seg = torch.cat((enc2, up3_seg), dim=1)
        dec3_seg = self.decoder3_seg(up3_seg)

        up2_seg = self.deconv2_seg(dec3_seg)
        up2_seg = torch.cat((enc1, up2_seg), dim=1)
        dec2_seg = self.decoder2_seg(up2_seg)

        up1_seg = self.deconv1_seg(dec2_seg)
        up1_seg = torch.cat((enc_input, up1_seg), dim=1)
        dec1_seg = self.decoder1_seg(up1_seg)

        final_seg = self.final_seg(dec1_seg)
        final_seg = torch.sigmoid(final_seg)

        # Regression decoder operations
        up4_reg = self.deconv4_reg(attention_fuse)
        up4_reg = torch.cat((enc3, up4_reg), dim=1)
        dec4_reg = self.decoder4_reg(up4_reg)

        up3_reg = self.deconv3_reg(dec4_reg)
        up3_reg = torch.cat((enc2, up3_reg), dim=1)
        dec3_reg = self.decoder3_reg(up3_reg)

        up2_reg = self.deconv2_reg(dec3_reg)
        up2_reg = torch.cat((enc1, up2_reg), dim=1)
        dec2_reg = self.decoder2_reg(up2_reg)

        up1_reg = self.deconv1_reg(dec2_reg)
        up1_reg = torch.cat((enc_input, up1_reg), dim=1)
        dec1_reg = self.decoder1_reg(up1_reg)

        final_reg = self.final_reg(dec1_reg)
        final_reg = torch.sigmoid(final_reg)  # or use another activation function as needed

        return final_seg, final_reg
