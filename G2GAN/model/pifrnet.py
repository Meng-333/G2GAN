import argparse
import os
import numpy as np
import time
import math
import itertools
import sys
from collections import OrderedDict
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

# ########### skimage version 0.15.0 ###########
from skimage.measure import compare_ssim
from skimage.measure import compare_mse
from skimage.measure import compare_psnr

########### skimage version >2.0 ###########
# from skimage.metrics import structural_similarity as compare_ssim
# from skimage.metrics import mean_squared_error as compare_mse
# from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from torchvision.transforms import ToPILImage, ToTensor
from torchvision.models import vgg19

from utils import utils as srutils
from utils.logger import Logger, PrintLogger
from torch.nn import Parameter
from data.data import get_IR_trainDataset, get_IR_testDataset


# from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def weights_init_normal(m, mean=0.0, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('ConvTranspose2d') != -1:
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        vgg19_model = vgg19(pretrained=True)

        # Extracts features at the 11th layer
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:12])

    def forward(self, img):
        out = self.feature_extractor(img)
        return out


class EdgeLoss(nn.Module):
    def __init__(self):
        """
        Initialize the hybrid HSI super-resolution loss.

        Args:
            la1: Weight of the SAM spectral fidelity loss.
            la2: Weight of the gradient consistency loss.
            la3: Weight of the edge preservation loss.
            use_l1: If True, use L1 loss as the spatial reconstruction loss;
                    otherwise, use MSE loss.
        """
        super(EdgeLoss, self).__init__()

    @staticmethod
    def compute_edge_preservation(pred, target):
        """
        Compute the edge preservation loss.

        The loss is calculated by approximating spatial gradients with finite differences.
        """
        # Compute gradients of the predicted image
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]  # Difference along the x direction
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]  # Difference along the y direction

        # Pad the gradient maps to maintain the original tensor shape
        pred_dx = F.pad(pred_dx, (0, 1, 0, 0))
        pred_dy = F.pad(pred_dy, (0, 0, 0, 1))

        # Compute gradient magnitude as edge strength
        pred_edges = torch.sqrt(pred_dx.pow(2) + pred_dy.pow(2) + 1e-8)

        # Compute gradients of the target image
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]  # Difference along the x direction
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]  # Difference along the y direction

        # Pad the gradient maps to maintain the original tensor shape
        target_dx = F.pad(target_dx, (0, 1, 0, 0))
        target_dy = F.pad(target_dy, (0, 0, 0, 1))

        # Compute gradient magnitude as edge strength
        target_edges = torch.sqrt(target_dx.pow(2) + target_dy.pow(2) + 1e-8)

        # Compute the edge preservation loss
        return F.l1_loss(pred_edges, target_edges)

    def forward(self, pred, target):
        """
        Compute the total hybrid loss.

        Args:
            pred: Predicted HSI tensor with shape [B, C, H, W].
            target: Ground-truth HSI tensor with shape [B, C, H, W].

        Returns:
            Total loss value.
        """

        # Edge preservation loss
        edge_loss = self.compute_edge_preservation(pred, target)

        return edge_loss


class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))


class ResidualChannelAttentionModule(nn.Module):
    """
    残差通道注意力模块 (Residual Channel-Attention Module)
    对应论文公式 (1)-(3)
    """
    def __init__(self, channels):
        super(ResidualChannelAttentionModule, self).__init__()
        self.conv3x3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.channel_attention = ChannelAttention(channels)
        self.conv1x1_identity = nn.Conv2d(channels, channels, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 残差路径: Conv3x3 -> ChannelAttention
        res_att = self.channel_attention(self.conv3x3(x))
        # 恒等路径: Conv1x1
        identity = self.conv1x1_identity(x)
        # 相加 + ReLU
        out = self.relu(identity + res_att)
        return out


class InfraredDetailEnhancementModule(nn.Module):
    """
    红外细节增强模块 (IR Detail-Enhancement Module)
    对应论文公式 (4)-(5)
    使用 Laplacian 算子提取边缘细节
    """
    def __init__(self, channels):
        super(InfraredDetailEnhancementModule, self).__init__()
        # Laplacian 边缘提取卷积层
        self.laplacian_conv = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False
        )
        self._init_laplacian()
        self.conv3x3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def _init_laplacian(self):
        """使用 Laplacian 算子初始化边缘提取卷积核"""
        kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32)
        with torch.no_grad():
            self.laplacian_conv.weight.copy_(
                kernel.view(1, 1, 3, 3).repeat(self.laplacian_conv.out_channels, 1, 1, 1)
            )

    def forward(self, x):
        edge = self.laplacian_conv(x)
        edge_out = self.conv3x3(edge)
        out = self.relu(x + edge_out)
        return out


class FrequencyDomainRefinementModule(nn.Module):
    """
    频域精炼模块 (Frequency Domain-Refinement Module)
    对应论文公式 (6)-(7)
    使用 FFT 卷积路径作为残差路径

    关键：FFT 后的张量形状为 (B, C, H, W//2+1) 的复数张量
    对其做 1x1 卷积时，需要同时处理实部和虚部
    """

    def __init__(self, channels):
        super(FrequencyDomainRefinementModule, self).__init__()
        # 频域卷积：将实部和虚部作为独立的通道处理
        # 输入 2*channels (实部+虚部)，输出 2*channels
        self.conv1x1_fft = nn.Conv2d(channels * 2, channels * 2, 1)
        self.conv1x1_out = nn.Conv2d(channels, channels, 1)
        self.conv1x1_fusion = nn.Conv2d(channels * 2, channels, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        B, C, H, W = x.shape

        # ============ FFT 卷积路径 ============
        # rfft2: 输出形状 (B, C, H, W//2+1), 复数
        fft_x = torch.fft.rfft2(x, norm='ortho')

        # 将复数转为实数张量: (B, C, H, W//2+1) -> (B, C, H, W//2+1, 2)
        # 再转为 (B, 2*C, H, W//2+1) 以便用 Conv2d 处理
        fft_x_real = torch.view_as_real(fft_x)  # (B, C, H, Wf, 2)
        fft_x_real = fft_x_real.permute(0, 1, 4, 2, 3).reshape(B, C * 2, H, W // 2 + 1)

        # 频域 1x1 卷积
        fft_out = self.relu(self.conv1x1_fft(fft_x_real))  # (B, 2C, H, Wf)

        # 还原为复数: (B, 2C, H, Wf) -> (B, C, H, Wf, 2) -> complex
        fft_out = fft_out.reshape(B, C, 2, H, W // 2 + 1).permute(0, 1, 3, 4, 2)
        fft_out = torch.view_as_complex(fft_out.contiguous())  # (B, C, H, Wf)

        # IFFT: 恢复空间域，指定输出尺寸 s=(H, W)
        ifft_x = torch.fft.irfft2(fft_out, s=(H, W), norm='ortho')  # (B, C, H, W)
        fft_conv_out = self.conv1x1_out(ifft_x)

        # ============ 与输入拼接融合 ============
        out = self.relu(self.conv1x1_fusion(torch.cat([x, fft_conv_out], dim=1)))
        return out


class CascadeIRFeatureRefinementBlock(nn.Module):
    """
    级联红外特征精炼块 (Cascade IR Feature-Refinement Block)
    由三个模块级联组成：ResidualChannelAttention -> IRDetailEnhancement -> FrequencyDomainRefinement
    """
    def __init__(self, channels):
        super(CascadeIRFeatureRefinementBlock, self).__init__()
        self.rca = ResidualChannelAttentionModule(channels)
        self.irde = InfraredDetailEnhancementModule(channels)
        self.fdr = FrequencyDomainRefinementModule(channels)

    def forward(self, x):
        x = self.rca(x)
        x = self.irde(x)
        x = self.fdr(x)
        return x


class ViTBlock(nn.Module):   #(HCT) 模块
    """标准 Vision Transformer Block"""
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0):
        super(ViTBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        # x: (B, N, C)
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = residual + attn_out
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        return x


class HCTModule(nn.Module):
    """
    Hybrid CNN-Transformer 模块
    对应论文公式 (8)-(10)
    """
    def __init__(self, channels, num_heads=4):
        super(HCTModule, self).__init__()
        self.conv3x3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv1x1_local = nn.Conv2d(channels, channels, 1)
        self.vit = ViTBlock(channels, num_heads=num_heads)
        self.conv1x1_global = nn.Conv2d(channels, channels, 1)
        self.conv3x3_fusion = nn.Conv2d(channels * 2, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        B, C, H, W = x.shape
        # 局部编码
        local = self.relu(self.conv1x1_local(self.conv3x3(x)))
        # 全局注意力 (ViT)
        local_flat = local.flatten(2).transpose(1, 2)  # (B, HW, C)
        global_flat = self.vit(local_flat)
        global_out = global_flat.transpose(1, 2).view(B, C, H, W)
        global_out = self.conv1x1_global(global_out)
        # 融合
        out = self.relu(self.conv3x3_fusion(torch.cat([x, global_out], dim=1)))
        return out


class EfficientFeatureFusionBlock(nn.Module):
    """高效特征融合块：拼接 -> 3x3 Conv -> ReLU"""
    def __init__(self, channels):
        super(EfficientFeatureFusionBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class GeneratorResNet(nn.Module):
    def __init__(
            self,
            in_channels=3,
            out_channels=3,
            n_blocks=4,
            base_channels=32,
            upscale_factor=8,
            num_heads=4,
            ga_mode='CA',
            addconv=False,
            norm_type=None,
    ):
        super(GeneratorResNet, self).__init__()

        self.upscale_factor = upscale_factor
        self.n_blocks = n_blocks
        self.ga_mode = ga_mode
        self.addconv = addconv

        # ============================================================
        # 1. 浅层特征提取
        # ============================================================
        self.shallow_channels = 16
        self.shallow_conv = nn.Conv2d(in_channels, self.shallow_channels, 3, padding=1)
        self.shallow_relu = nn.ReLU(inplace=True)
        self.shallow_to_deep = nn.Conv2d(
            self.shallow_channels, base_channels, 3, padding=1
        )

        # ============================================================
        # 2. 深层特征提取
        # ============================================================
        # 通道数配置: [32, 64, 96, 128]
        self.channels = [base_channels * (2 ** i) if i < 3 else base_channels * 4
                         for i in range(n_blocks)]
        if n_blocks != 4:
            self.channels = [base_channels * (2 ** min(i, 2)) if i < 3
                             else base_channels * 4
                             for i in range(n_blocks)]

        # ★★★ 新增：通道过渡层 (1x1 Conv 升维) ★★★
        self.channel_transitions = nn.ModuleList()
        for i in range(1, n_blocks):
            self.channel_transitions.append(
                nn.Sequential(
                    nn.Conv2d(self.channels[i - 1], self.channels[i], 1, bias=False),
                    nn.ReLU(inplace=True)
                )
            )

        # 级联红外特征精炼块
        self.refinement_blocks = nn.ModuleList([
            CascadeIRFeatureRefinementBlock(self.channels[i])
            for i in range(n_blocks)
        ])

        # HCT 模块
        self.hct_modules = nn.ModuleList([
            HCTModule(self.channels[i], num_heads=num_heads)
            for i in range(n_blocks)
        ])

        # 下连接
        self.down_connections = nn.ModuleList()
        for i in range(n_blocks - 1):
            self.down_connections.append(
                nn.Sequential(
                    nn.Conv2d(self.channels[i], self.channels[i + 1],
                              3, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                )
            )

        # ============================================================
        # 3. 特征融合策略
        # ============================================================
        # 高效特征融合块 (自底向上)
        # ★★★ 修复：efb_down 的输入通道数要按实际拼接情况配置
        # 下连接后 down_feat 已经是 channels[i] (因为下连接里已做 Conv 到 channels[i])
        # 所以拼接后是 channels[i] + channels[i] = 2 * channels[i]
        self.efb_down = nn.ModuleList()
        for i in range(1, n_blocks):
            in_ch = self.channels[i] + self.channels[i]  # 下连接 + 当前层
            self.efb_down.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, self.channels[i], 3, padding=1),
                    nn.ReLU(inplace=True)
                )
            )

        # 反向融合 (自顶向下)
        # 修正：输入通道数应基于实际拼接的通道数
        # 但 reverse_fusion 里 higher_feats 已经是 channels[j]，
        # 而 fused_down[i] 是 channels[i]，所以拼接后 = channels[i] + sum(channels[i+1:])
        self.reverse_fusion = nn.ModuleList()
        for i in range(n_blocks):
            in_ch = sum(self.channels[i:])
            self.reverse_fusion.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, self.channels[i], 3, padding=1),
                    nn.ReLU(inplace=True)
                )
            )

        # 最终融合卷积
        total_ch = sum(self.channels)
        self.final_fusion = nn.Sequential(
            nn.Conv2d(total_ch + self.shallow_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 残差连接
        self.residual_conv = nn.Conv2d(
            self.shallow_channels, base_channels, 3, padding=1
        )

        # ============================================================
        # 4. 全局注意力
        # ============================================================
        if ga_mode is not None:
            if 'CA' in ga_mode:
                self.ca = ChannelAttention(base_channels)
            if addconv:
                self.ga_conv = nn.Conv2d(base_channels, base_channels, 3, padding=1)

        # ============================================================
        # 5. 上采样重建
        # ============================================================
        self.upsample = self._build_upsample(base_channels, out_channels, upscale_factor)

    def _build_upsample(self, in_channels, out_channels, upscale_factor):
        upsampling = []
        if (upscale_factor & (upscale_factor - 1)) == 0:
            num_up = int(math.log2(upscale_factor))
            for _ in range(num_up):
                upsampling.append(nn.Conv2d(in_channels, in_channels * 4, 3, padding=1))
                upsampling.append(nn.PixelShuffle(2))
                upsampling.append(nn.LeakyReLU(0.2, inplace=True))
        elif upscale_factor % 3 == 0:
            num_up = int(math.log(upscale_factor, 3))
            for _ in range(num_up):
                upsampling.append(nn.Conv2d(in_channels, in_channels * 9, 3, padding=1))
                upsampling.append(nn.PixelShuffle(3))
                upsampling.append(nn.LeakyReLU(0.2, inplace=True))
        else:
            raise ValueError(f"不支持的放大倍数: {upscale_factor}")
        upsampling.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        return nn.Sequential(*upsampling)

    def forward(self, x):
        # 1. 浅层特征
        shallow = self.shallow_relu(self.shallow_conv(x))
        deep = self.shallow_to_deep(shallow)

        # 2. 深层特征提取
        hct_outputs = []
        for i in range(self.n_blocks):
            # ★★★ 通道过渡 (除了第 0 块) ★★★
            if i > 0:
                deep = self.channel_transitions[i - 1](deep)
            # 级联红外特征精炼
            deep = self.refinement_blocks[i](deep)
            # HCT
            deep = self.hct_modules[i](deep)
            hct_outputs.append(deep)

        # 3. 特征融合
        # 3.1 下连接融合 (自底向上)
        fused_down = [hct_outputs[0]]
        for i in range(1, self.n_blocks):
            # 下采样 + 通道映射到 channels[i]
            down_feat = self.down_connections[i - 1](hct_outputs[i - 1])
            # 空间尺寸对齐
            if down_feat.shape[-2:] != hct_outputs[i].shape[-2:]:
                down_feat = F.interpolate(
                    down_feat, size=hct_outputs[i].shape[-2:],
                    mode='bilinear', align_corners=False
                )
            # 拼接: (channels[i]) + (channels[i])
            concat = torch.cat([down_feat, hct_outputs[i]], dim=1)
            fused_i = self.efb_down[i - 1](concat)
            fused_down.append(fused_i)

        # 3.2 反向融合 (自顶向下)
        reverse_fused = [None] * self.n_blocks
        reverse_fused[-1] = fused_down[-1]
        for i in range(self.n_blocks - 2, -1, -1):
            higher_feats = []
            for j in range(i + 1, self.n_blocks):
                feat = reverse_fused[j]
                if feat.shape[-2:] != fused_down[i].shape[-2:]:
                    feat = F.interpolate(
                        feat, size=fused_down[i].shape[-2:],
                        mode='bilinear', align_corners=False
                    )
                higher_feats.append(feat)
            concat = torch.cat([fused_down[i]] + higher_feats, dim=1)
            reverse_fused[i] = self.reverse_fusion[i](concat)

        # 3.3 最终融合
        final = torch.cat(reverse_fused, dim=1)
        final = torch.cat([final, shallow], dim=1)
        final = self.final_fusion(final)

        # 3.4 残差连接
        residual = self.residual_conv(shallow)
        out = final + residual

        # 4. 全局注意力
        if self.ga_mode is not None:
            if self.ga_mode == 'CA':
                out = self.ca(out)
            elif self.ga_mode == 'SA':
                out = self.sa(out)
            elif self.ga_mode == 'CA-SA':
                out = self.ca(out)
                out = self.sa(out)
                if self.addconv:
                    out = self.ga_conv(out)
            elif self.ga_mode == 'SA-CA':
                out = self.sa(out)
                out = self.ca(out)
                if self.addconv:
                    out = self.ga_conv(out)
            elif self.ga_mode == 'CA|SA':
                ca_out = self.ca(out)
                sa_out = self.sa(out)
                out = self.ga_conv(torch.cat([ca_out, sa_out], dim=1))

        # 5. 上采样
        out = self.upsample(out)
        return out


class PIFRNet(object):
    def __init__(self, args):
        # parameters
        self.model_name = args.model_name
        self.train_dataset = args.train_dataset
        self.test_dataset = args.test_dataset
        self.crop_size = args.crop_size
        self.test_crop_size = args.test_crop_size
        self.hr_height = args.hr_height
        self.hr_width = args.hr_width
        self.num_threads = args.num_threads
        self.num_channels = args.num_channels
        self.scale_factor = args.scale_factor
        self.epoch = args.epoch
        self.num_epochs = args.num_epochs
        self.save_epochs = args.save_epochs
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.lr = args.lr
        self.b1 = args.b1
        self.b2 = args.b2
        self.data_dir = args.data_dir
        self.root_dir = args.root_dir
        self.save_dir = args.save_dir
        self.gpu_mode = args.gpu_mode
        self.n_cpu = args.n_cpu
        self.sample_interval = args.sample_interval

        self.clip_value = args.clip_value
        self.lambda_gp = args.lambda_gp
        self.gp = args.gp
        self.penalty_type = args.penalty_type
        self.grad_penalty_Lp_norm = args.grad_penalty_Lp_norm
        self.relative = args.relativeGan
        self.loss_Lp_norm = args.loss_Lp_norm
        self.weight_gan = args.weight_gan
        self.weight_content = args.weight_content
        self.max_train_samples = args.max_train_samples

        self.cuda = True if torch.cuda.is_available() else False
        # import PerceptualSimilarity
        from utils import PerceptualSimilarity
        self.PerceptualModel = PerceptualSimilarity.PerceptualLoss(model='net-lin', net='alex', use_gpu=self.cuda)

        self.log_dict = OrderedDict()

        # set the logger
        log_dir = os.path.join(self.root_dir, self.save_dir, 'logs')
        log_freq = 200
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        self.logger = Logger(self.model_name, log_dir, log_freq)

    def load_dataset(self, dataset='train', max_samples=20000):
        if self.num_channels == 1:
            is_gray = True
        else:
            is_gray = False

        if dataset == 'train':
            print('Loading train dct_datasets...')
            train_set = get_IR_trainDataset(self.data_dir, self.train_dataset, self.crop_size, self.scale_factor, is_gray=is_gray)
            return DataLoader(dataset=train_set, num_workers=self.num_threads, batch_size=self.batch_size, shuffle=True, drop_last=True)
        elif dataset == 'test':
            print('Loading test dct_datasets...')
            test_set = get_IR_testDataset(self.data_dir, self.test_dataset, self.crop_size, self.scale_factor, is_gray=is_gray)
            return DataLoader(dataset=test_set, num_workers=self.num_threads, batch_size=self.test_batch_size, shuffle=False, drop_last=True)

    def train(self):
        device_ids = [0]
        cuda = True if torch.cuda.is_available() else False
        self.device = torch.device('cuda' if cuda else 'cpu')

        # Initialize generator and discriminator

        self.generator = GeneratorResNet(upscale_factor=self.scale_factor)
        self.feature_extractor = FeatureExtractor()
        # self.feature_extractor = VGGFeatureExtractor(feature_layer=34, use_bn=False, use_input_norm=True, device=self.device)

        # srutils.print_network_to_file(self.generator, save_dir=self.save_dir, tag='Generator', input_size=(1,3,int(256/self.scale_factor),int(256/self.scale_factor)))
        # srutils.print_network_to_file(self.generator, save_dir=self.save_dir, tag='Generator')
        # srutils.print_network_to_file(self.feature_extractor, save_dir=self.save_dir, tag='FeatureExtractor')


        # Losses
        # self.criterion_GAN = torch.nn.MSELoss().to(self.device)
        if self.loss_Lp_norm=="L1":
            # self.criterion_content = torch.nn.L1Loss().to(self.device)
            self.criterion_content = torch.nn.SmoothL1Loss().to(self.device)  # smoothL1
            self.criterion_edge = EdgeLoss().to(self.device)
        else:
            # self.criterion_content = torch.nn.MSELoss().to(self.device)
            self.criterion_content = torch.nn.SmoothL1Loss().to(self.device)  # smoothL1
            self.criterion_edge = EdgeLoss().to(self.device)

        if len(device_ids) > 1:
            self.generator = nn.DataParallel(self.generator).to(self.device)
            self.feature_extractor = nn.DataParallel(self.feature_extractor).to(self.device)

        if cuda:
            self.generator = self.generator.cuda(device_ids[0])
            self.feature_extractor = self.feature_extractor.cuda(device_ids[0])
            # self.criterion_GAN = self.criterion_GAN.cuda(device_ids[0])
            self.criterion_content = self.criterion_content.cuda(device_ids[0])
            self.criterion_edge = self.criterion_edge.cuda(device_ids[0])

        model_dir = os.path.join(self.save_dir, 'model')
        if self.epoch != 0:
            # Load pretrained models
            G_model_path = model_dir+'/generator_param_epoch_%d.pkl' % self.epoch
            self.load_epoch_network(load_path=G_model_path, network=self.generator, strict=True)
        else:
            # Initialize weights
            self.generator.apply(srutils.weights_init_normal)

        # Optimizers
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
        input_lr = Tensor(self.batch_size, self.num_channels, self.crop_size // self.scale_factor, self.crop_size // self.scale_factor)
        input_hr = Tensor(self.batch_size, self.num_channels, self.crop_size, self.crop_size)
        input_bc = Tensor(self.batch_size, self.num_channels, self.crop_size, self.crop_size)
        #load train dataset
        dataloader = self.load_dataset(dataset='train', max_samples=self.max_train_samples)

        sys.stdout = PrintLogger(self.save_dir)
        # ----------
        #  Training
        # ----------
        print('Training is started.')
        avg_loss_G = []
        avg_loss_D = []
        # avg_mse = []
        avg_psnr = []
        avg_ssim = []
        avg_ergas = []
        avg_lpips = []
        step = 0
        start_time = time.time()

        tensor_to_img = ToPILImage()
        global val_psnr
        global val_ssim
        global val_ergas
        global val_lpips
        global val_psnr_max
        global val_ssim_max
        global val_ergas_max
        global val_lpips_max
        global val_loss_no_improve_count
        global val_loss_noimprove_max_count
        global val_loss_best_step
        val_psnr = 0
        val_ssim = 0
        val_ergas = 0
        val_lpips = 0
        val_psnr_max = 0
        val_ssim_max = 0
        val_ergas_max = 10000
        val_lpips_max = 10000
        val_loss_no_improve_count = 0
        val_loss_noimprove_max_count = 5
        val_loss_best_step = 0
        global epoch
        epoch = self.epoch
        while epoch < self.num_epochs and self.lr >= 0.00001:
        # for epoch in range(self.epoch, self.num_epochs):
            # learning rate is decayed by a factor of 10 every 20 epochs
            # if (epoch + 1) % 10 == 0:
            #     for param_group in optimizer_G.param_groups:
            #         param_group["lr"] /= 2.0
            #     print("Learning rate decay: lr={}".format(optimizer_G.param_groups[0]["lr"]))

            epoch_loss_G = 0
            for i, (input, target, bc_input, imgpath) in enumerate(dataloader):

                # Configure model input
                imgs_lr = Variable(input_lr.copy_(input))
                imgs_hr = Variable(input_hr.copy_(target))
                imgs_bc = Variable(input_bc.copy_(bc_input))

                # ------------------
                #  Train Generators
                # ------------------

                optimizer_G.zero_grad()

                # Generate a high resolution image from low resolution input
                gen_hr = self.generator(imgs_lr)
                # print(gen_hr.shape, imgs_hr.shape)   # 256, 216
                # pixel loss
                pixel_loss_G = self.criterion_content(gen_hr, imgs_hr)
                # Content loss
                gen_features = self.feature_extractor(gen_hr)
                real_features = Variable(self.feature_extractor(imgs_hr).data, requires_grad=False)
                loss_content = self.criterion_content(gen_features, real_features)


                # edge loss
                edge_loss_G = self.criterion_edge(gen_hr, imgs_hr)

                # Total loss
                loss_G = pixel_loss_G + 0.2 * edge_loss_G

                loss_G.backward()
                optimizer_G.step()
                # optimizer_G.module.step()

                print("[Epoch %d/%d] [Batch %d/%d] [G loss: %f]" %
                      (epoch, self.num_epochs, i, len(dataloader), loss_G.item()))

                epoch_loss_G += loss_G.data.item()
                # tensorboard logging
                self.logger.scalar_summary('loss_G', loss_G.item(), step + 1)

                step += 1

                batches_done = epoch * len(dataloader) + i
                if batches_done % self.sample_interval == 0:
                    # Save image sample
                    # bc_imgs = srutils.img_interp(imgs_lr.cpu(), 4)
                    bc_img = tensor_to_img(imgs_bc[0].cpu())
                    bc_img = np.array(bc_img)

                    hr_img = tensor_to_img(imgs_hr[0].cpu())
                    hr_img = np.array(hr_img)

                    gen_img = tensor_to_img(gen_hr[0].cpu())
                    gen_img = np.array(gen_img)

                    # mse = compare_mse(gen_img, hr_img)
                    recon_psnr = compare_psnr(gen_img, hr_img)
                    recon_ssim = compare_ssim(gen_img, hr_img, multichannel=True)   # skimage version 0.15.0
                    # recon_ssim = compare_ssim(gen_img, hr_img, channel_axis=-1)     # skimage version >2.0
                    recon_ergas = srutils.compare_ergas2(hr_img, gen_img, scale=self.scale_factor)
                    recon_lpips = self.PerceptualModel.forward(gen_hr[0], imgs_hr[0], normalize=True).detach().item()

                    # mse1 = compare_mse(bc_img, hr_img)
                    bc_psnr = compare_psnr(bc_img, hr_img)
                    bc_ssim = compare_ssim(bc_img, hr_img, multichannel=True)  # skimage version 0.15.0
                    # bc_ssim = compare_ssim(bc_img, hr_img, channel_axis=-1)      # skimage version >2.0
                    bc_ergas = srutils.compare_ergas2(hr_img, bc_img, scale=self.scale_factor)
                    bc_lpips = self.PerceptualModel.forward(imgs_bc[0], imgs_hr[0], normalize=True).detach().item()

                    result_imgs = [imgs_hr[0].cpu(), imgs_lr[0].cpu(), imgs_bc[0].cpu(), gen_hr[0].cpu()]
                    psnrs = [None, None, bc_psnr, recon_psnr]
                    ssims = [None, None, bc_ssim, recon_ssim]
                    ergas = [None, None, bc_ergas, recon_ergas]
                    lpips = [None, None, bc_lpips, recon_lpips]
                    indicators = {"PSNR":psnrs, "SSIM":ssims, "ERGAS":ergas, "LPIPS":lpips}
                    srutils.plot_result_by_name(result_imgs, indicators, batches_done, imgpath[0],
                                                     save_dir=self.save_dir, is_training=True, show=False)
                    # srutils.plot_test_result_by_name(result_imgs, psnrs, ssims, batches_done, imgpath[0], save_dir=self.save_dir, is_training=True, show=False)
                    # save_image(torch.cat((gen_hr.data, imgs_hr.data), -2),
                    #          'images/%d.png' % batches_done, normalize=True)
                    # log
                    time_elapsed = time.time() - start_time
                    print_rlt = OrderedDict()
                    print_rlt['model'] = self.model_name
                    print_rlt['epoch'] = epoch
                    print_rlt['iters'] = step
                    # print_rlt['G_lr'] = optimizer_G.module.param_groups[0]['lr']
                    # print_rlt['D_lr'] = optimizer_D.module.param_groups[0]['lr']
                    print_rlt['G_lr'] = optimizer_G.param_groups[0]['lr']
                    print_rlt['time'] = time_elapsed
                    print_rlt['G_loss'] = loss_G.item()
                    print_rlt['bicubic_psnr'] = bc_psnr
                    print_rlt['bicubic_ssim'] = bc_ssim
                    print_rlt['bicubic_ergas'] = bc_ergas
                    print_rlt['bicubic_lpips'] = bc_lpips
                    print_rlt['srwgan_psnr'] = recon_psnr
                    print_rlt['srwgan_ssim'] = recon_ssim
                    print_rlt['srwgan_ergas'] = recon_ergas
                    print_rlt['srwgan_lpips'] = recon_lpips
                    for k, v in self.log_dict.items():
                        print_rlt[k] = v
                    self.logger.print_format_results('train', print_rlt)

                    del bc_img, hr_img, gen_img, recon_psnr, recon_ssim, recon_ergas, recon_lpips, bc_ssim, bc_psnr, bc_ergas, bc_lpips, result_imgs, psnrs, ssims
                del imgs_hr, imgs_bc, imgs_lr, gen_hr, gen_features, real_features
                torch.cuda.empty_cache()

            # avg. loss per epoch
            avg_loss_G.append(epoch_loss_G / len(dataloader))

            val_psnr, val_ssim, val_ergas, val_lpips = self.validate(epoch=epoch, save_img=((epoch + 1) % self.save_epochs == 0))

            avg_psnr.append(val_psnr)
            avg_ssim.append(val_ssim)
            avg_ergas.append(val_ergas)
            avg_lpips.append(val_lpips)

            if val_psnr - val_psnr_max > 0:
                val_psnr_max = val_psnr
                val_loss_no_improve_count = 0
                val_loss_best_step = epoch
            elif val_ssim - val_ssim_max > 0:
                val_ssim_max = val_ssim
                val_loss_no_improve_count = 0
                val_loss_best_step = epoch
            elif val_ergas - val_ergas_max < 0:
                val_ergas_max = val_ergas
                val_loss_no_improve_count = 0
                val_loss_best_step = epoch
            elif val_lpips - val_lpips_max < 0:
                val_lpips_max = val_lpips
                val_loss_no_improve_count = 0
                val_loss_best_step = epoch
            else:
                val_loss_no_improve_count = val_loss_no_improve_count + 1

            self.save_epoch_network(save_dir=model_dir, network=self.generator, network_label='generator',
                                    iter_label=epoch + 1)
            # global epoch
            epoch += 1
            if val_loss_no_improve_count >= val_loss_noimprove_max_count:
                G_model_path = model_dir + '/generator_param_epoch_%d.pkl' % (val_loss_best_step + 1)
                self.load_epoch_network(load_path=G_model_path, network=self.generator)
                # for param_group in optimizer_G.module.param_groups:
                #     param_group["lr"] /= 2.0
                # print("Learning rate decay: lr={}".format(optimizer_G.module.param_groups[0]["lr"]))
                for param_group in optimizer_G.param_groups:
                    param_group["lr"] /= 2.0
                print("optimizer_G_Learning rate decay: lr={}".format(optimizer_G.param_groups[0]["lr"]))
                # global epoch
                self.lr /= 2.0
                epoch = val_loss_best_step + 1
                val_loss_no_improve_count = 0
                for _ in range(0, val_loss_noimprove_max_count):
                    avg_psnr.pop()
                    avg_ssim.pop()
                    avg_ergas.pop()
                    avg_lpips.pop()
            # if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            #     # Save model checkpoints
            #     torch.save(generator.state_dict(), 'saved_models/generator_%d.pth' % epoch)
            # Save trained parameters of model
            # if (epoch + 1) % self.save_epochs == 0:
            #     # self.save_model(epoch + 1)
            #     self.save_epoch_network(save_dir=model_dir, network=self.generator, network_label='generator', iter_label=epoch+1)
            torch.cuda.empty_cache()
        # Plot avg. loss
        srutils.plot_loss([avg_loss_G], self.num_epochs, save_dir=self.save_dir)
        srutils.plot_loss([avg_psnr], self.num_epochs, save_dir=self.save_dir, label='PSNR')
        srutils.plot_loss([avg_ssim], self.num_epochs, save_dir=self.save_dir, label='SSIM')
        srutils.plot_loss([avg_ergas], self.num_epochs, save_dir=self.save_dir, label='ERGAS')
        srutils.plot_loss([avg_lpips], self.num_epochs, save_dir=self.save_dir, label='LPIPS')
        print("Training is finished.")

        # Save final trained parameters of model
        self.save_model(epoch=None)

    def validate(self, epoch=0, save_img=False):
        # networks
        cuda = True if torch.cuda.is_available() else False

        # load model
        if self.gpu_mode:
            self.generator.cuda()

        # load dataset
        test_data_loader = self.load_dataset(dataset='test')

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
        input_lr = Tensor(self.test_batch_size, self.num_channels, self.hr_height // self.scale_factor, self.hr_width // self.scale_factor)
        input_hr = Tensor(self.test_batch_size, self.num_channels, self.hr_height, self.hr_width)
        input_bc = Tensor(self.test_batch_size, self.num_channels, self.hr_height, self.hr_width)
        tensor_to_img = ToPILImage()
        # Test
        print('Test is started.')
        img_num = 0
        self.generator.eval()
        start_time = time.time()
        avg_bicubic_mse = 0.0
        avg_bicubic_psnr = 0.0
        avg_bicubic_ssim = 0.0
        avg_bicubic_ergas = 0.0
        avg_bicubic_lpips = 0.0
        avg_srcnn_mse = 0.0
        avg_srcnn_psnr = 0.0
        avg_srcnn_ssim = 0.0
        avg_srcnn_ergas = 0.0
        avg_srcnn_lpips = 0.0
        for iter, (input, target, bc_input, imgpath) in enumerate(test_data_loader):
            # Configure model input
            imgs_lr = Variable(input_lr.copy_(input))
            imgs_hr = Variable(input_hr.copy_(target))
            imgs_bc = Variable(input_bc.copy_(bc_input))

            # prediction
            recon_imgs = self.generator(imgs_lr)
            for i in range(self.test_batch_size):
                img_num += 1
                recon_img = recon_imgs[i].cpu().data
                gt_img = imgs_hr[i]  # utils.shave(target[i], border_size=8)
                # lr_img = imgs_lr[i]
                bc_img = imgs_bc[i]  # srutils.img_interp(lr_img.cpu(), 4)

                # calculate psnrs
                bc_img = tensor_to_img(bc_img.cpu())
                bc_img = np.array(bc_img)

                gt_img = tensor_to_img(gt_img.cpu())
                gt_img = np.array(gt_img)

                recon_img = tensor_to_img(recon_img.cpu())
                recon_img = np.array(recon_img)
                recon_mse = compare_mse(recon_img, gt_img)
                recon_psnr = compare_psnr(recon_img, gt_img)
                recon_ssim = compare_ssim(recon_img, gt_img, multichannel=True)
                # recon_ssim = compare_ssim(recon_img, gt_img, channel_axis=-1)
                recon_ergas = srutils.compare_ergas2(gt_img, recon_img, scale=self.scale_factor)
                # recon_lpips = srutils.compare_lpips(gt_img, recon_img, use_gpu=cuda)
                recon_lpips = self.PerceptualModel.forward(recon_imgs[i], imgs_hr[i], normalize=True).detach().item()

                bc_mse = compare_mse(bc_img, gt_img)
                bc_psnr = compare_psnr(bc_img, gt_img)
                bc_ssim = compare_ssim(bc_img, gt_img, multichannel=True)
                # bc_ssim = compare_ssim(bc_img, gt_img, channel_axis=-1)
                bc_ergas = srutils.compare_ergas2(gt_img, bc_img, scale=self.scale_factor)
                # bc_lpips = srutils.compare_lpips(gt_img, bc_img, use_gpu=cuda)
                bc_lpips = self.PerceptualModel.forward(imgs_bc[i], imgs_hr[i], normalize=True).detach().item()

                if save_img and iter % 50 == 0:
                    # save result images
                    result_imgs = [imgs_hr[i].cpu(), imgs_lr[i].cpu(), imgs_bc[i].cpu(), recon_imgs[i].cpu()]
                    psnrs = [None, None, bc_psnr, recon_psnr]
                    ssims = [None, None, bc_ssim, recon_ssim]
                    ergas = [None, None, bc_ergas, recon_ergas]
                    lpips = [None, None, bc_lpips, recon_lpips]
                    indicators = {"PSNR": psnrs, "SSIM": ssims, "ERGAS": ergas, "LPIPS": lpips}
                    # srutils.plot_test_result_by_name(result_imgs, psnrs, ssims, epoch, imgpath[i],
                                                     # save_dir=self.save_dir)
                    srutils.plot_result_by_name(result_imgs, indicators, epoch, imgpath[i],
                                                save_dir=self.save_dir)

                avg_bicubic_mse += bc_mse
                avg_bicubic_psnr += bc_psnr
                avg_bicubic_ssim += bc_ssim
                avg_bicubic_ergas += bc_ergas
                avg_bicubic_lpips += bc_lpips
                avg_srcnn_mse += recon_mse
                avg_srcnn_psnr += recon_psnr
                avg_srcnn_ssim += recon_ssim
                avg_srcnn_ergas += recon_ergas
                avg_srcnn_lpips += recon_lpips
                print("Saving %d test result images..." % img_num)

                del bc_img, gt_img, recon_img, recon_psnr, recon_ssim, recon_mse, recon_ergas, recon_lpips, bc_mse, bc_ssim, bc_psnr, bc_ergas, bc_lpips
            del imgs_hr, imgs_bc, imgs_lr, recon_imgs
            torch.cuda.empty_cache()

        avg_bicubic_mse = avg_bicubic_mse / img_num
        avg_bicubic_psnr = avg_bicubic_psnr / img_num
        avg_bicubic_ssim = avg_bicubic_ssim / img_num
        avg_bicubic_ergas = avg_bicubic_ergas / img_num
        avg_bicubic_lpips = avg_bicubic_lpips / img_num
        avg_srcnn_mse = avg_srcnn_mse / img_num
        avg_srcnn_psnr = avg_srcnn_psnr / img_num
        avg_srcnn_ssim = avg_srcnn_ssim / img_num
        avg_srcnn_ergas = avg_srcnn_ergas / img_num
        avg_srcnn_lpips = avg_srcnn_lpips / img_num
        time_elapsed = time.time() - start_time
        # Save to log
        print_rlt = OrderedDict()
        print_rlt['model'] = self.model_name
        print_rlt['epoch'] = epoch
        print_rlt['iters'] = epoch
        print_rlt['time'] = time_elapsed
        print_rlt['bicubic_mse'] = avg_bicubic_mse
        print_rlt['bicubic_psnr'] = avg_bicubic_psnr
        print_rlt['bicubic_ssim'] = avg_bicubic_ssim
        print_rlt['bicubic_ergas'] = avg_bicubic_ergas
        print_rlt['bicubic_lpips'] = avg_bicubic_lpips
        print_rlt['srcnn_mse'] = avg_srcnn_mse
        print_rlt['srcnn_psnr'] = avg_srcnn_psnr
        print_rlt['srcnn_ssim'] = avg_srcnn_ssim
        print_rlt['srcnn_ergas'] = avg_srcnn_ergas
        print_rlt['srcnn_lpips'] = avg_srcnn_lpips
        self.logger.print_format_results('val', print_rlt)
        print('-----------------------------------')
        del test_data_loader
        torch.cuda.empty_cache()
        return avg_srcnn_psnr, avg_srcnn_ssim, avg_srcnn_ergas, avg_srcnn_lpips

    # helper saving function that can be used by subclasses
    def save_epoch_network(self, save_dir, network, network_label, iter_label):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_filename = '{}_param_epoch_{}.pkl'.format(network_label, iter_label)
        save_path = os.path.join(save_dir, save_filename)
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    # helper loading function that can be used by subclasses
    def load_epoch_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel):
            network = network.module
        network.load_state_dict(torch.load(load_path), strict=strict)
        print('Trained model is loaded.')

    def save_model(self, epoch=None):
        model_dir = os.path.join(self.save_dir, 'model')
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if epoch is not None:
            torch.save(self.generator.state_dict(), model_dir + '/generator_param_epoch_%d.pkl' % epoch)
        else:
            torch.save(self.generator.state_dict(), model_dir + '/generator_param.pkl')
        print('Trained model is saved.')

    def load_model(self):
        model_dir = os.path.join(self.save_dir, 'model')
        model_name = model_dir + '/generator_param.pkl'
        if os.path.exists(model_name):
            self.generator.load_state_dict(torch.load(model_name), strict=False)
            print('Trained model is loaded.')
            return True
        else:
            print('No model exists to load.')
            return False

    def load_epoch_model(self, epoch):
        model_dir = os.path.join(self.save_dir, 'model')
        model_name = model_dir + '/generator_param_epoch_%d.pkl' % epoch
        if os.path.exists(model_name):
            self.generator.load_state_dict(torch.load(model_name))
            print('Trained model is loaded.')
            return True
        else:
            print('No model exists to load.')
            return False

    def mfeNew_validate(self, epoch=100, modelpath=None):
        # networks
        cuda = True if torch.cuda.is_available() else False

        # load model
        self.generator = GeneratorResNet(upscale_factor=self.scale_factor)
        if self.gpu_mode:
            self.generator.cuda()

        if modelpath is not None:
            self.generator.load_state_dict(torch.load(modelpath), strict=False)

        # self.generator.eval()

        # load dataset
        test_data_loader = self.load_dataset(dataset='test')

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
        input_lr = Tensor(self.test_batch_size, self.num_channels, self.hr_height // self.scale_factor, self.hr_width // self.scale_factor)
        input_hr = Tensor(self.test_batch_size, self.num_channels, self.hr_height, self.hr_width)
        input_bc = Tensor(self.test_batch_size, self.num_channels, self.hr_height, self.hr_width)
        tensor_to_img = ToPILImage()
        # Test
        print('Test is started.')
        img_num = 0
        self.generator.eval()
        start_time = time.time()
        avg_bicubic_mse = 0.0
        avg_bicubic_psnr = 0.0
        avg_bicubic_ssim = 0.0
        avg_bicubic_ergas = 0.0
        avg_bicubic_lpips = 0.0
        avg_srcnn_mse = 0.0
        avg_srcnn_psnr = 0.0
        avg_srcnn_ssim = 0.0
        avg_srcnn_ergas = 0.0
        avg_srcnn_lpips = 0.0
        for iter, (input, target, bc_input, imgpath) in enumerate(test_data_loader):
            # Configure model input
            imgs_lr = Variable(input_lr.copy_(input))
            imgs_hr = Variable(input_hr.copy_(target))
            imgs_bc = Variable(input_bc.copy_(bc_input))

            # prediction
            recon_imgs = self.generator(imgs_lr)
            for i in range(self.test_batch_size):
                img_num += 1
                recon_img = recon_imgs[i].cpu().data
                gt_img = imgs_hr[i]  # utils.shave(target[i], border_size=8)
                # lr_img = imgs_lr[i]
                bc_img = imgs_bc[i]  # srutils.img_interp(lr_img.cpu(), 4)

                # calculate psnrs
                bc_img = tensor_to_img(bc_img.cpu())
                bc_img = np.array(bc_img)

                gt_img = tensor_to_img(gt_img.cpu())
                gt_img = np.array(gt_img)

                recon_img = tensor_to_img(recon_img.cpu())
                recon_img = np.array(recon_img)
                recon_mse = compare_mse(recon_img, gt_img)
                recon_psnr = compare_psnr(recon_img, gt_img)
                recon_ssim = compare_ssim(recon_img, gt_img, multichannel=True)
                # recon_ssim = compare_ssim(recon_img, gt_img, channel_axis=-1)
                recon_ergas = srutils.compare_ergas2(gt_img, recon_img, scale=self.scale_factor)
                recon_lpips = self.PerceptualModel.forward(recon_imgs[i], imgs_hr[i], normalize=True).detach().item()

                bc_mse = compare_mse(bc_img, gt_img)
                bc_psnr = compare_psnr(bc_img, gt_img)
                bc_ssim = compare_ssim(bc_img, gt_img, multichannel=True)
                # bc_ssim = compare_ssim(bc_img, gt_img, channel_axis=-1)
                bc_ergas = srutils.compare_ergas2(gt_img, bc_img, scale=self.scale_factor)
                bc_lpips = self.PerceptualModel.forward(imgs_bc[i], imgs_hr[i], normalize=True).detach().item()



                result_imgs = [imgs_hr[i].cpu(), imgs_lr[i].cpu(), imgs_bc[i].cpu(), recon_imgs[i].cpu()]
                mses = [None, None, bc_mse, recon_mse]
                psnrs = [None, None, bc_psnr, recon_psnr]
                ssims = [None, None, bc_ssim, recon_ssim]
                ergas = [None, None, bc_ergas, recon_ergas]
                lpips = [None, None, bc_lpips, recon_lpips]
                srutils.mfe_plot_test_result2(result_imgs, mses, psnrs, ssims, ergas, lpips, img_num, save_dir=self.save_dir)

                avg_bicubic_mse += bc_mse
                avg_bicubic_psnr += bc_psnr
                avg_bicubic_ssim += bc_ssim
                avg_bicubic_ergas += bc_ergas
                avg_bicubic_lpips += bc_lpips
                avg_srcnn_mse += recon_mse
                avg_srcnn_psnr += recon_psnr
                avg_srcnn_ssim += recon_ssim
                avg_srcnn_ergas += recon_ergas
                avg_srcnn_lpips += recon_lpips
                print("Saving the %d test dataset images" % img_num)

                del bc_img, gt_img, recon_img, recon_psnr, recon_ssim, recon_mse, recon_ergas, recon_lpips, bc_mse, bc_ssim, bc_psnr, bc_ergas, bc_lpips
            del imgs_hr, imgs_bc, imgs_lr, recon_imgs
            torch.cuda.empty_cache()

        avg_bicubic_mse = avg_bicubic_mse / img_num
        avg_bicubic_psnr = avg_bicubic_psnr / img_num
        avg_bicubic_ssim = avg_bicubic_ssim / img_num
        avg_bicubic_ergas = avg_bicubic_ergas / img_num
        avg_bicubic_lpips = avg_bicubic_lpips / img_num
        avg_srcnn_mse = avg_srcnn_mse / img_num
        avg_srcnn_psnr = avg_srcnn_psnr / img_num
        avg_srcnn_ssim = avg_srcnn_ssim / img_num
        avg_srcnn_ergas = avg_srcnn_ergas / img_num
        avg_srcnn_lpips = avg_srcnn_lpips / img_num
        time_elapsed = time.time() - start_time
        # Save to log
        print_rlt = OrderedDict()
        print_rlt['model'] = self.model_name
        print_rlt['epoch'] = epoch
        print_rlt['iters'] = epoch
        print_rlt['time'] = time_elapsed
        print_rlt['bicubic_mse'] = avg_bicubic_mse
        print_rlt['bicubic_psnr'] = avg_bicubic_psnr
        print_rlt['bicubic_ssim'] = avg_bicubic_ssim
        print_rlt['bicubic_ergas'] = avg_bicubic_ergas
        print_rlt['bicubic_lpips'] = avg_bicubic_lpips
        print_rlt['pifrnet_mse'] = avg_srcnn_mse
        print_rlt['pifrnet_psnr'] = avg_srcnn_psnr
        print_rlt['pifrnet_ssim'] = avg_srcnn_ssim
        print_rlt['pifrnet_ergas'] = avg_srcnn_ergas
        print_rlt['pifrnet_lpips'] = avg_srcnn_lpips
        self.logger.print_format_results('val', print_rlt)
        print('-----------------------------------')
        del test_data_loader
        torch.cuda.empty_cache()
        return avg_srcnn_psnr, avg_srcnn_ssim, avg_srcnn_ergas, avg_srcnn_lpips

    def mfeNew_validateByClass(self, epoch, save_img=False, modelpath=None):
        # networks
        cuda = True if torch.cuda.is_available() else False
        # load model
        self.generator = GeneratorResNet(upscale_factor=self.scale_factor)

        if modelpath is not None:
            self.generator.load_state_dict(torch.load(modelpath), strict=False)
        # srutils.print_network_to_file(self.generator, save_dir=self.save_dir, tag='PIFRNet')

        if self.gpu_mode:
            self.generator.cuda()

        self.generator.eval()
        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
        input_lr = Tensor(self.test_batch_size, self.num_channels, self.hr_height // self.scale_factor, self.hr_width // self.scale_factor)
        input_hr = Tensor(self.test_batch_size, self.num_channels, self.hr_height, self.hr_width)
        input_bc = Tensor(self.test_batch_size, self.num_channels, self.hr_height, self.hr_width)
        # Test
        print('Test is started.')
        tensor_to_img = ToPILImage()
        start_time = time.time()
        average_bicubic_psnr = 0.0
        average_bicubic_ssim = 0.0
        average_bicubic_mse = 0.0
        average_bicubic_ergas = 0.0
        average_bicubic_lpips = 0.0
        average_srrgan_mse = 0.0
        average_srrgan_psnr = 0.0
        average_srrgan_ssim = 0.0
        average_srrgan_ergas = 0.0
        average_srrgan_lpips = 0.0
        num = 0
        # create train and val dataloader
        testDataset_dirs = []
        dataset = self.test_dataset[0]
        if dataset == 'Test':
            file_path = os.path.join(self.data_dir, dataset)
            dirs = sorted(os.listdir(file_path))
            for dir in dirs:
                dir_path = os.path.join(file_path, dir)
                if os.path.isdir(dir_path):
                    testDataset_dirs.append(dir_path)  # 最后一层类别文件夹
        for datasetDir in testDataset_dirs:
            datasetDir_list = []
            datasetDir_list.append(datasetDir)
            test_set = get_IR_testDataset(self.data_dir, datasetDir_list, self.crop_size, self.scale_factor,
                                    is_gray=False)
            test_data_loader = DataLoader(dataset=test_set, num_workers=self.num_threads, batch_size=self.test_batch_size, shuffle=False, drop_last=True)
            print('Number of val images in [{:s}]: {:d}'.format(datasetDir, len(test_data_loader)))
            datasetName = os.path.split(datasetDir)[-1]

            avg_bicubic_psnr = 0.0
            avg_bicubic_ssim = 0.0
            avg_bicubic_mse = 0.0
            avg_bicubic_ergas = 0.0
            avg_bicubic_lpips = 0.0
            avg_srcnn_mse = 0.0
            avg_srcnn_psnr = 0.0
            avg_srcnn_ssim = 0.0
            avg_srcnn_ergas = 0.0
            avg_srcnn_lpips = 0.0
            img_num = 0
            for i, (input, target, bc_input, imgpath) in enumerate(test_data_loader):
                # Configure model input
                imgs_lr = Variable(input_lr.copy_(input))
                imgs_hr = Variable(input_hr.copy_(target))
                imgs_bc = Variable(input_bc.copy_(bc_input))

                # prediction
                recon_imgs = self.generator(imgs_lr)
                for i in range(self.test_batch_size):
                    img_num += 1
                    num += 1
                    recon_img = recon_imgs[i].cpu().data
                    gt_img = imgs_hr[i]  # utils.shave(target[i], border_size=8)
                    lr_img = imgs_lr[i]
                    bc_img = imgs_bc[i]  # srutils.img_interp(lr_img.cpu(), 4)

                    # calculate psnrs
                    bc_img = tensor_to_img(bc_img.cpu())
                    bc_img = np.array(bc_img)

                    gt_img = tensor_to_img(gt_img.cpu())
                    gt_img = np.array(gt_img)

                    recon_img = tensor_to_img(recon_img.cpu())
                    recon_img = np.array(recon_img)
                    recon_mse = compare_mse(recon_img, gt_img)
                    recon_psnr = compare_psnr(recon_img, gt_img)
                    recon_ssim = compare_ssim(recon_img, gt_img, multichannel=True)
                    # recon_ssim = compare_ssim(recon_img, gt_img, channel_axis=-1)
                    recon_ergas = srutils.compare_ergas2(gt_img, recon_img, scale=self.scale_factor)
                    # recon_lpips = srutils.compare_lpips(gt_img, recon_img, use_gpu=cuda)
                    recon_lpips = self.PerceptualModel.forward(recon_imgs[i], imgs_hr[i], normalize=True).detach().item()

                    bc_mse = compare_mse(bc_img, gt_img)
                    bc_psnr = compare_psnr(bc_img, gt_img)
                    bc_ssim = compare_ssim(bc_img, gt_img, multichannel=True)
                    # bc_ssim = compare_ssim(bc_img, gt_img, channel_axis=-1)
                    bc_ergas = srutils.compare_ergas2(gt_img, bc_img, scale=self.scale_factor)
                    # bc_lpips = srutils.compare_lpips(gt_img, bc_img, use_gpu=cuda)
                    bc_lpips = self.PerceptualModel.forward(imgs_bc[i], imgs_hr[i], normalize=True).detach().item()

                    if save_img:
                        # save result images
                        result_imgs = [imgs_hr[i].cpu(), imgs_lr[i].cpu(), imgs_bc[i].cpu(), recon_imgs[i].cpu()]
                        mses = [None, None, bc_mse, recon_mse]
                        psnrs = [None, None, bc_psnr, recon_psnr]
                        ssims = [None, None, bc_ssim, recon_ssim]
                        ergas = [None, None, bc_ergas, recon_ergas]
                        lpips = [None, None, bc_lpips, recon_lpips]
                        indicators = {"PSNR": psnrs, "SSIM": ssims, "ERGAS": ergas, "LPIPS": lpips}
                        srutils.plot_result_by_name(result_imgs, indicators, epoch, imgpath[i], is_training=False,
                                                    is_validation=True, classname=datasetName, save_dir=self.save_dir)
                        img_name = os.path.splitext(os.path.basename(imgpath[i]))[0]
                        img_dir = os.path.join(self.save_dir, 'validate', datasetName, '{:s}_x{:d}_{:d}.png'.format(img_name, self.scale_factor, epoch))
                        srutils.save_img1(recon_imgs[i].cpu(), os.path.join(self.save_dir, 'validate'), img_dir, cuda=cuda)

                    avg_bicubic_mse += bc_mse
                    avg_bicubic_psnr += bc_psnr
                    avg_bicubic_ssim += bc_ssim
                    avg_bicubic_ergas += bc_ergas
                    avg_bicubic_lpips += bc_lpips
                    avg_srcnn_mse += recon_mse
                    avg_srcnn_psnr += recon_psnr
                    avg_srcnn_ssim += recon_ssim
                    avg_srcnn_ergas += recon_ergas
                    avg_srcnn_lpips += recon_lpips

                    average_bicubic_mse += bc_mse
                    average_bicubic_psnr += bc_psnr
                    average_bicubic_ssim += bc_ssim
                    average_bicubic_ergas += bc_ergas
                    average_bicubic_lpips += bc_lpips
                    average_srrgan_mse += recon_mse
                    average_srrgan_psnr += recon_psnr
                    average_srrgan_ssim += recon_ssim
                    average_srrgan_ergas += recon_ergas
                    average_srrgan_lpips += recon_lpips
                    # loss_mse += mse
                    print("Saving %d test result images..." % img_num)

                    del bc_img, gt_img, recon_img, recon_mse, recon_psnr, recon_ssim, recon_ergas, recon_lpips, bc_mse, bc_ssim, bc_psnr, bc_ergas, bc_lpips
                del imgs_hr, imgs_bc, imgs_lr, recon_imgs

            avg_bicubic_mse = avg_bicubic_mse / img_num
            avg_bicubic_psnr = avg_bicubic_psnr / img_num
            avg_bicubic_ssim = avg_bicubic_ssim / img_num
            avg_bicubic_ergas = avg_bicubic_ergas / img_num
            avg_bicubic_lpips = avg_bicubic_lpips / img_num
            avg_srcnn_mse = avg_srcnn_mse / img_num
            avg_srcnn_psnr = avg_srcnn_psnr / img_num
            avg_srcnn_ssim = avg_srcnn_ssim / img_num
            avg_srcnn_ergas = avg_srcnn_ergas / img_num
            avg_srcnn_lpips = avg_srcnn_lpips / img_num
            time_elapsed = time.time() - start_time
            # Save to log
            print_rlt = OrderedDict()
            print_rlt['model'] = datasetName
            print_rlt['epoch'] = epoch
            print_rlt['iters'] = epoch
            print_rlt['time'] = time_elapsed
            print_rlt['bicubic_mse'] = avg_bicubic_mse
            print_rlt['bicubic_psnr'] = avg_bicubic_psnr
            print_rlt['bicubic_ssim'] = avg_bicubic_ssim
            print_rlt['bicubic_ergas'] = avg_bicubic_ergas
            print_rlt['bicubic_lpips'] = avg_bicubic_lpips
            print_rlt['pifrnet_mse'] = avg_srcnn_mse
            print_rlt['pifrnet_psnr'] = avg_srcnn_psnr
            print_rlt['pifrnet_ssim'] = avg_srcnn_ssim
            print_rlt['pifrnet_ergas'] = avg_srcnn_ergas
            print_rlt['pifrnet_lpips'] = avg_srcnn_lpips
            self.logger.print_format_results('val', print_rlt)

            del test_data_loader
            del test_set

        average_bicubic_mse = average_bicubic_mse / num
        average_bicubic_psnr = average_bicubic_psnr / num
        average_bicubic_ssim = average_bicubic_ssim / num
        average_bicubic_ergas = average_bicubic_ergas / num
        average_bicubic_lpips = average_bicubic_lpips / num
        average_srrgan_mse = average_srrgan_mse / num
        average_srrgan_psnr = average_srrgan_psnr / num
        average_srrgan_ssim = average_srrgan_ssim / num
        average_srrgan_ergas = average_srrgan_ergas / num
        average_srrgan_lpips = average_srrgan_lpips / num
        time_elapsed = time.time() - start_time
        # Save to log
        print_rlt = OrderedDict()
        print_rlt['model'] = "Total"  # opt['model']
        print_rlt['epoch'] = epoch
        print_rlt['iters'] = epoch
        print_rlt['time'] = time_elapsed
        print_rlt['bicubic_mse'] = average_bicubic_mse
        print_rlt['bicubic_psnr'] = average_bicubic_psnr
        print_rlt['bicubic_ssim'] = average_bicubic_ssim
        print_rlt['bicubic_ergas'] = average_bicubic_ergas
        print_rlt['bicubic_lpips'] = average_bicubic_lpips
        print_rlt['pifrnet_mse'] = average_srrgan_mse
        print_rlt['pifrnet_psnr'] = average_srrgan_psnr
        print_rlt['pifrnet_ssim'] = average_srrgan_ssim
        print_rlt['pifrnet_ergas'] = average_srrgan_ergas
        print_rlt['pifrnet_lpips'] = average_srrgan_lpips
        self.logger.print_format_results('val', print_rlt)
        print('-----------------------------------')
        torch.cuda.empty_cache()