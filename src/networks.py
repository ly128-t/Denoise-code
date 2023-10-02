import torch
import matplotlib.pyplot as plt
import numpy as np
from src.utils import bmtm, bmtv, bmmt, bbmv
from src.lie_algebra import SO3


class BaseNet(torch.nn.Module):
    def __init__(self, in_dim, out_dim, c0, dropout, ks, ds, momentum):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # channel dimension，通道维度，论文中c0=16
        c1 = 2*c0
        c2 = 2*c1
        c3 = 2*c2
        # kernel dimension (odd number)，核维度，论文中全是7
        k0 = ks[0]
        k1 = ks[1]
        k2 = ks[2]
        k3 = ks[3]
        # dilation dimension，扩张维度，和论文中不同，这里是扩张维度按照指数排列，这里全是4
        d0 = ds[0]
        d1 = ds[1]
        d2 = ds[2]
        # padding，填充率或者也叫填充个数
        p0 = (k0-1) + d0*(k1-1) + d0*d1*(k2-1) + d0*d1*d2*(k3-1)
        # nets
        self.cnn = torch.nn.Sequential(
            torch.nn.ReplicationPad1d((p0, 0)), # padding at start，边缘复制填充
            torch.nn.Conv1d(in_dim, c0, k0, dilation=1), # 一维卷积，扩张率是1，下面的数据和论文一致
            torch.nn.BatchNorm1d(c0, momentum=momentum), # 一维归一化层，动量momentum是批归一化层（Batch Normalization）中的一个参数，用于控制在计算均值和方差时历史信息的权重。
            # 具体来说，它表示历史均值和历史方差的更新速度。较大的momentum值会使得历史信息的权重更大，从而对当前的均值和方差计算有更大的影响。
            # 一般来说，momentum取较小的值（例如0.1或0.01）能够在一定程度上平衡历史信息和当前样本的影响，从而提高模型的泛化能力和训练稳定性。
            torch.nn.GELU(), # 激活函数
            torch.nn.Dropout(dropout), # 这里用于随机损失神经元，防止过拟合
            torch.nn.Conv1d(c0, c1, k1, dilation=d0),
            torch.nn.BatchNorm1d(c1, momentum=momentum),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(c1, c2, k2, dilation=d0*d1),
            torch.nn.BatchNorm1d(c2, momentum=momentum),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(c2, c3, k3, dilation=d0*d1*d2),
            torch.nn.BatchNorm1d(c3, momentum=momentum),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(c3, out_dim, 1, dilation=1),
            torch.nn.ReplicationPad1d((0, 0)), # no padding at end
        )
        # for normalizing inputs
        self.mean_u = torch.nn.Parameter(torch.zeros(in_dim), # 初始化输入均值
            requires_grad=False)
        self.std_u = torch.nn.Parameter(torch.ones(in_dim), # 初始化输入标准差
            requires_grad=False)

    def forward(self, us):
        u = self.norm(us).transpose(1, 2)
        y = self.cnn(u) # 一维卷积
        return y

    def norm(self, us):
        """
        Normalize inputs.
        因为加入的噪声是高斯白噪声，因此这里对输入进行归一化，使得输入的均值为0，方差为1
        """
        return (us-self.mean_u)/self.std_u

    def set_normalized_factors(self, mean_u, std_u):
        """
        Set factors for normalizing inputs.
        """
        mean_u = torch.Tensor(mean_u).cuda()
        self.mean_u = torch.nn.Parameter(mean_u.cuda(), requires_grad=False)

        self.std_u = torch.nn.Parameter(std_u.cuda(), requires_grad=False)


class GyroNet(BaseNet):
    def __init__(self, in_dim, out_dim, c0, dropout, ks, ds, momentum,
        gyro_std):
        super().__init__(in_dim, out_dim, c0, dropout, ks, ds, momentum)
        gyro_std = torch.Tensor(gyro_std)
        self.gyro_std = torch.nn.Parameter(gyro_std, requires_grad=False)

        gyro_Rot = 0.05*torch.randn(3, 3).cuda()
        self.gyro_Rot = torch.nn.Parameter(gyro_Rot)
        self.Id3 = torch.eye(3).cuda()

    def forward(self, us):
        """
        参数：
            us: IMU测量值
        """
        ys = super().forward(us)
        Rots = (self.Id3 + self.gyro_Rot).expand(us.shape[0], us.shape[1], 3, 3) 
        # 根据代码，Rots表示旋转矩阵，计算方式为将单位矩阵self.Id3与陀螺仪旋转矩阵self.gyro_Rot相加。
        # 使用expand函数将其扩展为与输入数据us相同的形状（us.shape[0]表示批次大小，us.shape[1]表示序列长度）。
        # 最终的Rots是一个形状为(批次大小, 序列长度, 3, 3)的张量，表示每个样本在不同时间步的旋转矩阵。
        # 这段代码的目的是根据陀螺仪旋转矩阵调整输入数据的旋转。
        Rot_us = bbmv(Rots, us[:, :, :3]) # us[:, :, :3]为取前三列，即陀螺仪测量值
        return self.gyro_std*ys.transpose(1, 2) + Rot_us

