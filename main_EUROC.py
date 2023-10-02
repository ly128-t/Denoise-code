import os
import torch
import src.learning as lr
import src.networks as sn
import src.losses as sl
import src.dataset as ds
import numpy as np

base_dir = os.path.dirname(os.path.realpath(__file__)) #当前Python文件的绝对路径
# data_dir = '/path/to/EUROC/dataset' # 需替换
data_dir = base_dir + '/data/EUROC/dataset'
# test a given network
# address = os.path.join(base_dir, 'results/EUROC/2020_02_18_16_52_55/')
# address = os.path.join(base_dir, 'results/EUROC/2023_05_06_09_47_01/')
# address = os.path.join(base_dir, 'results/EUROC/2023_05_12_10_51_57/')
            #2023_05_12_10_51_57：x[:,[1,2,3]]=2
            # np.savetxt(path, x, header=header, delimiter=" ",
            #         fmt='%1.9f')
# address = os.path.join(base_dir, 'results/EUROC/2023_05_15_17_53_24/')
# or test the last trained network
address = "last"
################################################################################
# Network parameters
################################################################################
net_class = sn.GyroNet # src.networks.py下的GyroNet类
net_params = {
    'in_dim': 6, # 输入维度
    'out_dim': 3, # 输出维度
    'c0': 16, # 第一层通道数
    'dropout': 0.1, # 神经元丢弃率
    # 'ks': [7, 7, 7, 7], # 原始论文中的卷积核参数
    # 'ks': [6, 6, 6, 6],
    'ks': [8, 8, 8, 8],
    # 2023_05_17_09_46_35,核为8，seq如文中设置
    'ds': [4, 4, 4], # 扩张率或者膨胀率，第一层膨胀率1（不扩张），每层乘以四，最后一层不扩张
    'momentum': 0.1, # 动量，历史信息的权重
    'gyro_std': [1*np.pi/180, 2*np.pi/180, 5*np.pi/180], # 陀螺仪的噪声标准差，数据集中给定的
}
################################################################################
# Dataset parameters
################################################################################
dataset_class = ds.EUROCDataset # src.dataset.py下的EUROCDataset类
dataset_params = {
    """
    dataset_params 是一个包含数据集参数的字典。这些参数用于配置数据集的训练、验证和测试序列，并指定数据存储的位置和相关的大小设置。
    以下是 dataset_params 字典中包含的参数及其含义：
    data_dir: 原始数据所在的目录路径。
    predata_dir: 预加载数据存储的目录路径。
    train_seqs: 训练序列的列表，指定用于训练的数据序列。
    val_seqs: 验证序列的列表，指定用于验证的数据序列。
    test_seqs: 测试序列的列表，指定用于测试的数据序列。
    N: 训练期间的轨迹大小，表示在训练期间使用的轨迹数据点数量。这个值应该是一个整数，乘以 'max_train_freq'。
    min_train_freq: 训练期间的最小训练频率，表示相邻两个训练数据点之间的最小时间间隔。
    max_train_freq: 训练期间的最大训练频率，表示相邻两个训练数据点之间的最大时间间隔。
    这些参数用于配置数据集，包括选择使用哪些序列进行训练、验证和测试，以及指定数据存储的位置和训练期间使用的轨迹大小等设置。
    """
    # where are raw data ?
    'data_dir': data_dir,
    # where record preloaded data ?
    'predata_dir': os.path.join(base_dir, 'data/EUROC'),
    # set train, val and test sequence
    'train_seqs': [
        'MH_01_easy',
        'MH_03_medium',
        'MH_05_difficult',
        'V1_02_medium',
        'V2_01_easy',
        'V2_03_difficult',
        ],
    'val_seqs': [
       'MH_01_easy',
        'MH_03_medium',
        'MH_05_difficult',
        'V1_02_medium',
        'V2_01_easy',
        'V2_03_difficult',
        ],
    'test_seqs': [
        'MH_02_easy',
        'MH_04_difficult',
        'V2_02_medium',
        'V1_03_difficult',
        'V1_01_easy',
        ],
    # size of trajectory during training
    'N': 32 * 500, # should be integer * 'max_train_freq'
    'min_train_freq': 16, # 论文中损失函数
    'max_train_freq': 32,
}
################################################################################
# Training parameters
################################################################################
train_params = {
    'optimizer_class': torch.optim.Adam,
    'optimizer': {
        'lr': 0.01, # 初始学习率
        'weight_decay': 1e-1, # 权重衰减（阿尔法）
        'amsgrad': False, # 是否使用amsgrad，优化算法的一种，可以解决Adam算法出现的学习率下降的问题
    },
    'loss_class': sl.GyroLoss,
    'loss': {
        'min_N': int(np.log2(dataset_params['min_train_freq'])), # 损失函数计算批次的次数小值
        'max_N': int(np.log2(dataset_params['max_train_freq'])), # 损失函数计算批次的次数大值
        'w':  1e6,
        'target': 'rotation matrix', # 损失函数的目标对象
        'huber': 0.005, # huber损失函数的阈值
        'dt': 0.005, # IMU采样时间间隔
    },
    'scheduler_class': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, # 
    'scheduler': {
        'T_0': 600, # 初始变化周期
        'T_mult': 2, # 重启周期因子，下一次周期是上一次周期的T_mult倍
        'eta_min': 1e-3, # 最小学习率
    },
    'dataloader': {
        'batch_size': 10, # 每个batch的大小
        'pin_memory': False, # 是否将数据保存在pin memory区，pin memory中的数据转到GPU会快一些
        'num_workers': 0, # 用于数据加载的子进程数
        'shuffle': False, # 是否在每个epoch开始的时候对数据进行重新排序
    },
    # frequency of validation step
    'freq_val': 600, # 每600个epoch进行一次验证
    # total number of epochs
    'n_epochs': 1800, # 总共训练1800个epoch
    # where record results ?
    'res_dir': os.path.join(base_dir, "results/EUROC"),
    # where record Tensorboard log ?
    'tb_dir': os.path.join(base_dir, "results/runs/EUROC"),
}
################################################################################
# Train on training data set
################################################################################
learning_process = lr.GyroLearningBasedProcessing(train_params['res_dir'],
   train_params['tb_dir'], net_class, net_params, None,
   train_params['loss']['dt'])
learning_process.train(dataset_class, dataset_params, train_params)#！
################################################################################
# Test on full data set
################################################################################
learning_process = lr.GyroLearningBasedProcessing(train_params['res_dir'],
    train_params['tb_dir'], net_class, net_params, address=address,
    dt=train_params['loss']['dt'])
learning_process.test(dataset_class, dataset_params, ['test'])
