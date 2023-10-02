# 本文件是专门用于数据处理的文件，主要包括数据集的读取、数据的预处理、数据的加载等功能。

from src.utils import pdump, pload, bmtv, bmtm
from src.lie_algebra import SO3
from termcolor import cprint
from torch.utils.data.dataset import Dataset
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import torch
import sys

class BaseDataset(Dataset):

    def __init__(self, predata_dir, train_seqs, val_seqs, test_seqs, mode, N,
        min_train_freq=128, max_train_freq=512, dt=0.005):
        super().__init__()
        # where record pre loaded data
        self.predata_dir = predata_dir

        # nf.p是经过初始化的数据是训练出来的，这个函数添加nf.p文件的路径。如果路径不存在
        self.path_normalize_factors = os.path.join(predata_dir, 'nf.p')

        self.mode = mode
        # choose between training, validation or test sequences
        train_seqs, self.sequences = self.get_sequences(train_seqs, val_seqs,
            test_seqs) # 这里的train_seqs, val_seqs, test_seqs是在main.py中定义的，train_seqs是一定要有的，而另一个返回的是self.mode，这个选择在leaning.py中的调用中。
        # get and compute value for normalizing inputs
        self.mean_u, self.std_u = self.init_normalize_factors(train_seqs)  # 该函数用于初始化数据归一化因子（Normalization Factors）并返回计算得到的均值（mean_u）和标准差（std_u）。
        self.mode = mode  # train, val or test
        self._train = False
        self._val = False
        # noise density
        self.imu_std = torch.Tensor([8e-5, 1e-3]).float()
        # bias repeatability (without in-run bias stability)
        self.imu_b0 = torch.Tensor([1e-3, 1e-3]).float()
        # IMU sampling time
        self.dt = dt # (s)
        # sequence size during training
        self.N = N # power of 2
        self.min_train_freq = min_train_freq # 训练频率的最小值
        self.max_train_freq = max_train_freq # 训练频率的最大值
        # torch.distributions.uniform.Uniform函数，用于创建一个均匀分布（Uniform Distribution）的随机变量。
        # 具体而言，torch.distributions.uniform.Uniform函数接受两个参数，这里的参数是-torch.ones(1)和torch.ones(1)。这两个参数分别表示均匀分布的下界和上界
        self.uni = torch.distributions.uniform.Uniform(-torch.ones(1),
            torch.ones(1))

    def get_sequences(self, train_seqs, val_seqs, test_seqs):
        """Choose sequence list depending on dataset mode
            这里的train_seqs, val_seqs, test_seqs是在main.py中定义的，train_seqs是一定要有的，而另一个返回的是self.mode，这个选择在leaning.py中的调用中。
        """
        sequences_dict = {
            'train': train_seqs,
            'val': val_seqs,
            'test': test_seqs,
        }
        return sequences_dict['train'], sequences_dict[self.mode]

    def __getitem__(self, i):
        mondict = self.load_seq(i)
        N_max = mondict['xs'].shape[0]
        if self._train: # random start
            n0 = torch.randint(0, self.max_train_freq, (1, ))
            nend = n0 + self.N
        elif self._val: # end sequence
            n0 = self.max_train_freq + self.N
            nend = N_max - ((N_max - n0) % self.max_train_freq)
        else:  # full sequence
            n0 = 0
            nend = N_max - (N_max % self.max_train_freq)
        u = mondict['us'][n0: nend]
        x = mondict['xs'][n0: nend]
        return u, x

    def __len__(self):
        return len(self.sequences)

    def add_noise(self, u):
        """
        Add Gaussian noise and bias to input
        将高斯噪声和偏差添加到输入中，输出为加了噪声和偏差的输入。
        """
        noise = torch.randn_like(u)
        noise[:, :, :3] = noise[:, :, :3] * self.imu_std[0]
        noise[:, :, 3:6] = noise[:, :, 3:6] * self.imu_std[1]

        # bias repeatability (without in run bias stability)
        b0 = self.uni.sample(u[:, 0].shape).cuda()
        b0[:, :, :3] = b0[:, :, :3] * self.imu_b0[0]
        b0[:, :, 3:6] =  b0[:, :, 3:6] * self.imu_b0[1]
        u = u + noise + b0.transpose(1, 2)
        return u

    def init_train(self):
        self._train = True
        self._val = False

    def init_val(self):
        self._train = False
        self._val = True

    def length(self):
        return self._length

    def load_seq(self, i):
        return pload(self.predata_dir, self.sequences[i] + '.p')

    def load_gt(self, i):
        return pload(self.predata_dir, self.sequences[i] + '_gt.p')

    def init_normalize_factors(self, train_seqs):
        """
        该函数用于初始化数据归一化因子（Normalization Factors）并返回计算得到的均值（mean_u）和标准差（std_u），并且将这些数据保存到predata_dir下的nf.p文件中。
        函数首先检查是否存在预先计算好的归一化因子文件（nf.p），如果存在，则直接加载并返回其中的均值和标准差。如果文件不存在，则需要计算归一化因子。
        计算归一化因子的过程如下：
        遍历训练序列（train_seqs）中的每个序列，读取对应的数据文件（.p文件）。
        对于每个序列，提取其中的输入数据（us）和状态变量（sms）。
        计算所有训练数据的总和，用于计算均值和标准差。
        首先计算均值（mean_u），将输入数据（us）按列求和，并除以总数据量得到均值。
        其次，计算标准差（std_u），将输入数据（us）减去均值后平方，按列求和，并除以总数据量，再进行开方运算得到标准差。
        将计算得到的均值和标准差保存到归一化因子文件（nf.p）中。
        最后，函数返回计算得到的均值（mean_u）和标准差（std_u）。
        值得注意的是，归一化因子的计算应该基于训练集数据，并且这些因子在训练和测试过程中需要保持一致以确保数据的一致性。
        """
        # 如果存在预先计算好的归一化因子文件（nf.p），则直接加载并返回其中的均值和标准差。
        if os.path.exists(self.path_normalize_factors):
            mondict = pload(self.path_normalize_factors)
            return mondict['mean_u'], mondict['std_u']

        # 如果文件不存在，则需要计算归一化因子。
        path = os.path.join(self.predata_dir, train_seqs[0] + '.p') # 读取对应的数据集合文件（.p文件）
        if not os.path.exists(path):
            print("init_normalize_factors not computed") # 如果文件不存在，直接返回0，0
            return 0, 0
        # 上述两步都是数据集二进制文件存在的情况下，直接读取数据集文件，如果不存在，则需要进行数据集的预处理，这时可以先看看EURCDataset类中的read_data函数中进行数据集的预处理

        print('Start computing normalizing factors ...')
        cprint("Do it only on training sequences, it is vital!", 'yellow')
        # first compute mean
        num_data = 0

        for i, sequence in enumerate(train_seqs):
            pickle_dict = pload(self.predata_dir, sequence + '.p')
            us = pickle_dict['us']
            sms = pickle_dict['xs']
            if i == 0:
                mean_u = us.sum(dim=0)
                num_positive = sms.sum(dim=0)
                num_negative = sms.shape[0] - sms.sum(dim=0)
            else:
                mean_u += us.sum(dim=0)
                num_positive += sms.sum(dim=0)
                num_negative += sms.shape[0] - sms.sum(dim=0)
            num_data += us.shape[0]
        mean_u = mean_u / num_data
        pos_weight = num_negative / num_positive

        # second compute standard deviation
        for i, sequence in enumerate(train_seqs):
            pickle_dict = pload(self.predata_dir, sequence + '.p')
            us = pickle_dict['us']
            if i == 0:
                std_u = ((us - mean_u) ** 2).sum(dim=0)
            else:
                std_u += ((us - mean_u) ** 2).sum(dim=0)
        std_u = (std_u / num_data).sqrt()
        normalize_factors = {
            'mean_u': mean_u,
            'std_u': std_u,
        }
        print('... ended computing normalizing factors')
        print('pos_weight:', pos_weight)
        print('This values most be a training parameters !')
        print('mean_u    :', mean_u)
        print('std_u     :', std_u)
        print('num_data  :', num_data)
        pdump(normalize_factors, self.path_normalize_factors)
        return mean_u, std_u

    def read_data(self, data_dir):
        """
        这个read_data方法是一个占位符方法，它引发了NotImplementedError异常。在代码中，如果调用了这个方法，将会抛出异常并提示开发者需要在子类中实现该方法。
        NotImplementedError异常通常用于指示子类需要重写父类中的某个方法，以便提供特定的实现。在这种情况下，read_data方法需要在子类中根据具体的需求实现数据的读取逻辑。
        你需要查看子类中是否对read_data方法进行了具体的实现，如果没有，则需要在子类中添加相应的逻辑来读取数据。
        """
        raise NotImplementedError

    @staticmethod
    def interpolate(x, t, t_int):
        """
        Interpolate ground truth at the sensor timestamps
        该函数是为了对齐imu0数据和真值数据的数组维度，保证二者之间的大小相同。
        这段代码定义了一个静态方法interpolate，用于对数据进行插值操作。下面是这个方法的功能解释：
        x: 输入的数据矩阵，大小为(N, M)，其中N是时间步数，M是特征维度。
        t: 输入数据对应的时间戳数组，大小为(N,)，与数据矩阵x的行数相同。
        t_int: 需要进行插值的时间戳数组，大小为(K,)，其中K是需要插值的时间步数。
        方法首先创建一个大小为(K, M)的空矩阵x_int，用于存储插值结果。然后对于每个特征维度，使用np.interp函数对数据进行线性插值，将插值结果存储在x_int中。
        注意，对于索引为4, 5, 6, 7的特征维度（即四元数部分），跳过插值操作。
        最后，对四元数部分进行插值。首先对输入的四元数进行规范化，然后使用SO3.qinterp函数对规范化后的四元数进行插值，得到插值后的四元数结果，并将结果存储在x_int的相应位置。
        最终，方法返回插值后的数据矩阵x_int。
        """
        """
        方便理解函数作用，此为调用函数代码：gt = self.interpolate(gt, gt[:, 0]/1e9, ts)；参数中gt是真值时间戳，ts是imu的时间戳
        由于imu数据集和gt数据集的时间戳不一样，gt中的时间戳要少于imu中的时间戳，即imu采样频率高于gt，
        因此这个函数的作用就是按照imu时间戳，利用线性插值的办法，扩张gt矩阵行数；
        对于除了四元数以外的gt数据，进行线性插值扩张；四元数的处理办法在src.SO3类中的qinterp函数中。
        """

        # vector interpolation，线性插值，但是四元数不能简单的线性插值
        x_int = np.zeros((t_int.shape[0], x.shape[1])) # t_int是ts，即imu时间戳数组，x是gt数据矩阵，该行进行要获取的插值数据初始化，初始化为0
        for i in range(x.shape[1]): # 对于gt中的每列数据
            if i in [4, 5, 6, 7]: # 直接跳过四元数四列到下一个循环
                continue
            # t_int是ts，即imu时间戳数组, t是gt时间戳数组，x[:, i]是gt数据矩阵的第i列数据；x_int[:, i]获取gt数据矩阵的第i列数据的插值结果
            x_int[:, i] = np.interp(t_int, t, x[:, i])
        # quaternion interpolation，处理四元数插值，采用球面线性插值
        t_int = torch.Tensor(t_int - t[0])
        t = torch.Tensor(t - t[0])
        qs = SO3.qnorm(torch.Tensor(x[:, 4:8])) # 再次归一化一次四元数，因为原始数据精度问题，可能会导致四元数不是单位四元数
        x_int[:, 4:8] = SO3.qinterp(qs, t, t_int).numpy()
        return x_int


class EUROCDataset(BaseDataset):
    """
        Dataloader for the EUROC Data Set.
    """

    def __init__(self, data_dir, predata_dir, train_seqs, val_seqs,
                test_seqs, mode, N, min_train_freq, max_train_freq, dt=0.005): # dt间隔时间默认IMU是200Hz
        # 这里先执行了父类的初始化函数，
        super().__init__(predata_dir, train_seqs, val_seqs, test_seqs, mode, N, min_train_freq, max_train_freq, dt) 
        # convert raw data to pre loaded data
        self.read_data(data_dir) # 读取数据

    def read_data(self, data_dir):
        r"""
        Read the data from the dataset
        该段代码的作用是如果没有预处理数据，则读取原始数据，并进行预处理，最后将处理后的数据保存为 .p 文件放在predata_dir路径下。
        函数的主要步骤如下：
        根据给定的 data_dir 构建文件路径，其中包括 IMU 数据和 ground truth 数据的路径。
        遍历数据集中的每个序列。
        读取 IMU 数据和 ground truth 数据，并进行时间同步。
        对数据进行子采样，仅保留指定时间范围内的数据。
        进行插值操作，将 ground truth 数据与 IMU 数据的时间对齐。
        提取 ground truth 的位置和姿态信息。
        将数据转换为 PyTorch 的 Tensor 格式。
        计算预积分因子（pre-integration factors）用于训练。通过计算相邻两个时刻之间的位移量，得到位姿变化的微分（dxi_ij）。
        将处理后的数据保存为 .p 文件，包括预积分因子和 ground truth 数据。
        总体而言，这个方法用于读取并预处理数据，以准备用于训练模型。
        """

        f = os.path.join(self.predata_dir, 'MH_01_easy.p')
        if True and os.path.exists(f): # 如果已经存在预处理数据文件，则直接返回，不存在向下执行
            return

        print("Start read_data, be patient please") # 开始读取数据并且处理数据使其二进制化
        def set_path(seq):
            """
            该函数的作用是根据给定的序列名称构建 IMU 数据和 ground truth 数据的路径。
            """
            path_imu = os.path.join(data_dir, seq, "mav0", "imu0", "data.csv") # 将IMU数据路径加载到path_imu中
            path_gt = os.path.join(data_dir, seq, "mav0", "state_groundtruth_estimate0", "data.csv") # 将ground truth数据路径加载到path_gt中
            return path_imu, path_gt

        sequences = os.listdir(data_dir)
        """
        os.listdir(data_dir)是一个用于获取给定目录下所有文件和文件夹名称的函数。在这种情况下，data_dir是一个目录的路径，通过调用os.listdir(data_dir)可以返回该目录下所有文件和文件夹的名称列表。
        例如，如果data_dir是'/path/to/data'，那么os.listdir(data_dir)将返回data_dir目录中所有文件和文件夹的名称列表。
        请注意，返回的列表中包含了目录中的所有项，包括子目录和文件。如果只需要特定类型的文件或者要过滤掉某些文件或目录，可以使用适当的条件语句对列表进行筛选或处理。
        """
        # read each sequence
        for sequence in sequences:
            print("\nSequence name: " + sequence)
            path_imu, path_gt = set_path(sequence)
            imu = np.genfromtxt(path_imu, delimiter=",", skip_header=1) # 具体的从csv文件中读取IMU数据
            """
            这段代码使用 NumPy 的 genfromtxt() 函数从文件中读取数据，并将其存储在名为 imu 和 gt 的 NumPy 数组中。
            genfromtxt() 函数是 NumPy 提供的一个功能强大的函数，用于从文本文件中加载数据并创建数组。以下是对该函数使用的参数进行解释：
            path_imu 和 path_gt：包含要加载的文件路径的字符串。这些文件应该是包含逗号分隔值的文本文件。
            delimiter=","：指定文件中值之间的分隔符，默认为逗号。在这种情况下，值是用逗号分隔的。
            skip_header=1：指定要跳过的文件头的行数。在这种情况下，跳过了第一行，即文件中的标题行。
            通过这些参数，genfromtxt() 函数将逐行读取文件中的数据，并将其解析为 NumPy 数组。最终，imu 和 gt 分别包含了从 path_imu 和 path_gt 文件中读取的数据。
            """
            gt = np.genfromtxt(path_gt, delimiter=",", skip_header=1) # 具体的从csv文件读取ground truth数据

            # time synchronization between IMU and ground truth
            t0 = np.max([gt[0, 0], imu[0, 0]]) # 起始时间之后的最早时间，取IMU和ground truth的最大值，因为最大值对应的时间戳是共有的最早时间戳
            t_end = np.min([gt[-1, 0], imu[-1, 0]]) # 终止时间之前的最晚时间，取IMU和ground truth的最小值，因为最小值对应的时间戳是共有的最晚时间戳

            # start index，searchsorted函数默认left为True，即返回第一个大于等于t0的数的索引
            idx0_imu = np.searchsorted(imu[:, 0], t0) # 二分查找，返回第一个大于等于t0的数的索引
            idx0_gt = np.searchsorted(gt[:, 0], t0) # 二分查找，返回第一个于等于t0的数的索引

            # end index，searchsorted函数right为True时，即返回第一个大于等于t0的数的索引
            idx_end_imu = np.searchsorted(imu[:, 0], t_end, 'right') # 二分查找，返回第一个小于等于t_end的数的索引
            idx_end_gt = np.searchsorted(gt[:, 0], t_end, 'right') # 二分查找，返回第一个小于等于t_end的数的索引

            # subsample，将数据暂存以便插值
            imu = imu[idx0_imu: idx_end_imu]
            gt = gt[idx0_gt: idx_end_gt]
            ts = imu[:, 0]/1e9 # 固定时间戳并且降低时间戳的大小

            # interpolate，对gt进行线性插值，以和IMU数据对齐
            gt = self.interpolate(gt, gt[:, 0]/1e9, ts)

            # take ground truth position，获取gt真值的第二列到第四列，即gt的R坐标系位置
            p_gt = gt[:, 1:4]
            p_gt = p_gt - p_gt[0] # 删除第一行的表头

            # take ground true quaternion pose，旋转四元数的真值
            q_gt = torch.Tensor(gt[:, 4:8]).double()
            q_gt = q_gt / q_gt.norm(dim=1, keepdim=True)
            Rot_gt = SO3.from_quaternion(q_gt.cuda(), ordering='wxyz').cpu() # 从四元数获取旋转矩阵

            # convert from numpy，将numpy矩阵转变为torch的张量
            p_gt = torch.Tensor(p_gt).double()
            v_gt = torch.tensor(gt[:, 8:11]).double()
            imu = torch.Tensor(imu[:, 1:]).double()

            # compute pre-integration factors for all training
            mtf = self.min_train_freq # 最小训练频率，由main.py中的参数指定
            # Rot_gt[:-mtf] 表示 Rot_gt 张量的从第一个元素到倒数第 mtf 个元素的切片，而 Rot_gt[mtf:] 表示 Rot_gt 张量的从第 mtf 个元素到最后一个元素的切片。
            dRot_ij = bmtm(Rot_gt[:-mtf], Rot_gt[mtf:])
            dRot_ij = SO3.dnormalize(dRot_ij.cuda())
            # 下列计算参考论文，是损失函数的属于真值一部分，这个参数已经插值过的是所选区域的旋转矩阵的真值的SO3对数映射，也是两个元素之间的距离
            dxi_ij = SO3.log(dRot_ij).cpu()

            # save for all training，以二进制保存所有的待训练数据
            mondict = {
                'xs': dxi_ij.float(), # 旋转矩阵的真值的SO3对数映射
                'us': imu.float(), # IMU测量值
            }
            pdump(mondict, self.predata_dir, sequence + ".p")
            # save ground truth，以二进制保存所有的真值数据
            mondict = {
                'ts': ts, # 时间戳
                'qs': q_gt.float(), # 四元数真值
                'vs': v_gt.float(), # 速度真值
                'ps': p_gt.float(), # 位置真值
            }
            pdump(mondict, self.predata_dir, sequence + "_gt.p")


class TUMVIDataset(BaseDataset):
    """
        Dataloader for the TUM-VI Data Set.
    """

    def __init__(self, data_dir, predata_dir, train_seqs, val_seqs,
                test_seqs, mode, N, min_train_freq, max_train_freq, dt=0.005):
        super().__init__(predata_dir, train_seqs, val_seqs, test_seqs, mode, N,
            min_train_freq, max_train_freq, dt)
        # convert raw data to pre loaded data
        self.read_data(data_dir)
        # noise density
        self.imu_std = torch.Tensor([8e-5, 1e-3]).float()
        # bias repeatability (without in-run bias stability)
        self.imu_b0 = torch.Tensor([1e-3, 1e-3]).float()

    def read_data(self, data_dir):
        r"""Read the data from the dataset"""

        f = os.path.join(self.predata_dir, 'dataset-room1_512_16_gt.p')
        if True and os.path.exists(f):
            return

        print("Start read_data, be patient please")
        def set_path(seq):
            path_imu = os.path.join(data_dir, seq, "mav0", "imu0", "data.csv")
            path_gt = os.path.join(data_dir, seq, "mav0", "mocap0", "data.csv")
            return path_imu, path_gt

        sequences = os.listdir(data_dir)

        # read each sequence
        for sequence in sequences:
            print("\nSequence name: " + sequence)
            if 'room' not in sequence:
                continue

            path_imu, path_gt = set_path(sequence)
            imu = np.genfromtxt(path_imu, delimiter=",", skip_header=1)
            gt = np.genfromtxt(path_gt, delimiter=",", skip_header=1)

            # time synchronization between IMU and ground truth
            t0 = np.max([gt[0, 0], imu[0, 0]])
            t_end = np.min([gt[-1, 0], imu[-1, 0]])

            # start index
            idx0_imu = np.searchsorted(imu[:, 0], t0)
            idx0_gt = np.searchsorted(gt[:, 0], t0)

            # end index
            idx_end_imu = np.searchsorted(imu[:, 0], t_end, 'right')
            idx_end_gt = np.searchsorted(gt[:, 0], t_end, 'right')

            # subsample
            imu = imu[idx0_imu: idx_end_imu]
            gt = gt[idx0_gt: idx_end_gt]
            ts = imu[:, 0]/1e9

            # interpolate
            t_gt = gt[:, 0]/1e9
            gt = self.interpolate(gt, t_gt, ts)

            # take ground truth position
            p_gt = gt[:, 1:4]
            p_gt = p_gt - p_gt[0]

            # take ground true quaternion pose
            q_gt = SO3.qnorm(torch.Tensor(gt[:, 4:8]).double())
            Rot_gt = SO3.from_quaternion(q_gt.cuda(), ordering='wxyz').cpu()

            # convert from numpy
            p_gt = torch.Tensor(p_gt).double()
            v_gt = torch.zeros_like(p_gt).double()
            v_gt[1:] = (p_gt[1:]-p_gt[:-1])/self.dt
            imu = torch.Tensor(imu[:, 1:]).double()

            # compute pre-integration factors for all training
            mtf = self.min_train_freq
            dRot_ij = bmtm(Rot_gt[:-mtf], Rot_gt[mtf:])
            dRot_ij = SO3.dnormalize(dRot_ij.cuda())
            dxi_ij = SO3.log(dRot_ij).cpu()

            # masks with 1 when ground truth is available, 0 otherwise
            masks = dxi_ij.new_ones(dxi_ij.shape[0])
            tmp = np.searchsorted(t_gt, ts[:-mtf])
            diff_t = ts[:-mtf] - t_gt[tmp]
            masks[np.abs(diff_t) > 0.01] = 0

            # save all the sequence
            mondict = {
                'xs': torch.cat((dxi_ij, masks.unsqueeze(1)), 1).float(), # 
                'us': imu.float(), # 加速度计和陀螺仪的数据
            }
            pdump(mondict, self.predata_dir, sequence + ".p")

            # save ground truth
            mondict = {
                'ts': ts,
                'qs': q_gt.float(), # 四元数
                'vs': v_gt.float(), # 速度
                'ps': p_gt.float(), # 位置
            }
            pdump(mondict, self.predata_dir, sequence + "_gt.p")
