
import torch
import time
import matplotlib.pyplot as plt
plt.rcParams["legend.loc"] = "upper right"
plt.rcParams['axes.titlesize'] = 'x-large'
plt.rcParams['axes.labelsize'] = 'x-large'
plt.rcParams['legend.fontsize'] = 'x-large'
plt.rcParams['xtick.labelsize'] = 'x-large'
plt.rcParams['ytick.labelsize'] = 'x-large'
from termcolor import cprint
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from src.utils import pload, pdump, yload, ydump, mkdir, bmv
from src.utils import bmtm, bmtv, bmmt
from datetime import datetime
from src.lie_algebra import SO3, CPUSO3


class LearningBasedProcessing:
    def __init__(self, res_dir, tb_dir, net_class, net_params, address, dt):
        """
        这段代码定义了一个名为 LearningBasedProcessing 的类，用于实现基于学习的处理。它继承了 Python 内置的 object 类，这是所有类的基类。
        类的构造函数接受以下参数：
        res_dir: 存储神经网络模型参数和训练信息的路径。
        tb_dir: 存储 TensorBoard 日志文件的路径。
        net_class: 神经网络模型的类。
        net_params: 神经网络模型的参数。
        address: 存储神经网络模型参数和训练信息的路径，如果为 None，则创建一个新的路径。
        dt: 采样时间间隔（s）。这个参数用于计算损失函数中的积分项。

        补充：
        TensorBoard 是一个可视化工具，用于可视化神经网络模型的训练过程和结果。它可以可视化训练过程中的损失函数值、学习率、权重、梯度等等。
        在 PyTorch 中，TensorBoard 也可以用于可视化和分析深度学习模型的训练过程和结果。使用 PyTorch 的 TensorBoard 功能，您可以记录和可视化以下内容：
        训练指标：您可以记录和跟踪训练过程中的损失函数值、准确率、学习率等指标，并通过 TensorBoard 的图表和曲线进行可视化，以便更好地理解模型的训练进展和性能。
        模型图：TensorBoard 可以可视化您构建的神经网络模型的计算图，包括输入、层操作、参数等，帮助您理解模型的结构和信息流动。
        参数直方图：您可以将模型中的参数（如权重和偏置）记录为直方图，并在 TensorBoard 中查看它们的分布情况，以便进行权重初始化、正则化等方面的分析和调优。
        梯度直方图：TensorBoard 可以记录和可视化模型训练过程中的梯度值，帮助您分析梯度的变化、梯度爆炸或消失等问题，并进行相应的调整。
        嵌入向量：如果您在模型中使用了嵌入层，TensorBoard 可以将嵌入向量在高维空间中进行降维和可视化，帮助您理解和分析不同样本之间的相似性和差异性。
        通过使用 PyTorch 的 TensorBoard 功能，您可以更加直观地监控和分析模型的训练过程，优化模型的性能，以及与团队成员分享实验结果。
        """
        self.res_dir = res_dir # 存储神经网络模型参数和训练信息的路径。
        self.tb_dir = tb_dir # 存储 TensorBoard 日志文件的路径。
        self.net_class = net_class
        self.net_params = net_params
        self._ready = False
        self.train_params = {}
        self.figsize = (20, 12)
        self.dt = dt # (s)
        self.address, self.tb_address = self.find_address(address)
        if address is None:  # create new address
            pdump(self.net_params, self.address, 'net_params.p') # 将 net_params 序列化并将字节流写入文件 f 中。
            ydump(self.net_params, self.address, 'net_params.yaml') # 将 net_params 转换为 YAML 格式并将其写入文件中。
        else:  # pick the network parameters
            self.net_params = pload(self.address, 'net_params.p')
            self.train_params = pload(self.address, 'train_params.p')
            self._ready = True
        self.path_weights = os.path.join(self.address, 'weights.pt')
        self.net = self.net_class(**self.net_params)
        if self._ready:  # fill network parameters，如果整个网络已经准备好了，就填充网络参数准备运行
            self.load_weights() # 将网络参数加载到神经网络模型中并且准备加载数据和将神经网络模型移动到 CUDA 设备上

    def find_address(self, address):
        """return path where net and training info are saved
        这段代码定义了一个名为 find_address 的方法，它接受一个参数 address。该方法用于查找存储神经网络和训练信息的路径。
        首先，它检查 address 的值。如果 address 等于字符串 'last'，则执行以下步骤：
        通过调用 os.listdir(self.res_dir) 获取 self.res_dir 目录下的所有文件和文件夹的名称，并进行排序。
        通过 addresses[-1] 获取排序后的最后一个元素，即最新的文件或文件夹的名称。
        使用 os.path.join 将 self.res_dir 和最新的名称拼接成完整的路径，并将结果赋值给 address。
        使用 str(len(addresses)) 创建一个新的路径，用于存储 TensorBoard 日志文件，并将结果赋值给 tb_address。
        如果 address 的值为 None，则执行以下步骤：
        使用 datetime.now().strftime("%Y_%m_%d_%H_%M_%S") 获取当前日期和时间的格式化字符串，例如 "2023_05_23_10_30_00"。
        使用 os.path.join 将 self.res_dir 和格式化后的日期时间字符串拼接成完整的路径，并将结果赋值给 address。
        调用 mkdir(address) 创建一个新的目录，用于存储神经网络和训练信息。
        使用 os.path.join 将 self.tb_dir 和格式化后的日期时间字符串拼接成完整的路径，并将结果赋值给 tb_address。
        如果 address 的值不是 'last' 且不是 None，则执行以下步骤：
        将 tb_address 的值设置为 None。
        最后，方法返回 address 和 tb_address，这两个值将用于其他操作或进一步处理。
        """
        if address == 'last':
            addresses = sorted(os.listdir(self.res_dir)) # sorted() 函数对所有可迭代的对象进行从小到大排序操作。
            tb_address = os.path.join(self.tb_dir, str(len(addresses)))
            address = os.path.join(self.res_dir, addresses[-1])
        elif address is None:
            now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            address = os.path.join(self.res_dir, now)
            mkdir(address)
            tb_address = os.path.join(self.tb_dir, now)
        else:
            tb_address = None
        return address, tb_address

    def load_weights(self):
        """
        这段代码定义了一个名为 load_weights 的方法，用于加载神经网络的权重。
        首先，它通过调用 torch.load(self.path_weights) 加载保存在 self.path_weights 路径下的权重文件，并将加载的结果赋值给变量 weights。
        接下来，它调用 self.net.load_state_dict(weights)，将加载的权重应用到神经网络模型 self.net 的状态字典中，实现权重的加载。
        最后，它调用 self.net.cuda() 将神经网络模型移动到 CUDA 设备上，以便在 GPU 上进行计算加速（如果可用的话）。
        通过这些步骤，方法实现了加载预训练的神经网络权重，并将神经网络模型移动到 CUDA 设备上的操作。
        """
        weights = torch.load(self.path_weights) # 加载保存在 self.path_weights 路径下的权重文件
        self.net.load_state_dict(weights) # 将加载的权重应用到神经网络模型 self.net 的状态字典中
        self.net.cuda() # 将神经网络模型移动到 CUDA 设备上

    def train(self, dataset_class, dataset_params, train_params):
        """train the neural network. GPU is assumed
        这段代码是一个训练神经网络的方法。以下是代码的主要步骤：
        将训练参数保存为 pickle 文件和 YAML 文件。
        根据提供的数据集类和数据集参数，初始化训练集和验证集。
        根据训练参数中指定的优化器、学习率调度器和损失函数类，初始化相应的实例。
        定义数据加载器、优化器、学习率调度器和损失函数。
        根据数据集的特性，对神经网络进行初始化。
        开始 TensorBoard 的写入操作，并记录开始时间和最佳损失值。
        定义一些辅助函数，用于记录训练过程的进展和验证损失的变化。
        进行训练循环，每个 epoch 更新一次参数，并记录训练损失、学习率和时间等信息。
        每隔一定的 epoch 进行验证，计算验证集上的损失，并记录训练过程中的时间消耗。
        根据验证损失的变化情况，更新最佳损失值，并保存模型。
        训练结束后，进行最终的测试，并记录最终的验证损失和测试损失。
        关闭 TensorBoard 写入器，并保存最终的损失信息。
        通过调用该方法，可以进行神经网络的训练过程，并使用 TensorBoard 记录和可视化训练过程和结果，包括损失值、学习率、模型结构等信息。        
        """
        self.train_params = train_params
        pdump(self.train_params, self.address, 'train_params.p')
        ydump(self.train_params, self.address, 'train_params.yaml')
        """
        在这段代码中，".p" 和 ".yaml" 是文件扩展名，用于指示保存的文件类型。这里使用了两种不同的文件格式来保存数据。
        ".p" 文件：这是一个扩展名常用于保存使用 Python 中的 pickle 库进行序列化的对象。
        pickle 是 Python 标准库中的模块，用于将 Python 对象转化为字节流，以便可以保存到文件或在网络上传输，并在需要时重新加载为 Python 对象。在这段代码中，使用 ".p" 文件保存了训练参数和最终的损失值等数据。
        ".yaml" 文件：这是一个扩展名常用于保存 YAML（YAML Ain't Markup Language）格式的文件。YAML 是一种用于序列化数据的标记语言，具有人类可读的结构。
        它被广泛用于配置文件、数据传输和存储等场景。在这段代码中，使用 ".yaml" 文件保存了训练参数、超参数和最终的损失值等数据。
        这两种文件格式都可以方便地保存和加载数据，以便在训练过程中进行参数的持久化和结果的记录与分析。
        """

        hparams = self.get_hparams(dataset_class, dataset_params, train_params) # 调用 get_hparams 方法，获取所有的超参数，并将其保存为字典形式。
        ydump(hparams, self.address, 'hparams.yaml') # 将超参数转换为 YAML 格式并将其写入文件中。

        # define datasets，加载数据集并且初始化数据集，下面四行函数全在 src.dataset.py 下
        dataset_train = dataset_class(**dataset_params, mode='train') # 这里的类是 src.dataset.py 下的类
        dataset_train.init_train() # 初始化训练集
        dataset_val = dataset_class(**dataset_params, mode='val') # 验证模式，模式的选择是在 src.dataset.py 下的 BaseDataset 类中，具体调用数据集类型需要在 src.dataset.py 下的 __init__ 方法中进行设置
        dataset_val.init_val() # 初始化验证集

        # get class
        Optimizer = train_params['optimizer_class'] # 从 train_params 中获取训练过程中使用的优化器类（Optimizer）
        Scheduler = train_params['scheduler_class'] # 从 train_params 中获取训练过程中使用的调度器类（Scheduler）
        Loss = train_params['loss_class'] # 从 train_params 中获取训练过程中使用的损失函数类（Loss）

        # get parameters
        dataloader_params = train_params['dataloader'] # 从 train_params 中获取数据加载器参数（dataloader_params）
        optimizer_params = train_params['optimizer'] # 从 train_params 中获取优化器参数（optimizer_params）
        scheduler_params = train_params['scheduler'] # 从 train_params 中获取调度器参数（scheduler_params）
        loss_params = train_params['loss'] # 从 train_params 中获取损失函数参数（loss_params）

        # define optimizer, scheduler and loss
        dataloader = DataLoader(dataset_train, **dataloader_params) # 定义数据加载器，这里的 DataLoader 是 torch.utils.data.DataLoader
        optimizer = Optimizer(self.net.parameters(), **optimizer_params) # 定义优化器，这里的 self.net.parameters() 是神经网络的参数函数，其实际上是torch.nn.Module.parameters()
        scheduler = Scheduler(optimizer, **scheduler_params) # 定义学习率调度器
        criterion = Loss(**loss_params) # 定义损失函数类

        # remaining training parameters
        freq_val = train_params['freq_val'] # 验证频率
        n_epochs = train_params['n_epochs'] # 训练轮数

        # init net w.r.t dataset
        self.net = self.net.cuda()
        mean_u, std_u = dataset_train.mean_u, dataset_train.std_u
        self.net.set_normalized_factors(mean_u, std_u) # 自定义函数，在learning.py中，用于设置网络的归一化因子

        # start tensorboard writer
        writer = SummaryWriter(self.tb_address) # 定义 TensorBoard 写入器，这里的 SummaryWriter 是 torch.utils.tensorboard.SummaryWriter
        start_time = time.time() # 记录开始时间
        best_loss = torch.Tensor([float('Inf')]) # 初始化最佳损失值

        # define some function for seeing evolution of training
        def write(epoch, loss_epoch):
            """
            根据epoch和损失值，将其写入 TensorBoard 中
            """
            writer.add_scalar('loss/train', loss_epoch.item(), epoch)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            print('Train Epoch: {:2d} \tLoss: {:.4f}'.format(
                epoch, loss_epoch.item()))
            scheduler.step(epoch)

        def write_time(epoch, start_time):
            """
            根据epoch和开始时间，计算训练过程中的时间，并将其写入 TensorBoard 中
            """
            delta_t = time.time() - start_time
            print("Amount of time spent for epochs " +
                "{}-{}: {:.1f}s\n".format(epoch - freq_val, epoch, delta_t))
            writer.add_scalar('time_spend', delta_t, epoch)

        def write_val(loss, best_loss):
            """
            根据损失值和最佳损失值，将其写入 TensorBoard 中
            """
            if 0.5*loss <= best_loss:
                msg = 'validation loss decreases! :) '
                msg += '(curr/prev loss {:.4f}/{:.4f})'.format(loss.item(),
                    best_loss.item())
                cprint(msg, 'green')
                best_loss = loss
                self.save_net()
            else:
                msg = 'validation loss increases! :( '
                msg += '(curr/prev loss {:.4f}/{:.4f})'.format(loss.item(),
                    best_loss.item())
                cprint(msg, 'yellow')
            writer.add_scalar('loss/val', loss.item(), epoch)
            return best_loss

        # training loop !，训练开始并且运行训练循环
        for epoch in range(1, n_epochs + 1):
            loss_epoch = self.loop_train(dataloader, optimizer, criterion)
            write(epoch, loss_epoch)
            scheduler.step(epoch)
            if epoch % freq_val == 0:
                loss = self.loop_val(dataset_val, criterion)
                write_time(epoch, start_time)
                best_loss = write_val(loss, best_loss)
                start_time = time.time()
        # training is over !

        # test on new data
        dataset_test = dataset_class(**dataset_params, mode='test')
        self.load_weights()
        test_loss = self.loop_val(dataset_test, criterion)
        dict_loss = {
            'final_loss/val': best_loss.item(),
            'final_loss/test': test_loss.item()
            }
        writer.add_hparams(hparams, dict_loss)
        ydump(dict_loss, self.address, 'final_loss.yaml')
        writer.close()

    def loop_train(self, dataloader, optimizer, criterion):
        """Forward-backward loop over training data
            dataloader：数据加载器
            optimizer：优化器
            criterion：损失函数
        """
        loss_epoch = 0
        optimizer.zero_grad()
        for us, xs in dataloader:
            us = dataloader.dataset.add_noise(us.cuda())
            hat_xs = self.net(us) # 这里的 self.net 是神经网络，对加了噪声的us进行前向传播
            loss = criterion(xs.cuda(), hat_xs)/len(dataloader)
            loss.backward()
            loss_epoch += loss.detach().cpu()
        optimizer.step()
        return loss_epoch

    def loop_val(self, dataset, criterion):
        """Forward loop over validation data
            dataset：验证数据集
            criterion：损失函数
        """
        loss_epoch = 0
        self.net.eval() # 这里的 self.net 是神经网络，eval()将其设置为评估模式
        with torch.no_grad():
            for i in range(len(dataset)):
                us, xs = dataset[i]
                hat_xs = self.net(us.cuda().unsqueeze(0))
                loss = criterion(xs.cuda().unsqueeze(0), hat_xs)/len(dataset)
                loss_epoch += loss.cpu()
        self.net.train()
        return loss_epoch

    def save_net(self):
        """save the weights on the net in CPU
        这段代码定义了一个名为save_net的方法，用于将神经网络的权重保存到文件中。下面是方法的具体步骤：
        self.net.eval().cpu()：将神经网络切换到评估模式，并将网络的计算设备切换到CPU。这是为了确保在保存权重时使用CPU。
        torch.save(self.net.state_dict(), self.path_weights)：使用torch.save()函数将神经网络的状态字典self.net.state_dict()保存到指定的文件路径self.path_weights中。
        状态字典包含了神经网络的所有参数和持久化缓存的状态。
        self.net.train().cuda()：将神经网络切换回训练模式，并将网络的计算设备切换回GPU（如果可用）。这是为了恢复网络的正常训练状态。
        """
        self.net.eval().cpu()
        torch.save(self.net.state_dict(), self.path_weights) # 将当前网络的状态字典保存到硬盘上，是weights.pt的参数
        self.net.train().cuda()

    def get_hparams(self, dataset_class, dataset_params, train_params):
        """
        return all training hyperparameters in a dict
        这段代码定义了一个方法 get_hparams(self, dataset_class, dataset_params, train_params)，用于获取所有的训练超参数（hyperparameters）并以字典形式返回。
        函数的主要步骤如下：
        从 train_params 中获取训练过程中使用的优化器类（Optimizer）、调度器类（Scheduler）和损失函数类（Loss）。
        从 train_params 中获取各个类的参数，包括数据加载器参数（dataloader_params）、优化器参数（optimizer_params）、调度器参数（scheduler_params）和损失函数参数（loss_params）。
        从 train_params 中获取剩余的训练参数，包括验证频率（freq_val）和训练轮数（n_epochs）。
        创建一个字典 dict_class，将优化器类、调度器类和损失函数类的字符串表示添加到字典中。
        将所有参数合并到一个字典中，并返回该字典。
        该方法的目的是将训练过程中使用的所有超参数整合到一个字典中，方便保存和记录训练过程的配置信息。
        """
        Optimizer = train_params['optimizer_class'] # 从 train_params 中获取训练过程中使用的优化器类（Optimizer）
        Scheduler = train_params['scheduler_class'] # 从 train_params 中获取训练过程中使用的调度器类（Scheduler） 
        Loss = train_params['loss_class'] # 从 train_params 中获取训练过程中使用的损失函数类（Loss）

        # get training class parameters
        dataloader_params = train_params['dataloader'] # 从 train_params 中获取数据加载器参数（dataloader_params）
        optimizer_params = train_params['optimizer'] # 从 train_params 中获取优化器参数（optimizer_params） 
        scheduler_params = train_params['scheduler'] # 从 train_params 中获取调度器参数（scheduler_params）
        loss_params = train_params['loss'] # 从 train_params 中获取损失函数参数（loss_params）

        # remaining training parameters
        freq_val = train_params['freq_val'] # 从 train_params 中获取验证频率（freq_val）
        n_epochs = train_params['n_epochs'] # 从 train_params 中获取训练轮数（n_epochs）

        dict_class = {
            'Optimizer': str(Optimizer),
            'Scheduler': str(Scheduler),
            'Loss': str(Loss)
        }

        return {**dict_class, **dataloader_params, **optimizer_params,
                **loss_params, **scheduler_params,
                'n_epochs': n_epochs, 'freq_val': freq_val}

    def test(self, dataset_class, dataset_params, modes):
        """test a network once training is over"""

        # get loss function
        Loss = self.train_params['loss_class']
        loss_params = self.train_params['loss']
        criterion = Loss(**loss_params)

        # test on each type of sequence
        # 这里的modes是一个列表，包含了训练集、验证集和测试集，其定义在dataset.py中，具体的值为['train', 'val', 'test']
        # 列表本质上是各类数据集，这里测试集test是一系列数据集的集合，包含了多个数据集，每个数据集都是一个序列
        for mode in modes:
            dataset = dataset_class(**dataset_params, mode=mode)
            self.loop_test(dataset, criterion) # 调用loop_test方法，对数据集中的具体的数据进行测试
            self.display_test(dataset, mode)

    def loop_test(self, dataset, criterion):
        """Forward loop over test data
            本测试函数主要是针对测试集中的数据集中的具体的数据进行的测试，而不是针对整个测试集进行的测试
        """
        self.net.eval()
        for i in range(len(dataset)):
            seq = dataset.sequences[i]
            us, xs = dataset[i]
            with torch.no_grad():
                hat_xs = self.net(us.cuda().unsqueeze(0))
            loss = criterion(xs.cuda().unsqueeze(0), hat_xs)
            mkdir(self.address, seq)
            mondict = {
                'hat_xs': hat_xs[0].cpu(),
                'loss': loss.cpu().item(),
            }
            pdump(mondict, self.address, seq, 'results.p')

    def display_test(self, dataset, mode):
        raise NotImplementedError


class GyroLearningBasedProcessing(LearningBasedProcessing):
    def __init__(self, res_dir, tb_dir, net_class, net_params, address, dt):
        super().__init__(res_dir, tb_dir, net_class, net_params, address, dt)
        self.roe_dist = [7, 14, 21, 28, 35] # m
        self.freq = 100 # subsampling frequency for RTE computation
        self.roes = { # relative trajectory errors
            'Rots': [],
            'yaws': [],
            }

    def display_test(self, dataset, mode):
        """display results for a test sequence"""
        self.roes = {
            'Rots': [],
            'yaws': [],
        }
        self.to_open_vins(dataset)
        for i, seq in enumerate(dataset.sequences):
            print('\n', 'Results for sequence ' + seq )
            self.seq = seq
            # get ground truth
            self.gt = dataset.load_gt(i)
            Rots = SO3.from_quaternion(self.gt['qs'].cuda())
            self.gt['Rots'] = Rots.cpu()
            self.gt['rpys'] = SO3.to_rpy(Rots).cpu()
            # get data and estimate
            self.net_us = pload(self.address, seq, 'results.p')['hat_xs']
            self.raw_us, _ = dataset[i]
            N = self.net_us.shape[0]
            self.gyro_corrections =  (self.raw_us[:, :3] - self.net_us[:N, :3])
            self.ts = torch.linspace(0, N*self.dt, N)

            self.convert()
            self.plot_gyro()
            self.plot_gyro_correction()
            plt.show()

    def to_open_vins(self, dataset):
        """
        Export results to Open-VINS format. Use them eval toolbox available 
        at https://github.com/rpng/open_vins/
        将训练好的网络的结果转换为Open-VINS的格式，以便于使用Open-VINS的工具箱进行评估
        """

        for i, seq in enumerate(dataset.sequences):
            self.seq = seq
            # get ground truth
            self.gt = dataset.load_gt(i)
            raw_us, _ = dataset[i]
            net_us = pload(self.address, seq, 'results.p')['hat_xs']
            N = net_us.shape[0]
            net_qs, imu_Rots, net_Rots = self.integrate_with_quaternions_superfast(N, raw_us, net_us)
            path = os.path.join(self.address, seq + '.txt')
            header = "timestamp(s) tx ty tz qx qy qz qw"
            x = np.zeros((net_qs.shape[0], 8))
            x[:, 0] = self.gt['ts'][:net_qs.shape[0]]
            x[:, [7, 4, 5, 6]] = net_qs
            # x[:,[1,2,3]]=3
            np.savetxt(path, x[::10], header=header, delimiter=" ",
                    fmt='%1.9f')
#x[::10]
    def convert(self):
        # s -> min
        l = 1/60
        self.ts *= l

        # rad -> deg
        l = 180/np.pi
        self.gyro_corrections *= l
        self.gt['rpys'] *= l

    def integrate_with_quaternions_superfast(self, N, raw_us, net_us):
        imu_qs = SO3.qnorm(SO3.qexp(raw_us[:, :3].cuda().double()*self.dt))
        net_qs = SO3.qnorm(SO3.qexp(net_us[:, :3].cuda().double()*self.dt))
        Rot0 = SO3.qnorm(self.gt['qs'][:2].cuda().double())
        imu_qs[0] = Rot0[0]
        net_qs[0] = Rot0[0]

        N = np.log2(imu_qs.shape[0])
        for i in range(int(N)):
            k = 2**i
            imu_qs[k:] = SO3.qnorm(SO3.qmul(imu_qs[:-k], imu_qs[k:]))
            net_qs[k:] = SO3.qnorm(SO3.qmul(net_qs[:-k], net_qs[k:]))

        if int(N) < N:
            k = 2**int(N)
            k2 = imu_qs[k:].shape[0]
            imu_qs[k:] = SO3.qnorm(SO3.qmul(imu_qs[:k2], imu_qs[k:]))
            net_qs[k:] = SO3.qnorm(SO3.qmul(net_qs[:k2], net_qs[k:]))

        imu_Rots = SO3.from_quaternion(imu_qs).float()
        net_Rots = SO3.from_quaternion(net_qs).float()
        return net_qs.cpu(), imu_Rots, net_Rots

    def plot_gyro(self):
        N = self.raw_us.shape[0]
        raw_us = self.raw_us[:, :3]
        net_us = self.net_us[:, :3]

        net_qs, imu_Rots, net_Rots = self.integrate_with_quaternions_superfast(N,
        raw_us, net_us)
        imu_rpys = 180/np.pi*SO3.to_rpy(imu_Rots).cpu()
        net_rpys = 180/np.pi*SO3.to_rpy(net_Rots).cpu()
        self.plot_orientation(imu_rpys, net_rpys, N)
        self.plot_orientation_error(imu_Rots, net_Rots, N)

    def plot_orientation(self, imu_rpys, net_rpys, N):
        title = "Orientation estimation"
        gt = self.gt['rpys'][:N]
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=self.figsize)
        axs[0].set(ylabel='roll (deg)', title=title)
        axs[1].set(ylabel='pitch (deg)')
        axs[2].set(xlabel='$t$ (min)', ylabel='yaw (deg)')

        for i in range(3):
            axs[i].plot(self.ts, gt[:, i], color='black', label=r'ground truth')
            axs[i].plot(self.ts, imu_rpys[:, i], color='red', label=r'raw IMU')
            axs[i].plot(self.ts, net_rpys[:, i], color='blue', label=r'net IMU')
            axs[i].set_xlim(self.ts[0], self.ts[-1])
        self.savefig(axs, fig, 'orientation')

    def plot_orientation_error(self, imu_Rots, net_Rots, N):
        gt = self.gt['Rots'][:N].cuda()
        raw_err = 180/np.pi*SO3.log(bmtm(imu_Rots, gt)).cpu()
        net_err = 180/np.pi*SO3.log(bmtm(net_Rots, gt)).cpu()
        title = "$SO(3)$ orientation error"
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=self.figsize)
        axs[0].set(ylabel='roll (deg)', title=title)
        axs[1].set(ylabel='pitch (deg)')
        axs[2].set(xlabel='$t$ (min)', ylabel='yaw (deg)')

        for i in range(3):
            axs[i].plot(self.ts, raw_err[:, i], color='red', label=r'raw IMU')
            axs[i].plot(self.ts, net_err[:, i], color='blue', label=r'net IMU')
            axs[i].set_ylim(-10, 10)
            axs[i].set_xlim(self.ts[0], self.ts[-1])
        self.savefig(axs, fig, 'orientation_error')

    def plot_gyro_correction(self):
        title = "Gyro correction" + self.end_title
        ylabel = 'gyro correction (deg/s)'
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set(xlabel='$t$ (min)', ylabel=ylabel, title=title)
        plt.plot(self.ts, self.gyro_corrections, label=r'net IMU')
        ax.set_xlim(self.ts[0], self.ts[-1])
        self.savefig(ax, fig, 'gyro_correction')

    @property
    def end_title(self):
        return " for sequence " + self.seq.replace("_", " ")

    def savefig(self, axs, fig, name):
        if isinstance(axs, np.ndarray):
            for i in range(len(axs)):
                axs[i].grid()
                axs[i].legend()
        else:
            axs.grid()
            axs.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.address, self.seq, name + '.png'))

