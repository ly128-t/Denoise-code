from src.utils import pload, pdump, yload, ydump, mkdir, bmv, bmtm, bmtv, bmmt
import src.dataset as dt
import torch
import numpy as np
import pickle
import os
import csv


# # torch.einsum的使用方法

# # 创建两个张量
# a = torch.tensor([1, 2, 3])
# b = torch.tensor([4, 5, 6])

# # 执行点积操作
# result = torch.einsum('i,i->', a, b)
# print(a, b, result)  # 输出: 32


# # 创建两个批量矩阵
# a = torch.tensor([[1, 2], [3, 4]])
# b = torch.tensor([[5, 6], [7, 8]])

# # 执行批量矩阵乘法操作
# result = torch.einsum('ij,jk->ik', a, b)
# print(a, b, result)  # 输出: tensor([[19, 22], [43, 50]])

# # 创建两个张量
# a = torch.tensor([1, 2, 3])
# b = torch.tensor([4, 5, 6])

# # 执行逐元素相乘操作
# result = torch.einsum('i,i->i', a, b)
# print(a, b, result)  # 输出: tensor([4, 10, 18])

# # 测试 utils.py 中的函数

# # 创建两个批量矩阵
# mat1 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
# mat2 = torch.tensor([[[9, 10, 11], [12, 13, 14]], [[15, 16, 17], [18, 19, 20]]])

# print(mat1.shape, mat2.shape)
# # 执行批量矩阵矩阵转置乘法
# result = bmtm(mat1, mat2)
# print(mat1)
# print(mat2)
# print(result)


# # 读取.p数据并且转化为txt数据
# base_dir = os.path.dirname(os.path.realpath(__file__)) #当前Python文件的绝对路径
# def convert_pkl_to_txt(pkl_file, txt_file):
#     # 读取.p文件
#     with open(pkl_file, 'rb') as f:
#         data = pickle.load(f)

#     # 将数据转换为文本格式
#     data_str = str(data)

#     # 写入.txt文件
#     with open(txt_file, 'w') as f:
#         f.write(data_str)

# # 指定.p文件路径和要保存的.txt文件路径
# pkl_file = os.path.join(base_dir, 'data/EUROC/nf.p')
# txt_file = os.path.join(base_dir, 'data/EUROC/nf.txt')

# # 调用函数进行转换
# convert_pkl_to_txt(pkl_file, txt_file)

# # np.searchsorted函数的验证
# a = np.array([1, 3, 5, 7, 9, 10])

# # 查找单个值的插入位置
# index = np.searchsorted(a, 6)
# print(index)  # 输出: 3

# # 如果元素存在
# index0 = np.searchsorted(a, 5)
# print(index0) # 输出: 2

# # 验证right
# index1 = np.searchsorted(a, 6, 'right')
# print(index1) # 输出: 3

# # 如果元素存在
# index2 = np.searchsorted(a, 9, 'right')
# print(index2) # 输出: 5

# # 查找多个值的插入位置
# values = [2, 4, 6, 8, 10]
# indices = np.searchsorted(a, values)
# print(indices)  # 输出: [1 2 3 4 5]

# imu = 1403636580928555520/1e9
# gt = 1403636580928550000.00/1e9
# print('imu', imu,'gt', gt)


# def rodrigues_rotation(r, theta):
#     # n旋转轴[3x1]
#     # theta为旋转角度
#     # 旋转是过原点的，n是旋转轴
#     r = np.array(r).reshape(3, 1)
#     rx, ry, rz = r[:, 0]
#     M = np.array([
#         [0, -rz, ry],
#         [rz, 0, -rx],
#         [-ry, rx, 0]
#     ])
#     R = np.eye(4)
#     R[:3, :3] = np.cos(theta) * np.eye(3) +        \
#                 (1 - np.cos(theta)) * r @ r.T +    \
#                 np.sin(theta) * M
#     return R

# print(rodrigues_rotation([0, 0, 1], 30 / 180 * np.pi))


# # 读取.p数据并且转化为txt数据检验数据预处理效果
# base_dir = os.path.dirname(os.path.realpath(__file__)) #当前Python文件的绝对路径
# data = dt.pload(base_dir + '/data/EUROC/MH_01_easy.p')
# predata = dt.pload(base_dir + '/data/EUROC/MH_01_easy_gt.p')
# nf = dt.pload(base_dir + '/data/EUROC/nf.p')
# print(nf)

# data1 = data['xs']
# data2 = data['us']
# nf_mean_u = np.array(nf['mean_u'])
# nf_std_u = np.array(nf['std_u'])
# nf_tmp = np.append(nf_mean_u, nf_std_u)

# predata1 = predata['ts']
# predata2 = predata['qs']
# predata3 = predata['vs']
# predata4 = predata['ps']

# a_data1 = np.array(data1)
# a_data2 = np.array(data2)

# a_predata1 = np.array(predata1)
# a_predata2 = np.array(predata2)
# a_predata3 = np.array(predata3)
# a_predata4 = np.array(predata4)

# np.savetxt('test/xs_dxi.csv', a_data1, delimiter=',')
# np.savetxt('test/us_imu.csv', a_data2, delimiter=',')
# np.savetxt('test/ts_time.csv', a_predata1, delimiter=',')
# np.savetxt('test/qs_quaternions_gt.csv', a_predata2, delimiter=',')
# np.savetxt('test/vs_velociaty_gt.csv', a_predata3, delimiter=',')
# np.savetxt('test/ps_poistion_gt.csv', a_predata4, delimiter=',')
# np.savetxt('test/nf_mean_std_u.csv', nf_tmp, delimiter=',')


# # 插值检验
# # 原始数据点
# xp = [1, 2, 3, 20, 25]  # x 坐标
# fp = [10, 20, 30, 40, 50]  # y 坐标

# # 目标点数组
# x = [1.5, 3, 21]

# # 在目标点数组进行线性插值
# y = np.interp(x, xp, fp)

# print("目标点数组的插值结果:", y) # 目标点数组的插值结果: [15. 30. 42.]

# 四元数应用，证明原先的真值数据是单位四元数
# w, x, y, z = 0.534108,-0.153029,-0.827383,-0.082152
# q = w**2 + x**2 + y**2 + z**2
# print(q)


# Dataset类的测试：参考连接https://zhuanlan.zhihu.com/p/87786297
class Fun:
    def __init__(self, x_list):
        """ initialize the class instance
        Args:
            x_list: data with list type
        Returns:
            None
        """
        if not isinstance(x_list, list):
            raise ValueError("input x_list is not a list type")
        self.data = x_list
        print("intialize success")
    
    def __getitem__(self, idx):
        print("__getitem__ is called")
        return self.data[idx]
    
    def __len__(self):
        print("__len__ is called")
        return len(self.data)
    
fun = Fun(x_list=[1, 2, 3, 4, 5])
print(fun[2]) # 调用获取参数的方法
print(len(fun)) # 调用获取长度的方法

