from src.utils import *
import numpy as np


class SO3:
    #  tolerance criterion
    """
    代码中的类属性包括：
    TOL：一个容差标准，用于比较浮点数的相等性。
    Id：一个3x3的单位矩阵，使用CUDA加速，并且数据类型为单精度浮点数。
    dId：一个3x3的单位矩阵，同样使用CUDA加速，但数据类型为双精度浮点数。
    """
    TOL = 1e-8
    Id = torch.eye(3).cuda().float()
    dId = torch.eye(3).cuda().double()

    @classmethod
    def exp(cls, phi):
        """
        这个方法的作用是计算李群SO(3)中的指数映射，将李代数中的旋转向量转换为旋转矩阵。
        angle = phi.norm(dim=1, keepdim=True)：计算输入旋转向量phi的范数（欧几里得范数），dim=1表示按行计算，keepdim=True表示保持维度不变，即保持为列向量。
        mask = angle[:, 0] < cls.TOL：根据范数的值与cls.TOL的比较，生成一个布尔掩码，用于判断旋转向量是否接近零。cls.TOL是一个阈值，用于定义接近零的条件。
        dim_batch = phi.shape[0]：获取输入旋转向量phi的批次大小，即旋转向量的数量。
        Id = cls.Id.expand(dim_batch, 3, 3)：将单位矩阵cls.Id扩展为与输入旋转向量相同大小的矩阵，形状为(dim_batch, 3, 3)。
        axis = phi[~mask] / angle[~mask]：根据掩码，选择非零范数的旋转向量，并计算归一化的旋转轴。
        c = angle[~mask].cos().unsqueeze(2)：根据掩码，计算非零范数的旋转向量的余弦值，并在第三维上增加一个维度。
        s = angle[~mask].sin().unsqueeze(2)：根据掩码，计算非零范数的旋转向量的正弦值，并在第三维上增加一个维度。
        Rot = phi.new_empty(dim_batch, 3, 3)：创建一个与输入旋转向量相同类型的空矩阵，形状为(dim_batch, 3, 3)。
        Rot[mask] = Id[mask] + SO3.wedge(phi[mask])：根据掩码，
        """
        angle = phi.norm(dim=1, keepdim=True)
        mask = angle[:, 0] < cls.TOL
        dim_batch = phi.shape[0]
        Id = cls.Id.expand(dim_batch, 3, 3)

        axis = phi[~mask] / angle[~mask]
        c = angle[~mask].cos().unsqueeze(2)
        s = angle[~mask].sin().unsqueeze(2)

        Rot = phi.new_empty(dim_batch, 3, 3)
        Rot[mask] = Id[mask] + SO3.wedge(phi[mask])
        Rot[~mask] = c*Id[~mask] + \
            (1-c)*cls.bouter(axis, axis) + s*cls.wedge(axis)
        return Rot

    @classmethod
    def log(cls, Rot):
        """
        这段代码用于计算旋转矩阵到李代数空间的映射，根据旋转矩阵的性质和角度值的大小，分别采用不同的计算方式来得到李代数向量。
        这段代码的功能：
        dim_batch = Rot.shape[0]：获取输入矩阵Rot的批次维度大小。
        Id = cls.Id.expand(dim_batch, 3, 3)：将类属性Id扩展为与输入矩阵相同大小的单位矩阵。
        cos_angle = (0.5 * cls.btrace(Rot) - 0.5).clamp(-1., 1.)：计算旋转矩阵Rot的迹的一半减去0.5，然后使用clamp函数将值限制在[-1, 1]范围内，得到余弦值。
        angle = cos_angle.acos()：计算余弦值的反余弦，得到角度值。
        mask = angle < cls.TOL：创建一个布尔掩码，用于标记角度值小于阈值cls.TOL的元素。
        if mask.sum() == 0:：如果没有角度值小于阈值的元素。
        angle = angle.unsqueeze(1).unsqueeze(1)：将角度值扩展为3x3维度。
        return cls.vee((0.5 * angle/angle.sin())*(Rot-Rot.transpose(1, 2)))：计算旋转矩阵的差值，并进行一系列的乘法和除法运算，最后返回结果。
        elif mask.sum() == dim_batch:：如果所有角度值都小于阈值。
        return cls.vee(Rot - Id)：直接计算旋转矩阵与单位矩阵的差值，并返回结果。
        phi = cls.vee(Rot - Id)：计算旋转矩阵与单位矩阵的差值，并得到对应的李代数向量。
        angle = angle：将角度值赋给新的变量angle。
        phi[~mask] = cls.vee((0.5 * angle[~mask]/angle[~mask].sin()).unsqueeze(1).unsqueeze(2)*(Rot[~mask] - Rot[~mask].transpose(1, 2)))：
        对于角度值大于等于阈值的元素，根据一系列的乘法和除法运算计算对应的李代数向量。
        return phi：返回最终的李代数向量。      
        """
        dim_batch = Rot.shape[0]
        Id = cls.Id.expand(dim_batch, 3, 3)

        cos_angle = (0.5 * cls.btrace(Rot) - 0.5).clamp(-1., 1.)
        # Clip cos(angle) to its proper domain to avoid NaNs from rounding
        # errors
        angle = cos_angle.acos()
        mask = angle < cls.TOL
        if mask.sum() == 0:
            angle = angle.unsqueeze(1).unsqueeze(1)
            return cls.vee((0.5 * angle/angle.sin())*(Rot-Rot.transpose(1, 2)))
        elif mask.sum() == dim_batch:
            # If angle is close to zero, use first-order Taylor expansion
            return cls.vee(Rot - Id)
        phi = cls.vee(Rot - Id)
        angle = angle
        phi[~mask] = cls.vee((0.5 * angle[~mask]/angle[~mask].sin()).unsqueeze(
            1).unsqueeze(2)*(Rot[~mask] - Rot[~mask].transpose(1, 2)))
        return phi

    @staticmethod
    def vee(Phi):
        """
        这段代码定义了一个函数vee，它接受一个输入参数Phi，这个参数是一个形状为(batch_size, 3, 3)的张量。
        函数的功能是将输入的3x3矩阵展开成一个3维向量。具体实现如下：
        使用torch.stack函数将矩阵中的特定元素按照指定的维度进行堆叠。
        通过索引操作，取出Phi张量中的特定位置的元素，分别是第3行第2列、第1行第3列和第2行第1列的元素。
        将这些元素按照顺序堆叠起来，形成一个形状为(batch_size, 3)的输出张量。
        这个函数的作用是在某些情况下将旋转矩阵转换为李代数向量，用于进行旋转相关的计算或操作。
        """
        return torch.stack((Phi[:, 2, 1],
                            Phi[:, 0, 2],
                            Phi[:, 1, 0]), dim=1)

    @staticmethod
    def wedge(phi):
        """
        phi：形状为 (dim_batch, 3) 的张量，表示旋转向量。
        函数的作用是根据旋转向量 phi 进行 Wedge 运算，得到一个 3x3 的反对称矩阵。
        具体而言，根据旋转向量的每个分量构造出一个反对称矩阵，然后将这些矩阵堆叠在一起形成一个形状为 (dim_batch, 3, 3) 的张量。        
        """
        dim_batch = phi.shape[0]
        zero = phi.new_zeros(dim_batch)
        return torch.stack((zero, -phi[:, 2], phi[:, 1],
                            phi[:, 2], zero, -phi[:, 0],
                            -phi[:, 1], phi[:, 0], zero), 1).view(dim_batch,
                            3, 3)

    @classmethod
    def from_rpy(cls, roll, pitch, yaw):
        """
        根据欧拉角（Roll、Pitch、Yaw）创建旋转矩阵。

        函数签名如下：

        def from_rpy(cls, roll, pitch, yaw):
            return cls.rotz(yaw).bmm(cls.roty(pitch).bmm(cls.rotx(roll)))
        参数说明：

        roll：Roll 角度值，表示绕 X 轴旋转的角度。
        pitch：Pitch 角度值，表示绕 Y 轴旋转的角度。
        yaw：Yaw 角度值，表示绕 Z 轴旋转的角度。
        函数的作用是根据给定的欧拉角值，按照 Roll-Pitch-Yaw 的顺序创建对应的旋转矩阵。
        """
        return cls.rotz(yaw).bmm(cls.roty(pitch).bmm(cls.rotx(roll)))

    @classmethod
    def rotx(cls, angle_in_radians):
        """
        用于创建绕 X 轴旋转的旋转矩阵。
        参数说明：
        angle_in_radians：绕 X 轴旋转的角度（弧度）。
        """
        c = angle_in_radians.cos()
        s = angle_in_radians.sin()
        mat = c.new_zeros((c.shape[0], 3, 3))
        mat[:, 0, 0] = 1
        mat[:, 1, 1] = c
        mat[:, 2, 2] = c
        mat[:, 1, 2] = -s
        mat[:, 2, 1] = s
        return mat

    @classmethod
    def roty(cls, angle_in_radians):
        """
        用于创建绕 Y 轴旋转的旋转矩阵。
        参数说明：
        angle_in_radians：绕 Y 轴旋转的角度（弧度）。
        """
        c = angle_in_radians.cos()
        s = angle_in_radians.sin()
        mat = c.new_zeros((c.shape[0], 3, 3))
        mat[:, 1, 1] = 1
        mat[:, 0, 0] = c
        mat[:, 2, 2] = c
        mat[:, 0, 2] = s
        mat[:, 2, 0] = -s
        return mat

    @classmethod
    def rotz(cls, angle_in_radians):
        """
         用于创建绕 Z 轴旋转的旋转矩阵。
        参数说明：
        angle_in_radians：绕 Z 轴旋转的角度（弧度）。       
        """
        c = angle_in_radians.cos()
        s = angle_in_radians.sin()
        mat = c.new_zeros((c.shape[0], 3, 3))
        mat[:, 2, 2] = 1
        mat[:, 0, 0] = c
        mat[:, 1, 1] = c
        mat[:, 0, 1] = -s
        mat[:, 1, 0] = s
        return mat

    @classmethod
    def isclose(cls, x, y):
        """
        isclose 函数是一个静态方法，它属于类 cls（这里 cls 是一个占位符，表示类的名称，具体的类名称需要根据上下文来确定）。
        这个方法用于比较两个数值 x 和 y 是否接近（近似相等）。
        在方法中，它使用了类的属性 TOL，该属性是一个容差值，在该类中是1e-8，用于确定接近性的阈值。
        方法通过计算 x 和 y 的差值的绝对值，然后与容差值进行比较，判断它们是否小于容差值。
        如果差值的绝对值小于容差值，则返回 True，表示 x 和 y 是接近的；否则返回 False，表示 x 和 y 不接近。
        这个方法可以用于比较浮点数或其他数值类型的接近性，避免由于精度问题导致的直接相等比较的不准确性。
        通过设定适当的容差值，可以灵活地控制接近性的定义。
        """
        return (x-y).abs() < cls.TOL

    @classmethod
    def to_rpy(cls, Rots):
        """Convert a rotation matrix to RPY Euler angles.
            将旋转矩阵转换为欧拉角（Roll、Pitch、Yaw）。
        """

        pitch = torch.atan2(-Rots[:, 2, 0],
            torch.sqrt(Rots[:, 0, 0]**2 + Rots[:, 1, 0]**2))
        yaw = pitch.new_empty(pitch.shape)
        roll = pitch.new_empty(pitch.shape)

        near_pi_over_two_mask = cls.isclose(pitch, np.pi / 2.)
        near_neg_pi_over_two_mask = cls.isclose(pitch, -np.pi / 2.)

        remainder_inds = ~(near_pi_over_two_mask | near_neg_pi_over_two_mask)

        yaw[near_pi_over_two_mask] = 0
        roll[near_pi_over_two_mask] = torch.atan2(
            Rots[near_pi_over_two_mask, 0, 1],
            Rots[near_pi_over_two_mask, 1, 1])

        yaw[near_neg_pi_over_two_mask] = 0.
        roll[near_neg_pi_over_two_mask] = -torch.atan2(
            Rots[near_neg_pi_over_two_mask, 0, 1],
            Rots[near_neg_pi_over_two_mask, 1, 1])

        sec_pitch = 1/pitch[remainder_inds].cos()
        remainder_mats = Rots[remainder_inds]
        yaw = torch.atan2(remainder_mats[:, 1, 0] * sec_pitch,
                          remainder_mats[:, 0, 0] * sec_pitch)
        roll = torch.atan2(remainder_mats[:, 2, 1] * sec_pitch,
                           remainder_mats[:, 2, 2] * sec_pitch)
        rpys = torch.cat([roll.unsqueeze(dim=1),
                        pitch.unsqueeze(dim=1),
                        yaw.unsqueeze(dim=1)], dim=1)
        return rpys

    @classmethod
    def from_quaternion(cls, quat, ordering='wxyz'):
        """Form a rotation matrix from a unit length quaternion.
        Valid orderings are 'xyzw' and 'wxyz'.
        根据单位长度的四元数（quaternion）形成旋转矩阵（rotation matrix）。
        根据指定的顺序（'xyzw' 或 'wxyz'）提取四元数的各个分量。
        如果顺序是 'xyzw'，则分别提取 qx、qy、qz、qw 分量；
        如果顺序是 'wxyz'，则分别提取 qw、qx、qy、qz 分量。
        """
        if ordering == 'xyzw':
            qx = quat[:, 0]
            qy = quat[:, 1]
            qz = quat[:, 2]
            qw = quat[:, 3]
        elif ordering == 'wxyz':
            qw = quat[:, 0]
            qx = quat[:, 1]
            qy = quat[:, 2]
            qz = quat[:, 3]

        # Form the matrix
        mat = quat.new_empty(quat.shape[0], 3, 3)

        qx2 = qx * qx
        qy2 = qy * qy
        qz2 = qz * qz

        mat[:, 0, 0] = 1. - 2. * (qy2 + qz2)
        mat[:, 0, 1] = 2. * (qx * qy - qw * qz)
        mat[:, 0, 2] = 2. * (qw * qy + qx * qz)

        mat[:, 1, 0] = 2. * (qw * qz + qx * qy)
        mat[:, 1, 1] = 1. - 2. * (qx2 + qz2)
        mat[:, 1, 2] = 2. * (qy * qz - qw * qx)

        mat[:, 2, 0] = 2. * (qx * qz - qw * qy)
        mat[:, 2, 1] = 2. * (qw * qx + qy * qz)
        mat[:, 2, 2] = 1. - 2. * (qx2 + qy2)
        return mat

    @classmethod
    def to_quaternion(cls, Rots, ordering='wxyz'):
        """Convert a rotation matrix to a unit length quaternion.
        Valid orderings are 'xyzw' and 'wxyz'.
        该方法用于将旋转矩阵（rotation matrix）转换为单位长度的四元数（quaternion）。
        根据指定的顺序（'xyzw' 或 'wxyz'）创建四元数的分量 qx、qy、qz、qw 或 qw、qx、qy、qz。
        """
        tmp = 1 + Rots[:, 0, 0] + Rots[:, 1, 1] + Rots[:, 2, 2]
        tmp[tmp < 0] = 0
        qw = 0.5 * torch.sqrt(tmp)
        qx = qw.new_empty(qw.shape[0])
        qy = qw.new_empty(qw.shape[0])
        qz = qw.new_empty(qw.shape[0])

        near_zero_mask = qw.abs() < cls.TOL

        if near_zero_mask.sum() > 0:
            cond1_mask = near_zero_mask * \
                (Rots[:, 0, 0] > Rots[:, 1, 1])*(Rots[:, 0, 0] > Rots[:, 2, 2])
            cond1_inds = cond1_mask.nonzero()

            if len(cond1_inds) > 0:
                cond1_inds = cond1_inds.squeeze()
                R_cond1 = Rots[cond1_inds].view(-1, 3, 3)
                d = 2. * torch.sqrt(1. + R_cond1[:, 0, 0] -
                    R_cond1[:, 1, 1] - R_cond1[:, 2, 2]).view(-1)
                qw[cond1_inds] = (R_cond1[:, 2, 1] - R_cond1[:, 1, 2]) / d
                qx[cond1_inds] = 0.25 * d
                qy[cond1_inds] = (R_cond1[:, 1, 0] + R_cond1[:, 0, 1]) / d
                qz[cond1_inds] = (R_cond1[:, 0, 2] + R_cond1[:, 2, 0]) / d

            cond2_mask = near_zero_mask * (Rots[:, 1, 1] > Rots[:, 2, 2])
            cond2_inds = cond2_mask.nonzero()

            if len(cond2_inds) > 0:
                cond2_inds = cond2_inds.squeeze()
                R_cond2 = Rots[cond2_inds].view(-1, 3, 3)
                d = 2. * torch.sqrt(1. + R_cond2[:, 1, 1] -
                                R_cond2[:, 0, 0] - R_cond2[:, 2, 2]).squeeze()
                tmp = (R_cond2[:, 0, 2] - R_cond2[:, 2, 0]) / d
                qw[cond2_inds] = tmp
                qx[cond2_inds] = (R_cond2[:, 1, 0] + R_cond2[:, 0, 1]) / d
                qy[cond2_inds] = 0.25 * d
                qz[cond2_inds] = (R_cond2[:, 2, 1] + R_cond2[:, 1, 2]) / d

            cond3_mask = near_zero_mask & cond1_mask.logical_not() & cond2_mask.logical_not()
            cond3_inds = cond3_mask

            if len(cond3_inds) > 0:
                R_cond3 = Rots[cond3_inds].view(-1, 3, 3)
                d = 2. * \
                    torch.sqrt(1. + R_cond3[:, 2, 2] -
                    R_cond3[:, 0, 0] - R_cond3[:, 1, 1]).squeeze()
                qw[cond3_inds] = (R_cond3[:, 1, 0] - R_cond3[:, 0, 1]) / d
                qx[cond3_inds] = (R_cond3[:, 0, 2] + R_cond3[:, 2, 0]) / d
                qy[cond3_inds] = (R_cond3[:, 2, 1] + R_cond3[:, 1, 2]) / d
                qz[cond3_inds] = 0.25 * d

        far_zero_mask = near_zero_mask.logical_not()
        far_zero_inds = far_zero_mask
        if len(far_zero_inds) > 0:
            R_fz = Rots[far_zero_inds]
            d = 4. * qw[far_zero_inds]
            qx[far_zero_inds] = (R_fz[:, 2, 1] - R_fz[:, 1, 2]) / d
            qy[far_zero_inds] = (R_fz[:, 0, 2] - R_fz[:, 2, 0]) / d
            qz[far_zero_inds] = (R_fz[:, 1, 0] - R_fz[:, 0, 1]) / d

        # Check ordering last
        if ordering == 'xyzw':
            quat = torch.stack([qx, qy, qz, qw], dim=1)
        elif ordering == 'wxyz':
            quat = torch.stack([qw, qx, qy, qz], dim=1)
        return quat

    @classmethod
    def normalize(cls, Rots):
        """
        单精度flout旋转矩阵的归一化。
        参数：
            cls：类对象
            Rots：旋转矩阵，形状为（batch_size, 3, 3）
        返回值：
            归一化后的旋转矩阵，形状为（batch_size, 3, 3）
        """
        U, _, V = torch.svd(Rots)
        S = cls.Id.clone().repeat(Rots.shape[0], 1, 1)
        S[:, 2, 2] = torch.det(U) * torch.det(V)
        return U.bmm(S).bmm(V.transpose(1, 2))

    @classmethod
    def dnormalize(cls, Rots):
        """
        双精度double旋转矩阵进行归一化处理。
        参数：
            cls：类对象
            Rots：旋转矩阵，形状为（batch_size, 3, 3）
        返回值：
            归一化后的旋转矩阵，形状为（batch_size, 3, 3）
        """
        U, _, V = torch.svd(Rots)
        S = cls.dId.clone().repeat(Rots.shape[0], 1, 1)
        S[:, 2, 2] = torch.det(U) * torch.det(V)
        return U.bmm(S).bmm(V.transpose(1, 2))

    @classmethod
    def qmul(cls, q, r, ordering='wxyz'):
        """
        Multiply quaternion(s) q with quaternion(s) r.
        将四元数 q 与四元数 r 相乘。
        参数：
            cls：类对象
            q：四元数，形状为（batch_size, 4）
            r：四元数，形状为（batch_size, 4）
            ordering：四元数的顺序，可选值为 'wxyz' 和 'xyzw'
            
        返回值：
            乘积后的四元数，形状为（batch_size, 4）
        """
        terms = cls.bouter(r, q)
        w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
        x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
        y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
        z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
        xyz = torch.stack((x, y, z), dim=1)
        xyz[w < 0] *= -1
        w[w < 0] *= -1
        if ordering == 'wxyz':
            q = torch.cat((w.unsqueeze(1), xyz), dim=1)
        else:
            q = torch.cat((xyz, w.unsqueeze(1)), dim=1)
        return q / q.norm(dim=1, keepdim=True)

    @staticmethod
    def sinc(x):
        """
        计算 sinc 函数的值。
        参数：
            x：输入张量
        返回值：
            sinc 函数的值，形状与输入张量相同
        """
        return x.sin() / x

    @classmethod
    def qexp(cls, xi, ordering='wxyz'):
        """
        Convert exponential maps to quaternions.
        将指数映射转换为四元数。
        参数：
            xi：输入张量，表示指数映射
            ordering：字符串，表示四元数的顺序，默认为'wxyz'
        返回值：
            转换后的四元数，与输入张量的形状相同
        """
        theta = xi.norm(dim=1, keepdim=True)
        w = (0.5*theta).cos()
        xyz = 0.5*cls.sinc(0.5*theta/np.pi)*xi
        return torch.cat((w, xyz), 1)

    @classmethod
    def qlog(cls, q, ordering='wxyz'):
        """
        Applies the log map to quaternions.
        将指数映射转换为四元数。
        参数：
            xi：输入张量，表示指数映射
            ordering：字符串，表示四元数的顺序，默认为'wxyz'
        返回值：
            转换后的四元数，与输入张量的形状相同
        """
        n = 0.5*torch.norm(q[:, 1:], p=2, dim=1, keepdim=True)
        n = torch.clamp(n, min=1e-8)
        q = q[:, 1:] * torch.acos(torch.clamp(q[:, :1], min=-1.0, max=1.0))
        r = q / n
        return r

    @classmethod
    def qinv(cls, q, ordering='wxyz'):
        """Quaternion inverse
        四元数的逆。
        参数：
            q：输入张量，表示四元数
            ordering：字符串，表示四元数的顺序，默认为'wxyz'
        返回值：
            四元数的逆，与输入张量的形状相同       
        """
        r = torch.empty_like(q)
        if ordering == 'wxyz':
            r[:, 1:4] = -q[:, 1:4]
            r[:, 0] = q[:, 0]
        else:
            r[:, :3] = -q[:, :3]
            r[:, 3] = q[:, 3]
        return r

    @classmethod
    def qnorm(cls, q):
        """Quaternion normalization
        四元数的归一化。
        参数：
            q：输入张量，表示四元数
            
        返回值：
            归一化后的四元数，与输入张量的形状相同
        """
        return q / q.norm(dim=1, keepdim=True)

    @classmethod
    def qinterp(cls, qs, t, t_int):
        """Quaternion interpolation
        四元数的插值。重点，涉及之后的旋转矩阵R的计算
        参数：
            qs：输入张量，表示一系列的四元数
            t：输入张量，表示时间序列
            t_int：输入张量，表示要进行插值的时间点
        返回值：
            插值后的四元数，与t_int的形状相同
        """
        idxs = np.searchsorted(t, t_int)
        idxs0 = idxs-1
        idxs0[idxs0 < 0] = 0
        idxs1 = idxs
        idxs1[idxs1 == t.shape[0]] = t.shape[0] - 1
        q0 = qs[idxs0]
        q1 = qs[idxs1]
        tau = torch.zeros_like(t_int)
        dt = (t[idxs1]-t[idxs0])[idxs0 != idxs1]
        tau[idxs0 != idxs1] = (t_int-t[idxs0])[idxs0 != idxs1]/dt
        return cls.slerp(q0, q1, tau)

    @classmethod
    def slerp(cls, q0, q1, tau, DOT_THRESHOLD = 0.9995):
        """
        Spherical linear interpolation.
        球面线性插值。
        参数：
            q0：输入张量，表示起始四元数
            q1：输入张量，表示结束四元数
            tau：输入张量，表示插值参数
            DOT_THRESHOLD：浮点数，表示点积阈值，默认为0.9995
            
        返回值：
            插值后的四元数
        """

        dot = (q0*q1).sum(dim=1)
        q1[dot < 0] = -q1[dot < 0]
        dot[dot < 0] = -dot[dot < 0]

        q = torch.zeros_like(q0)
        tmp = q0 + tau.unsqueeze(1) * (q1 - q0)
        tmp = tmp[dot > DOT_THRESHOLD]
        q[dot > DOT_THRESHOLD] = tmp / tmp.norm(dim=1, keepdim=True)

        theta_0 = dot.acos()
        sin_theta_0 = theta_0.sin()
        theta = theta_0 * tau
        sin_theta = theta.sin()
        s0 = (theta.cos() - dot * sin_theta / sin_theta_0).unsqueeze(1)
        s1 = (sin_theta / sin_theta_0).unsqueeze(1)
        q[dot < DOT_THRESHOLD] = ((s0 * q0) + (s1 * q1))[dot < DOT_THRESHOLD]
        return q / q.norm(dim=1, keepdim=True)

    @staticmethod
    def bouter(vec1, vec2):
        """
        batch outer product
        批量外积。
        参数：
            vec1：输入张量，表示第一个向量
            vec2：输入张量，表示第二个向量
        返回值：
            外积结果张量
        bouter方法实现了批量外积。它接受两个输入张量：vec1表示第一个向量，vec2表示第二个向量。
        函数通过torch.einsum函数计算两个向量的外积，返回外积结果张量。
        """
        return torch.einsum('bi, bj -> bij', vec1, vec2)

    @staticmethod
    def btrace(mat):
        """
        batch matrix trace
        批量矩阵迹。
        参数：
            mat：输入张量，表示批量矩阵
        返回值：
            迹结果张量
        btrace方法实现了批量矩阵迹。它接受一个输入张量mat，表示批量矩阵。
        函数通过torch.einsum函数计算每个矩阵的迹，并返回迹结果张量。        
        """
        return torch.einsum('bii -> b', mat)


class CPUSO3:
    # tolerance criterion
    TOL = 1e-8
    Id = torch.eye(3)

    @classmethod
    def qmul(cls, q, r):
        """
        Multiply quaternion(s) q with quaternion(s) r.
        将四元数 q 与四元数 r 相乘。
        """
        # Compute outer product
        terms = cls.outer(r, q)
        w = terms[0, 0] - terms[1, 1] - terms[2, 2] - terms[3, 3]
        x = terms[0, 1] + terms[1, 0] - terms[2, 3] + terms[3, 2]
        y = terms[0, 2] + terms[1, 3] + terms[2, 0] - terms[3, 1]
        z = terms[0, 3] - terms[1, 2] + terms[2, 1] + terms[3, 0]
        return torch.stack((w, x, y, z))

    @staticmethod
    def outer(a, b):
        return torch.einsum('i, j -> ij', a, b)