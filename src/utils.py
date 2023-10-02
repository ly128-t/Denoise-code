import torch
import os
import pickle
import yaml


def pload(*f_names):
    """Pickle load
    该函数是一个用于加载 pickle 文件的辅助函数。它接受一个或多个文件名作为参数，并将它们连接成完整的文件路径。然后，它使用 pickle 模块打开该文件，并使用 pickle.load() 函数从文件中加载对象。最后，它返回加载的对象。
    具体而言，函数的执行步骤如下：
    将传入的文件名连接成完整的文件路径，使用 os.path.join() 函数。
    使用 open() 函数以二进制模式打开文件。
    使用 pickle.load() 函数从文件中加载对象。
    关闭文件。
    返回加载的对象。
    这个函数的作用是加载 pickle 文件中保存的数据
    """
    f_name = os.path.join(*f_names) # 将 f_names 中的字符串连接起来，形成完整的文件路径，并将其赋值给变量 f_name。
    with open(f_name, "rb") as f: # 以二进制模式打开文件，r读取模式，b二进制模式
        pickle_dict = pickle.load(f) # 从文件中加载对象
    return pickle_dict

def pdump(pickle_dict, *f_names):
    """Pickle dump
    这段代码定义了一个名为 pdump 的函数，用于将给定的 Python 对象序列化并将其存储到文件中，使用了 pickle 模块。
    函数的参数如下：
    pickle_dict: 要序列化的 Python 对象，可以是字典、列表、类实例等等。
    *f_names: 一个可变参数，表示要存储序列化对象的文件路径。这个参数可以接受一个或多个字符串，os.path.join(*f_names) 将这些字符串连接起来形成一个完整的文件路径。
    函数的主要步骤如下：
    将 f_names 中的字符串连接起来，形成完整的文件路径，并将其赋值给变量 f_name。
    使用 open(f_name, "wb") 打开文件，模式为二进制写入模式，即以字节流形式写入文件。
    使用 pickle.dump(pickle_dict, f) 将 pickle_dict 序列化并将字节流写入文件 f 中。
    这样，调用 pdump 函数时，会将 pickle_dict 对象序列化，并将序列化的字节流存储到指定的文件中。
    """
    f_name = os.path.join(*f_names) # 将 f_names 中的字符串连接起来，形成完整的文件路径，并将其赋值给变量 f_name。
    with open(f_name, "wb") as f: # 以二进制写入模式打开文件，w写入模式，b二进制模式
        pickle.dump(pickle_dict, f) # 将 pickle_dict 序列化并将字节流写入文件 f 中。

def mkdir(*paths):
    """Create a directory if not existing.
    这段代码定义了一个名为 mkdir 的函数，用于创建目录（文件夹），如果目录不存在的话。它使用了 os 模块来进行文件和目录操作。
    函数的参数如下：
    *paths：一个可变参数，表示要创建的目录的路径。这个参数可以接受一个或多个字符串，os.path.join(*paths) 将这些字符串连接起来形成一个完整的目录路径。
    函数的主要步骤如下：
    将 paths 中的字符串连接起来，形成完整的目录路径，并将其赋值给变量 path。
    使用 os.path.exists(path) 检查路径 path 是否存在。
    如果路径不存在（即目录不存在），则使用 os.mkdir(path) 创建目录。
    这样，调用 mkdir 函数时，会根据提供的路径创建目录，但仅在该目录不存在的情况下才进行创建操作。如果目录已经存在，则函数不会执行任何操作。
    """
    path = os.path.join(*paths) # 将 paths 中的字符串连接起来，形成完整的目录路径，并将其赋值给变量 path。
    if not os.path.exists(path): # 检查路径 path 是否存在
        os.mkdir(path) # 创建目录

def yload(*f_names):
    """YAML load
    这段代码定义了一个名为 yload 的函数，用于加载 YAML 文件并将其解析为字典形式。它使用了 yaml 模块来处理 YAML 文件。
    函数的参数如下：
    *f_names：一个可变参数，表示要加载的 YAML 文件的路径。这个参数可以接受一个或多个字符串，os.path.join(*f_names) 将这些字符串连接起来形成一个完整的文件路径。
    函数的主要步骤如下：
    将 f_names 中的字符串连接起来，形成完整的文件路径，并将其赋值给变量 f_name。
    使用 open(f_name, 'r') 打开文件，并以只读模式进行操作。
    使用 yaml.load(f) 读取文件内容，并将其解析为字典形式，赋值给变量 yaml_dict。
    关闭文件。
    返回解析后的字典 yaml_dict。
    这样，调用 yload 函数时，会根据提供的文件路径加载相应的 YAML 文件，并返回解析后的字典数据。请注意，此代码段使用的是较旧的 yaml.load 函数，建议在使用时注意安全性，并参考 PyYAML 模块的文档以获取最新的用法和最佳实践。
    """
    f_name = os.path.join(*f_names) # 将 f_names 中的字符串连接起来，形成完整的文件路径，并将其赋值给变量 f_name。
    with open(f_name, 'r') as f: # 以只读模式打开文件，r读取模式
        yaml_dict = yaml.load(f) # 读取文件内容，并将其解析为字典形式，赋值给变量 yaml_dict。
    return yaml_dict

def ydump(yaml_dict, *f_names):
    """YAML dump
    这段代码定义了一个名为 ydump 的函数，用于将字典数据转换为 YAML 格式并将其保存到文件中。它使用了 yaml 模块来处理 YAML 文件。
    函数的参数如下：
    yaml_dict：要转换为 YAML 格式的字典数据。
    *f_names：一个可变参数，表示要保存的 YAML 文件的路径。这个参数可以接受一个或多个字符串，os.path.join(*f_names) 将这些字符串连接起来形成一个完整的文件路径。
    函数的主要步骤如下：
    将 f_names 中的字符串连接起来，形成完整的文件路径，并将其赋值给变量 f_name。
    使用 open(f_name, 'w') 打开文件，并以写入模式进行操作。
    使用 yaml.dump(yaml_dict, f, default_flow_style=False) 将字典数据转换为 YAML 格式，并将其写入文件中。
    yaml_dict 是要转换的字典数据。
    f 是文件对象，用于写入 YAML 数据。
    default_flow_style=False 参数指定以块样式而不是行内样式进行转换，使得生成的 YAML 文件更易读。
    关闭文件。
    这样，调用 ydump 函数时，会将提供的字典数据转换为 YAML 格式，并将其保存到指定的文件中。请注意，此代码段使用的是较旧的 yaml.dump 函数，建议在使用时注意安全性，并参考 PyYAML 模块的文档以获取最新的用法和最佳实践。
    """
    f_name = os.path.join(*f_names) # 将 f_names 中的字符串连接起来，形成完整的文件路径，并将其赋值给变量 f_name。
    with open(f_name, 'w') as f: # 以只写模式打开文件，w写入模式
        yaml.dump(yaml_dict, f, default_flow_style=False) # 将 yaml_dict 序列化并将字节流写入文件 f 中。

def bmv(mat, vec):
    """batch matrix vector product
    这段代码定义了一个名为 bmv 的函数，用于执行批量矩阵向量乘法操作。它使用了 PyTorch 的 torch.einsum() 函数来实现爱因斯坦求和约定。
    函数的参数如下：
    mat：表示一个批量的矩阵，维度为 (B, I, J)，其中 B 是批量大小，I 是矩阵的行数，J 是矩阵的列数。
    vec：表示一个向量，维度为 (B, J)，其中 B 是批量大小，J 是向量的长度。
    函数的主要步骤如下：
    使用 torch.einsum('bij, bj -> bi', mat, vec) 执行矩阵向量乘法操作。
    'bij, bj -> bi' 是 einsum() 函数的第一个参数，指定了求和的方式。在这个例子中，它表示对维度 j 进行求和，同时保留维度 b 和 i，得到维度为 (B, I) 的结果。
    mat 是一个形状为 (B, I, J) 的张量，表示批量的矩阵。
    vec 是一个形状为 (B, J) 的张量，表示批量的向量。
    返回矩阵向量乘法的结果，形状为 (B, I)。
    这样，调用 bmv 函数时，会对提供的批量矩阵和向量进行矩阵向量乘法操作，并返回结果。请注意，这段代码的实现使用了 PyTorch 框架中的函数和张量操作。
    """
    # torch.einsum() 函数用于实现爱因斯坦求和约定，它接受两个参数，第一个参数是一个字符串，用于指定求和的方式，第二个参数是一个或多个张量，用于指定要进行求和的张量。
    return torch.einsum('bij, bj -> bi', mat, vec)

def bbmv(mat, vec):
    """double batch matrix vector product
    这段代码定义了一个名为 bbmv 的函数，用于执行双批量矩阵向量乘法操作。同样地，它使用了 PyTorch 的 torch.einsum() 函数来实现爱因斯坦求和约定。
    函数的参数如下：
    mat：表示双批量的矩阵，维度为 (B1, B2, I, J)，其中 B1 是外层批量大小，B2 是内层批量大小，I 是矩阵的行数，J 是矩阵的列数。
    vec：表示双批量的向量，维度为 (B1, B2, J)，其中 B1 是外层批量大小，B2 是内层批量大小，J 是向量的长度。
    函数的主要步骤如下：
    使用 torch.einsum('baij, baj -> bai', mat, vec) 执行双批量矩阵向量乘法操作。
    'baij, baj -> bai' 是 einsum() 函数的第一个参数，指定了求和的方式。在这个例子中，它表示对维度 a、j 进行求和，同时保留维度 b 和 i，得到维度为 (B1, B2, I) 的结果。
    mat 是一个形状为 (B1, B2, I, J) 的张量，表示双批量的矩阵。
    vec 是一个形状为 (B1, B2, J) 的张量，表示双批量的向量。
    返回双批量矩阵向量乘法的结果，形状为 (B1, B2, I)。
    通过调用 bbmv 函数，你可以对提供的双批量矩阵和向量执行矩阵向量乘法操作，并返回结果。请注意，这段代码的实现同样使用了 PyTorch 框架中的函数和张量操作。
    """
    # torch.einsum() 函数用于实现爱因斯坦求和约定，它接受两个参数，第一个参数是一个字符串，用于指定求和的方式，第二个参数是一个或多个张量，用于指定要进行求和的张量。
    return torch.einsum('baij, baj -> bai', mat, vec)

def bmtv(mat, vec):
    """batch matrix transpose vector product
    这段代码定义了一个名为 bmtv 的函数，用于执行批量矩阵转置向量乘法操作。同样地，它使用了 PyTorch 的 torch.einsum() 函数来实现爱因斯坦求和约定。
    函数的参数如下：
    mat：表示批量的矩阵，维度为 (B, J, I)，其中 B 是批量大小，J 是矩阵的列数，I 是矩阵的行数。
    vec：表示批量的向量，维度为 (B, J)，其中 B 是批量大小，J 是向量的长度。
    函数的主要步骤如下：
    使用 torch.einsum('bji, bj -> bi', mat, vec) 执行批量矩阵转置向量乘法操作。
    'bji, bj -> bi' 是 einsum() 函数的第一个参数，指定了求和的方式。在这个例子中，它表示对维度 j 进行求和，同时保留维度 b 和 i，得到维度为 (B, I) 的结果。
    mat 是一个形状为 (B, J, I) 的张量，表示批量的矩阵。
    vec 是一个形状为 (B, J) 的张量，表示批量的向量。
    返回批量矩阵转置向量乘法的结果，形状为 (B, I)。
    通过调用 bmtv 函数，你可以对提供的批量矩阵和向量执行矩阵转置向量乘法操作，并返回结果。请注意，这段代码的实现同样使用了 PyTorch 框架中的函数和张量操作。
    """
    # torch.einsum() 函数用于实现爱因斯坦求和约定，它接受两个参数，第一个参数是一个字符串，用于指定求和的方式，第二个参数是一个或多个张量，用于指定要进行求和的张量。
    return torch.einsum('bji, bj -> bi', mat, vec)

def bmtm(mat1, mat2):
    """batch matrix transpose matrix product
    这段代码定义了一个名为 bmtm 的函数，用于执行批量矩阵转置矩阵乘法操作。同样地，它使用了 PyTorch 的 torch.einsum() 函数来实现爱因斯坦求和约定。
    函数的参数如下：
    mat1：表示批量的第一个矩阵，维度为 (B, J, I)，其中 B 是批量大小，J 是第一个矩阵的列数，I 是第一个矩阵的行数。
    mat2：表示批量的第二个矩阵，维度为 (B, J, K)，其中 B 是批量大小，J 是第二个矩阵的列数，K 是第二个矩阵的行数。
    函数的主要步骤如下：
    使用 torch.einsum('bji, bjk -> bik', mat1, mat2) 执行批量矩阵转置矩阵乘法操作。
    'bji, bjk -> bik' 是 einsum() 函数的第一个参数，指定了求和的方式。在这个例子中，它表示对维度 j 进行求和，同时保留维度 b、i 和 k，得到维度为 (B, I, K) 的结果。
    mat1 是一个形状为 (B, J, I) 的张量，表示批量的第一个矩阵。
    mat2 是一个形状为 (B, J, K) 的张量，表示批量的第二个矩阵。
    返回批量矩阵转置矩阵乘法的结果，形状为 (B, I, K)。
    通过调用 bmtm 函数，你可以对提供的批量矩阵执行转置矩阵乘法操作，并返回结果。请注意，这段代码的实现同样使用了 PyTorch 框架中的函数和张量操作。
    """
    # torch.einsum() 函数用于实现爱因斯坦求和约定，它接受两个参数，第一个参数是一个字符串，用于指定求和的方式，第二个参数是一个或多个张量，用于指定要进行求和的张量。
    return torch.einsum("bji, bjk -> bik", mat1, mat2)

def bmmt(mat1, mat2):
    """batch matrix matrix transpose product
    这段代码定义了一个名为 bmmt 的函数，用于执行批量矩阵矩阵转置乘法操作。它同样使用了 PyTorch 的 torch.einsum() 函数来实现爱因斯坦求和约定。
    函数的参数如下：
    mat1：表示批量的第一个矩阵，维度为 (B, I, J)，其中 B 是批量大小，I 是第一个矩阵的行数，J 是第一个矩阵的列数。
    mat2：表示批量的第二个矩阵，维度为 (B, K, J)，其中 B 是批量大小，K 是第二个矩阵的行数，J 是第二个矩阵的列数。
    函数的主要步骤如下：
    使用 torch.einsum('bij, bkj -> bik', mat1, mat2) 执行批量矩阵矩阵转置乘法操作。
    'bij, bkj -> bik' 是 einsum() 函数的第一个参数，指定了求和的方式。在这个例子中，它表示对维度 j 进行求和，同时保留维度 b、i 和 k，得到维度为 (B, I, K) 的结果。
    mat1 是一个形状为 (B, I, J) 的张量，表示批量的第一个矩阵。
    mat2 是一个形状为 (B, K, J) 的张量，表示批量的第二个矩阵的转置。
    返回批量矩阵矩阵转置乘法的结果，形状为 (B, I, K)。
    通过调用 bmmt 函数，你可以对提供的批量矩阵执行矩阵转置乘法操作，并返回结果。请注意，这段代码的实现同样使用了 PyTorch 框架中的函数和张量操作。
    """
    # torch.einsum() 函数用于实现爱因斯坦求和约定，它接受两个参数，第一个参数是一个字符串，用于指定求和的方式，第二个参数是一个或多个张量，用于指定要进行求和的张量。
    return torch.einsum("bij, bkj -> bik", mat1, mat2)
