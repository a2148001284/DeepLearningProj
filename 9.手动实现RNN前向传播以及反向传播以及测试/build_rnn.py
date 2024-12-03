import numpy as np
from utils import *

#单个cell前向传播的过程
def rnn_cell_forward(x_t, s_prev, parameters):
    """
    单个RNN-cell 的前向传播过程
    :param x_t: 单元的输入
    :param s_prev: 上一个单元的输入
    :param parameters: 单元中的参数
    :return: s_next, out_pred, cache
    """
    # 获取参数
    U = parameters["U"]
    W = parameters["W"]
    V = parameters["V"]
    ba = parameters["ba"]
    by = parameters["by"]

    # 计算激活函数
    s_next = np.tanh(np.dot(U, x_t) + np.dot(W, s_prev) + ba)
    # 计算当前cell输出预测结果
    out_pred = softmax(np.dot(V, s_next) + by)

    # 存储当前单元的结果
    cache = (s_next, s_prev, x_t, parameters)  #后续反向传播计算梯度时需要使用到这些参数 所以每层都存一下自己对应的数值

    return s_next, out_pred, cache

def test_rnn_cell_forward():  #测试单个cell前向传播的函数
    np.random.seed(1)

    x_t = np.random.randn(3, 1)
    s_prev = np.random.randn(5, 1)
    U = np.random.randn(5, 3)
    W = np.random.randn(5, 5)

    V = np.random.randn(3, 5)
    ba = np.random.randn(5, 1)
    by = np.random.randn(3, 1)
    parameters = {"U": U, "W": W, "V": V, "ba": ba, "by": by}

    s_next, out_pred, cache = rnn_cell_forward(x_t, s_prev, parameters)
    print("s_next = ", s_next)
    print("s_next.shape = ", s_next.shape)
    print("out_pred =", out_pred)
    print("out_pred.shape = ", out_pred.shape)

# test_rnn_cell_forward()

#所有cell前向传播的过程
def rnn_forward(x, s0, parameters):
    """
    对多个Cell的RNN进行前向传播
    :param x: T个时刻的X总输入形状  序列的输入
    :param s0: 隐层第一次输入
    :param parameters: 参数
    :return: s, y, caches

    初始的输入为序列的输入x和初始参数s0
    输出为所有隐层的输出序列s以及cell的输出序列y
    """
    # 初始化缓存
    caches = []

    # 根据X输入的形状确定cell的个数(3, 1, T)  每一个时刻都是3，1的形状 共有T个时刻
    # m是词的个数，n为自定义数字：(3, 5)
    m, _, T = x.shape
    # 根据输出
    m, n = parameters["V"].shape  #m为输入的词的个数 n为可控的字典内词的数量，自定义数值

    # 初始化所有cell的S，用于保存所有cell的隐层结果
    # 初始化所有cell的输出y，保存所有输出结果
    s = np.zeros((n, 1, T))
    y = np.zeros((m, 1, T))

    # 初始化第一个输入s_0
    s_next = s0

    # 根据cell的个数循环,并保存每组的
    for t in range(T):  #T也可以指代cell的个数 表示循环一轮
        # 更新每个隐层的输出计算结果，s,o,cache
        s_next, out_pred, cache = rnn_cell_forward(x[:, :, t], s_next, parameters)
        # 保存隐层的输出值s_next
        s[:, :, t] = s_next
        # 保存cell的预测值out_pred
        y[:, :, t] = out_pred
        # 保存每个cell缓存结果
        caches.append(cache)

    return s, y, caches

def test_run_forward():
    np.random.seed(1)
    # 定义了4个cell，每个词形状(3, 1)  一共词典里的个数比如为3个 那么每个输入x都为3x1的概率one-hot编码矩阵 共有四个输入 故为3x1x4
    #一共有四个输出 每个输出都有一个概率值 为3x1预测 因为词典有三个词
    x = np.random.randn(3, 1, 4)
    s0 = np.random.randn(5, 1)

    W = np.random.randn(5, 5)
    U = np.random.randn(5, 3)
    V = np.random.randn(3, 5)
    ba = np.random.randn(5, 1)
    by = np.random.randn(3, 1)
    parameters = {"U": U, "W": W, "V": V, "ba": ba, "by": by}

    s, y, caches = rnn_forward(x, s0, parameters)
    print("s = ", s)
    print("s.shape = ", s.shape)
    print("y =", y)
    print("y.shape = ", y.shape)
    # print(y[2][0][2])

# test_run_forward()

#单个cell的反向传播计算梯度
def rnn_cell_backward(ds_next, cache):
    """
    对单个cell进行反向传播
    :param ds_next: 当前隐层输出结果相对于损失的导数
    :param cache: 每个cell的缓存
    :return:
    """

    # 获取缓存值
    (s_next, s_prev, x_t, parameters) = cache
    print(type(parameters))

    # 获取参数
    U = parameters["U"]
    W = parameters["W"]
    V = parameters["V"]
    ba = parameters["ba"]
    by = parameters["by"]

    # 计算tanh的梯度通过对s_next
    dtanh = (1 - s_next ** 2) * ds_next

    # 计算U的梯度值
    dU = np.dot(dtanh, x_t.T)

    dx_t = np.dot(U.T, dtanh)

    # 计算W的梯度值
    dW = np.dot(dtanh, s_prev.T)

    ds_prev = np.dot(W.T, dtanh)

    # 计算b的梯度
    dba = np.sum(dtanh, axis=1, keepdims=1)

    # 梯度字典
    gradients = {"dx_t": dx_t, "ds_prev": ds_prev, "dU": dU, "dW": dW, "dba": dba}

    return gradients

#整个网络的反向传播
def rnn_backward(ds, caches):
    """
    对给定的一个序列进行RNN的发现反向传播
    :param ds:
    :param caches:
    :return:
    """

    # 获取第一个cell的数据,参数，输入输出值
    (s1, s0, x_1, parameters) = caches[0]

    # 获取总共cell的数量以及m和n的值  m为输入的词数量 n为词典的词数  cell的数量为T
    n, _, T = ds.shape
    m, _ = x_1.shape

    # 初始化梯度值  仅仅是为了便于测试
    dx = np.zeros((m, 1, T))
    dU = np.zeros((n, m))
    dW = np.zeros((n, n))
    dba = np.zeros((n, 1))
    ds0 = np.zeros((n, 1))
    ds_prevt = np.zeros((n, 1))

    # 循环从后往前进行反向传播
    for t in reversed(range(T)):
        # 根据时间T的s梯度，以及缓存计算当前的cell的反向传播梯度.
        # 最后一层的ds_prevt为0 用自己本身的ds去传播即可 其它层次的则自身的ds加上前一次的ds
        gradients = rnn_cell_backward(ds[:, :, t] + ds_prevt, caches[t])
        # 获取梯度准备进行更新
        dx_t, ds_prevt, dUt, dWt, dbat = gradients["dx_t"], gradients["ds_prev"], gradients["dU"], gradients[
            "dW"], gradients["dba"]
        # 进行每次t时间上的梯度接过相加，作为最终更新的梯度  梯度累计求和
        dx[:, :, t] = dx_t
        dU += dUt
        dW += dWt
        dba += dbat

    # 最后ds0的输出梯度值
    ds0 = ds_prevt  #没有用也可以不存
    # 存储更新的梯度到字典当中

    gradients = {"dx": dx, "ds0": ds0, "dU": dU, "dW": dW, "dba": dba}

    return gradients

def test_all():
    # backward
    np.random.seed(1)

    # 定义了4个cell，每个词形状(3, 1)
    x = np.random.randn(3, 1, 4)
    s0 = np.random.randn(5, 1)

    W = np.random.randn(5, 5)
    U = np.random.randn(5, 3)
    V = np.random.randn(3, 5)
    ba = np.random.randn(5, 1)
    by = np.random.randn(3, 1)
    parameters = {"U": U, "W": W, "V": V, "ba": ba, "by": by}

    s, y, caches = rnn_forward(x, s0, parameters)
    # 随机给一每个4个cell的隐层输出的导数结果（真实需要计算损失的导数）
    # 假设知道了所有时刻相对于损失的的ds梯度值
    ds = np.random.randn(5, 1, 4)

    gradients = rnn_backward(ds, caches)

    print(gradients)

# test_all()
