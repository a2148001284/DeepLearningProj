import scipy.io as sio
import numpy as np

'''
加载数据部分
'''
def load_data():
    mat_file = sio.loadmat('./data/feature.mat')
    # print(mat_file)
    my_data = mat_file['feature']
    #print(my_data.shape)  #(301, 558)
    # print(my_data.shape)
    train_data_all = my_data[0:300:1, :]
    label_data_all = my_data[300:301:1, :]
    # print(train_data_all.shape)  # (300, 558)
    # print(label_data_all.shape)  # (1, 558)
    train_data = train_data_all[:,0:458:1]
    train_data_label = label_data_all[:,0:458:1]
    test_data = train_data_all[:,458::]
    test_data_label = label_data_all[:,458::]
    # print(train_data.shape)  #(300, 458)
    # print(train_data_label.shape) #(1, 458)
    # print(test_data.shape)  #(300, 100)
    # print(test_data_label.shape)  #(1, 100)
    train_data_label = train_data_label[0]
    train_data = train_data.T
    # print(train_data.shape)  #(458, 300)
    # print(train_data_label.shape)  #(458,)
    test_data_label = test_data_label[0]
    test_data = test_data.T
    return train_data,train_data_label,test_data,test_data_label

def load_data2():
    mat_file = sio.loadmat('./data/feature.mat')
    my_data = mat_file['feature']
    train_data_all = my_data[0:300:1, :]
    label_data_all = my_data[300:301:1, :]
    print(train_data_all.shape)  # (300, 558)
    print(label_data_all.shape)  # (1, 558)
    # 获取数据的总列数，即样本数量
    num_samples = train_data_all.shape[1]

    # 随机生成一个包含所有样本下标的数组，并打乱顺序
    all_indices = np.arange(num_samples)
    np.random.shuffle(all_indices)

    # 选取前458个随机下标的样本作为训练集
    train_indices = all_indices[:458]

    # 剩下的样本作为测试集
    test_indices = all_indices[458:]

    # 根据训练集下标选取训练数据，并调整形状为 (300, 458)
    train_data = train_data_all[:, train_indices]
    train_data = train_data[:, :458].reshape(300, 458)

    # 根据训练集下标选取训练标签，并调整形状为 (1, 458)
    train_data_label = label_data_all[:, train_indices]
    train_data_label = train_data_label[:, :458]

    # 根据测试集下标选取测试数据，并调整形状为 (300, 100)
    test_data = train_data_all[:, test_indices]
    test_data = test_data[:, :100].reshape(300, 100)

    # 根据测试集下标选取测试标签，并调整形状为 (1, 100)
    test_data_label = label_data_all[:, test_indices]
    test_data_label = test_data_label[:, :100]

    '''
    (300, 458)
    (1, 458)
    (300, 100)
    (1, 100)
    '''
    # print(train_data.shape)
    # print(train_data_label.shape)
    # print(test_data.shape)
    # print(test_data_label.shape)
    train_data = train_data.T
    test_data = test_data.T
    train_data = train_data * 100
    test_data = test_data * 100
    # print(train_data)
    return train_data, train_data_label[0], test_data, test_data_label[0]


# load_data()
# load_data2()

