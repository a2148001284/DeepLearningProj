import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, RepeatVector, Concatenate, Dot, Activation, LSTM, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from nmt_utils import load_dataset, preprocess_data, string_to_int, softmax


class Seq2seq(object):
    def __init__(self, Tx=30, Ty=10, n_x=32, n_y=64):
        self.model_param = {
            "Tx": Tx,  # 定义encoder序列最大长度  输入的长度
            "Ty": Ty,  # decoder序列最大长度  可以理解成decoder部分 LSTM的个数
            "n_x": n_x,  # encoder的隐层输出值大小
            "n_y": n_y  # decoder的隐层输出值大小/cell输出值大小 就可以理解成输出的大小
        }

    def load_data(self, m):
        dataset, x_vocab, y_vocab = load_dataset(m)
        # print(len(x_vocab))
        # print(len(y_vocab))
        # print(y_vocab)
        X, Y, X_onehot, Y_onehot = preprocess_data(dataset, x_vocab, y_vocab,self.model_param["Tx"],self.model_param["Ty"])

        # print(X_onehot[0].shape)
        # print(X_onehot[0][0])

        # print(X_onehot)
        # print(X)

        # print("整个数据集特征值的形状:", X_onehot.shape) #整个数据集特征值的形状: (10000, 30, 37) 10000个数据样本 one-hot编码里，向量有30个维度   37表示有这么多个可能的字符串
        # print("整个数据集目标值的形状:", Y_onehot.shape) #整个数据集目标值的形状: (10000, 10, 11)

        # 打印数据集
        # print("查看第一条数据集格式：特征值:%s, 目标值: %s" % (dataset[0][0], dataset[0][1]))
        # print(X[0], Y[0])
        # print("one_hot编码：", X_onehot[0], Y_onehot[0])

        # 添加特征词不重复个数以及目标词的不重复个数
        self.model_param["x_vocab"] = x_vocab  #37的对应列表
        self.model_param["y_vocab"] = y_vocab  #11对应列表
        self.model_param["x_vocab_size"] = len(x_vocab)  #37
        self.model_param["y_vocab_size"] = len(y_vocab)  #11

        return X_onehot, Y_onehot

    def get_encoder(self):
        self.encoder = Bidirectional(LSTM(self.model_param["n_x"], return_sequences=True, name="bidirectional_1"),merge_mode='concat')
        return None

    def get_decoder(self):
        self.decoder = LSTM(self.model_param["n_y"], return_state=True)
        return None

    def get_output_layer(self):  #也就是一个全连接层 接上softmax函数即可
        self.output_layer = Dense(self.model_param["y_vocab_size"], activation='softmax')
        return None

    def get_attention(self):
        repeator = RepeatVector(self.model_param["Tx"])
        concatenator = Concatenate(axis=-1)
        densor1 = Dense(10, activation="tanh", name="Dense1")
        densor2 = Dense(1, activation="relu", name='Dense2')
        activator = Activation('softmax', name='attention_weights')
        dotor = Dot(axes=1)

        self.attention = {
            "repeator": repeator,
            "concatenator": concatenator,
            "densor1": densor1,
            "densor2": densor2,
            "activator": activator,
            "dotor": dotor
        }
        return None

    def init_seq2seq(self):
        self.get_encoder()
        self.get_decoder()
        self.get_attention()
        self.get_output_layer()
        return None

    def compute_one_attention(self, a, s_prev):
        s_prev = self.attention["repeator"](s_prev)
        concat = self.attention["concatenator"]([a, s_prev])
        e = self.attention["densor1"](concat)
        en = self.attention["densor2"](e)
        alphas = self.attention["activator"](en)
        context = self.attention["dotor"]([alphas, a])
        return context

    def model(self):
        '''
        样本的长度 X为(30,37) 对于一个样本来说，长度为30 37用于指定one-hot编码 只有一个为1表示对应的内容 所以仅仅一个日期就占用了30x37的单元
        人为规定对于encoder最大的输入长度为30 统一处理以后 对于decoder输入 就变成了统一长度为10 这些都可以理解成LSTM的个数
        而n_x和n_y则可以理解成每个LSTM输出的一个维度的大小 可以随意指定 一般指定为2的倍数的大小
        '''
        X = Input(shape=(self.model_param["Tx"], self.model_param["x_vocab_size"]), name="X")
        s0 = Input(shape=(self.model_param["n_y"],), name="s0")
        c0 = Input(shape=(self.model_param["n_y"],), name="c0")
        s, c = s0, c0
        outputs = []

        a = self.encoder(X)

        for t in range(self.model_param["Ty"]):
            context = self.compute_one_attention(a, s)  #计算context仅仅需要a和s
            s, _, c = self.decoder(context, initial_state=[s, c])  #对于每一个LSTM的decoder 都需要输入s,c,context
            out = self.output_layer(s)  #s需要输出 从而可以得到输出的概率分布
            outputs.append(out)

        model = Model(inputs=[X, s0, c0], outputs=outputs)
        return model

    def train(self, X_onehot, Y_onehot):
        model = self.model()
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.005),
            metrics=['accuracy'] * self.model_param["Ty"]
        )

        s0 = np.zeros((X_onehot.shape[0], self.model_param["n_y"]))
        c0 = np.zeros((X_onehot.shape[0], self.model_param["n_y"]))
        outputs = list(Y_onehot.swapaxes(0, 1))

        model.fit([X_onehot, s0, c0], outputs, epochs=10, batch_size=100)

        model.save_weights("models/model_weights.weights.h5")
        return None

    def load_model(self):
        model = s2s.model()
        model.load_weights("models/model_weights.weights.h5")
        return model

    def my_test(self,example):
        model = self.load_model()
        example = example
        source = string_to_int(example, self.model_param["Tx"], self.model_param["x_vocab"])
        source = np.expand_dims(
            np.array([to_categorical(x, num_classes=self.model_param["x_vocab_size"]) for x in source]), axis=0)
        # print(source)
        s0 = np.zeros((1, self.model_param["n_y"]))
        c0 = np.zeros((1, self.model_param["n_y"]))
        # 进行预测
        prediction = model.predict([source, s0, c0])
        prediction = np.argmax(prediction, axis=-1)
        # print(prediction) #预测是准确的

        # 输出预测结果 原始代码有问题 手撕进行修改
        # output = ''.join([self.model_param["y_vocab"].get(int(i), '') for i in prediction.flatten()])
        prediction = prediction.flatten()  #二维展平变成一维
        # print(prediction)
        # for i in prediction:
        #     a = self.model_param["y_vocab"]
        #     print(a)
        reverse_y_vocab = {v: k for k, v in self.model_param["y_vocab"].items()} #我们把字典的键值对进行一个反转就好处理了
        # print(reverse_y_vocab)

        #测试成功
        # for i in prediction:
        #     a = reverse_y_vocab.get(i)
        #     print(a)

        #修改后的结果
        output = ''.join([reverse_y_vocab.get(i, '') for i in prediction.flatten()])


        print("source:", example)
        print("output:", output)

    def test(self):
        model = self.model()
        # model.load_weights("./models/model.h5")
        #
        # example = '1 March 2001'
        # source = string_to_int(example, self.model_param["Tx"], self.model_param["x_vocab"])
        # source = np.expand_dims(np.array([to_categorical(x, num_classes=self.model_param["x_vocab_size"]) for x in source]), axis=0)
        # # print(source.shape)
        #
        # s0 = np.zeros((1, self.model_param["n_y"]))
        # c0 = np.zeros((1, self.model_param["n_y"]))
        # prediction = model.predict([source, s0, c0])
        # # print(prediction)
        # prediction = np.argmax(prediction, axis=-1)
        # # print(prediction)
        # output = ''.join([self.model_param["y_vocab"].get(i, '') for i in prediction.flatten()])
        # print("source:", example)
        # print("output:", output)
        # return None


if __name__ == '__main__':
    s2s = Seq2seq()
    X_onehot, Y_onehot = s2s.load_data(10000)
    s2s.init_seq2seq()
    # s2s.train(X_onehot, Y_onehot)
    s2s.my_test('1 March 2001')  #我写的test部分
    # s2s.my_test('2003 Auguest 16')
    # s2s.test()  #源文件的test 存在问题  最好还是用基于tf2.x训练的模型进行预测
