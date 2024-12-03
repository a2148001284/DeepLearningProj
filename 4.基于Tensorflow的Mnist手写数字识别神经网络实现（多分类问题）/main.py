import os
import tensorflow as tf
import matplotlib.pyplot as plt

# 加载输入部分
mnist = tf.keras.datasets.mnist
(train_data, train_label), (test_data, test_label) = mnist.load_data()
# print(train_data.shape)  #(60000, 28, 28)
# print(train_label.shape)  #(60000,)
train_data, test_data = train_data / 255.0, test_data / 255.0

model_path = "my_model.keras"

'''
构建模型部分
'''
if os.path.exists(model_path):
  model = tf.keras.models.load_model(model_path)
else:
  # 构建模型
  model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # 长度为 784（28×28）的一维向量 将二维矩阵28*28按照列或者行变成一维向量
    tf.keras.layers.Dense(128, activation='relu'),  # Dense层是全连接层 128个神经元
    tf.keras.layers.Dropout(0.2),  # Dropout层用于正则化，防止过拟合。 会随机将输入单元的 20% 设置为 0
    tf.keras.layers.Dense(10)  # 全连接层，有 10 个神经元
  ])

  # numpy()方法用于将 TensorFlow 的张量转换为 NumPy 数组
  # predictions = model(train_data[:1]).numpy()  #train_data[:1] 也就是一个 28×28 的单张图像数据 最终输出一个形状为(1, 10)的张量
  # print(predictions) #仅仅用于测试model的使用方法  可以用于生成预测结果

  # softmax函数会对每个样本的这些得分进行归一化，使得它们的总和为 1，并且每个元素的值都在 0 到 1 之间
  # tf.nn.softmax(predictions).numpy()
  # print(tf.nn.softmax(predictions).numpy()) #仅仅测试softmax函数

  # 它适用于当标签是整数形式（例如，手写数字识别中，真实标签是 0 - 9 的整数），而不是独热编码（One - Hot Encoding）形式的情况
  # from_logits=True表示模型的输出是未经归一化的 “对数几率”（logits）。在这种情况下，SparseCategoricalCrossentropy函数会在内部先应用softmax函数将模型输出转换为概率分布，然后再计算交叉熵损失。如果from_logits=False，则意味着模型输出已经是概率分布（经过softmax等操作），函数就直接计算交叉熵损失
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # 定义的损失函数
  # print(loss_fn(train_label[:1], predictions).numpy()) #损失值  仅仅测试

  model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

  # 训练并且评估模型
  model.fit(train_data, train_label, epochs=10)
  # 测试
  # 当verbose = 2时，表示以简洁的方式输出评估结果
  # 如果verbose = 0，则不会输出任何信息；如果verbose = 1，可能会输出更详细的评估过程信息，如每个批次（如果有批次划分）的评估情况等

  # 存储模型
  model.save(model_path)


'''
评估模型部分
'''
# 返回概率的模型
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
# print(probability_model(test_data[:5]))

# 评估模型 整体评估指标 不能看到细节
#model.evaluate(test_data,  test_label, verbose=1)

#具体预测的活动 看到细节的预测指标 但是并没有经过softmax处理 所以只是对数几率
predictions = model.predict(test_data)
#print(predictions) #二维矩阵张量
#获取每个样本预测得分最高的类别索引，即预测类别。axis = 1表示在列的方向上（对于每个样本）寻找最大值的索引

#二维张量有两个维度，axis = 0 代表行维度，axis = 1 代表列维度。 可以理解成处理方向吧
# 如果把列维度想象成一个个 “通道”，每个 “通道” 贯穿所有的行。
# 当 axis = 1 时，我们要在每个 “通道”（也就是每列贯穿的所有行）里的每一个元素集合（即每行）中找到最大值的索引。
predicted_classes = tf.argmax(predictions, axis = 1)
# for i in range(len(test_label)):
# for i in range(10):
#     print("真实值:", test_label[i], "预测值:", predicted_classes[i].numpy())

#可视化部分
#参考：https://blog.csdn.net/huhuhu1532/article/details/141951523
#https://tensorflow.google.cn/tutorials/quickstart/beginner?authuser=0&hl=zh-cn
plt.figure(figsize=(20,10))
for i in range(10):
  plt.subplot(2,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(test_data[i], cmap=plt.cm.binary)
  plt.xlabel(predicted_classes[i].numpy())
plt.show()



