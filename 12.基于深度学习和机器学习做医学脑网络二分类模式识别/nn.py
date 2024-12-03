import os
import tensorflow as tf
import matplotlib.pyplot as plt
from load_dataset import *



train_data,train_data_label,test_data,test_data_label = load_data2()
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(300,)),
    # tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    # tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(
    loss=loss_fn,
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

batch_size = 64
history = model.fit(train_data,train_data_label,epochs=2000,batch_size=batch_size)
loss_history = history.history['loss']
accuracy_history = history.history['accuracy']
# 绘制损失变化图
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# 绘制准确率变化图
plt.subplot(1, 2, 2)
plt.plot(accuracy_history)
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.show()

test_loss, test_accuracy = model.evaluate(test_data, test_data_label)

print(f"测试集上的损失值: {test_loss}")
print(f"测试集上的准确率: {test_accuracy}")

