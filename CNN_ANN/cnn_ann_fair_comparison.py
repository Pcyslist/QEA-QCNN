import tensorflow as tf
import numpy as np
import collections

# 可视化工具
import matplotlib.pyplot as plt
import dataset
# 持久化
import pickle
data_type = "CIFAR10"
x_train_bin,y_train_bin,x_test_bin,y_test_bin = dataset.generate_data(data_type)

def create_fair_cnn_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(4, [3, 3], activation='relu', input_shape=(4,4,1)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(8, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    return model

model = create_fair_cnn_model()
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
model.summary()

history = model.fit(x_train_bin,
          y_train_bin,
          batch_size=32,
          epochs=40,
          verbose=1,
          validation_data=(x_test_bin, y_test_bin))
plt.figure()
plt.plot(history.history['accuracy'], label='CNN Training')
plt.plot(history.history['val_accuracy'], label='CNN Validation')

fair_cnn_results = model.evaluate(x_test_bin, y_test_bin)

# ---------------------------ANN-------------------------
def create_fair_ann_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(4,4,1)))
    model.add(tf.keras.layers.Dense(8, activation='relu'))
    model.add(tf.keras.layers.Dense(4, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    return model


model = create_fair_ann_model()
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

model.summary()

history = model.fit(x_train_bin,
          y_train_bin,
          batch_size=32,
          epochs=40,
          verbose=1,
          validation_data=(x_test_bin, y_test_bin))

plt.plot(history.history['accuracy'], label='ANN Training')
plt.plot(history.history['val_accuracy'], label='ANN Validation')

plt.title(data_type + ' CNN/ANN Training')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig('CnnAnn_{}.svg'.format(data_type), format='svg')


fair_ann_results = model.evaluate(x_test_bin, y_test_bin)

with open("fair classical comparison_{}.txt".format(data_type),'w') as f:
    f.write("Accuracy:\n")
    f.write("CNN(185 params): " + str(fair_cnn_results[1]) + "\n")
    f.write("ANN(177 params): " + str(fair_ann_results[1]) + "\n")