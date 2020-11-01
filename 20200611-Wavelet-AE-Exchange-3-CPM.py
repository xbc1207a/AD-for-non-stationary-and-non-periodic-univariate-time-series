# -*- coding: utf-8 -*-
"""
Created on Mon May 25 10:03:33 2020

@author: xbc12
"""

import matplotlib.pyplot as plt
import numpy as np

"""
讀取資料
file_name是檔案名稱，
透過pandas來讀取csv檔的資料，
在將dataframe轉為numpy array的形式
"""
# Read data
import pandas as pd

file_name = "realAdExchange/exchange-3_cpm_results.csv"
path = "./NAB-master/data/{}".format(file_name)

df = pd.read_csv(path)

data_value = df["value"].values

# Draw picture
title_string = "{}".format(file_name)
max_value = max(data_value)
index_range = np.arange(0, len(data_value), 1)

plt.title("{}".format("Real Ad Exchange"))
plt.plot(data_value, color="blue", label="value")
plt.grid(True)
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()

"""
透過sklearn裡的StandardScaler對資料做正規化
"""
# Normalize the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

data_value = scaler.fit_transform(data_value.reshape(-1, 1))
data_value = data_value.reshape(-1)

"""
以資料最前面1000比做訓練資料
分別切出訓練資料和測試資料
"""
# Train and test data
train_size = 1000

train_data = data_value[: train_size]
test_data = data_value[train_size: ]

"""
對訓練資料做切割，
slide window的大小設為60，
若是切出的片段大小小於60就不加進list裡
"""
# Slide window data
slide_window_size = 60
step = 1

slide_window_data_list = []
for i in range(0, train_data.shape[0], step):
    window = train_data[i: i+slide_window_size]
    if window.shape[0] == slide_window_size:
        slide_window_data_list.append(window)

"""
對每個移動視窗做離散小波轉換
"""
# Wavelet
import pywt

wave = "haar"
levels = 4

dwt_data_list = []
for slide_window in slide_window_data_list:
    # coifs = pywt.wavedec(data=slide_window, wavelet=wave, level=levels)
    cA, cD = pywt.dwt(slide_window, wave)
    
    # dwt_data_list.append(np.hstack(coifs))
    dwt_data_list.append(np.hstack([cA, cD]))

"""
將list資料轉為numpy array的格式
"""
# Source data
train_x = np.array(dwt_data_list)

# Activation parameter
alpha = 0.2

"""
autoencoder架構
"""
# DNN AE
import tensorflow as tf

kr = tf.keras.regularizers.l1_l2(l1=0.15, l2=0.25)

i = tf.keras.Input(shape=(train_x.shape[1],))
x = tf.keras.layers.Dense(units=32, 
                          activation="selu")(i)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(units=16, 
                          activation="selu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(units=8, 
                          activation="selu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(units=4,
                          activation="selu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(units=2,
                          activation="selu", 
                          name="code")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(units=4,
                          activation="tanh")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(units=8, 
                          activation="tanh")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(units=16,
                          activation="tanh")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(units=32,
                          activation="tanh")(x)
o = tf.keras.layers.Dense(units=train_x.shape[1],
                          activation="tanh")(x)



model = tf.keras.models.Model(inputs=i,
                              outputs=o)

model.summary()

model.compile(optimizer="adam",
              loss="mse",
              metrics=["mape", "mae"])

early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                              patience=25)

history = model.fit(x=train_x,
                    y=train_x,
                    batch_size=16,
                    epochs=1000,
                    callbacks=[early_stop],
                    validation_split=0.02,
                    verbose=2)

# Draw Training History
plt.title("Training history")
plt.plot(history.history["loss"], label="loss", color="blue")
plt.plot(history.history["val_loss"], label="val_loss", color="red")
plt.grid(True)
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.show()

# Distance function
def get_distance(x_array, y_array):
    return np.linalg.norm(x_array-y_array)


"""
取訓練資料最後59比加上1比測試資料來作為輸入序列，
對該序列做離散小波轉換，
之後輸入到訓練好的autoencoder裡得到loss值，
之後將loss值存入list裡
"""
# Test the rest data
sequence = train_data[train_data.shape[0]-slide_window_size+1: ]
score_list = []
distance_list = []

for i in range(0, test_data.shape[0], step):
    print("i = {}".format(i))
    sequence = np.hstack([sequence, test_data[i]])
    
    # coifs = pywt.wavedec(data=sequence, wavelet=wave, level=levels)
    cA, cD = pywt.dwt(sequence, wave)
    # test_x = np.hstack(coifs)
    test_x = np.hstack([cA, cD])
    test_x = test_x.reshape(-1, test_x.shape[0])
    
    loss = model.evaluate(x=test_x,
                          y=test_x)
    test_y = model.predict(test_x)
    
    distance = get_distance(test_x.reshape(-1), test_y.reshape(-1))
    
    print("loss = {} distance = {}".format(loss, distance))
    
    score_list.append(loss[0])
    distance_list.append(distance)
    
    sequence = sequence[1: ]

figure_size = (20, 15)
max_score2 = max(test_data)
min_score2 = min(test_data)
x = np.arange(0, test_data.shape[0], 1)

plt.title("test data")
plt.plot(test_data)
plt.grid(True)
plt.fill_between(x, min_score2, max_score2, where=(x > 1045-train_size) & (x < 1197-train_size), facecolor='pink')
plt.xlabel("Index")
plt.ylabel("Value")
plt.show()

max_score2 = max(score_list)
min_score2 = min(score_list)
x = np.arange(0, test_data.shape[0], 1)

#plt.figure(figsize=figure_size)
plt.title("Reconstruction error")
plt.plot(score_list)
plt.grid(True)
plt.fill_between(x, min_score2, max_score2, where=(x > 1045-train_size) & (x < 1197-train_size), facecolor='pink')
plt.xlabel("Index")
plt.ylabel("Value")
#plt.xticks(np.arange(0, test_data.shape[0], 1000))
plt.yticks(np.arange(0, max_score2, 0.5))
plt.show()

"""
設定閥值和連續上升次數，
一旦重建誤差值超過閥值就開始計算連續上升次數，
當連續上升次數超過設定的閥值就將這個時間點是為異常並記錄異常點的index和分數
"""
threshold = 2.5
previous_loss = 0.0
score_up_times = 0
score_up_threshold = 1

temp_score_list = []
point_list = []
detect_score_list = []
for score_index in range(len(score_list)):
    if score_list[score_index] > threshold:
        if score_list[score_index] > previous_loss:
            previous_loss = score_list[score_index]
            
            score_up_times = score_up_times+1
        else:
            score_up_times = 0
        
        if score_up_times >= score_up_threshold:
            point_list.append(score_index)
            detect_score_list.append(score_list[score_index])
    else:
        score_up_times = 0
        previous_loss = 0.0

plt.title("Reconstruction error")
plt.plot(score_list, "+-")
plt.plot(point_list, detect_score_list, "ro")
plt.grid(True)
plt.fill_between(x, min_score2, max_score2, where=(x > 1045-train_size) & (x < 1197-train_size), facecolor='pink')
plt.xlabel("Index")
plt.ylabel("Value")
plt.xticks(np.arange(0, test_data.shape[0], 1000))
#plt.yticks(np.arange(0, max_score2, 0.5))
plt.show()

"""
畫出測試資料和重建誤差的圖，
fill_between用來劃出異常視窗的位置，
tight_layout會自動調整subplot之間的空白和文字位置
"""
max_value = max(test_data)
min_value = min(test_data)
max_score = max(score_list)
min_score = min(score_list)
x = np.arange(0, test_data.shape[0], 1)

test_value_list = []
for point in point_list:
    test_value_list.append(test_data[point])

plt.subplot(211)
plt.plot(test_data, label="NAB Ad Exchange-3 cpm", color="blue")
plt.plot(point_list, test_value_list, "ro")
plt.grid(True)
plt.fill_between(x, min_value, max_value, where=(x > 1045-train_size) & (x < 1197-train_size), facecolor='pink')
plt.legend()

plt.subplot(212)
plt.plot(score_list, label="Reconstruction Error", color="blue")
plt.plot(point_list, detect_score_list, "ro")
plt.grid(True)
plt.fill_between(x, min_score, max_score, where=(x > 1045-train_size) & (x < 1197-train_size), facecolor='pink')
plt.legend()

plt.tight_layout()
plt.show()

print("Done")
