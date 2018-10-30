# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 15:56:49 2018

@author: YUN

normalize

StandardScaler
"""

import numpy as np
import csv
from sklearn.preprocessing import StandardScaler

data_input = np.genfromtxt('data_input.csv', delimiter=',')
data_output = np.genfromtxt('data_output.csv', delimiter=',')
print("data_input shape:" + str(data_input.shape))
print("data_output shape:" + str(data_output.shape))

# 標準化_INPUT X* = (X-X_mean)/std
scaler = StandardScaler()
print(scaler.fit(data_input)) #Compute the mean and std to be used for later scaling.

# Fit to data, then transform it.
data_input_normed = scaler.fit_transform(data_input)

# 標準化_OUTPUT X* = (X-X_mean)/std
scaler2 = StandardScaler()
print(scaler2.fit(data_output))  #Compute the mean and std to be used for later scaling.
data_output_normed = scaler2.fit_transform(data_output)

with open('StandardScaler.csv', 'w', newline='') as csvfile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)
    # 寫入資料
    writer.writerow(scaler.mean_) #樣本均值
    writer.writerow(scaler.var_) #樣本方差
    writer.writerow(scaler2.mean_)
    writer.writerow(scaler2.var_)

with open('data_input_normailzed.csv', 'w', newline='') as csvfile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)
    # 寫入資料
    for row in data_input_normed:
        writer.writerow(row)  # data_input_normailzed.csv

with open('data_output_normailzed.csv', 'w', newline='') as csvfile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)
    # 寫入資料
    for row in data_output_normed:
        writer.writerow(row)  # data_output_normailzed.csv

data_input_trans = scaler.inverse_transform(data_input_normed)
print("data_input_trans = ")
print(np.round(data_input_trans-data_input))

data_output_trans = scaler2.inverse_transform(data_output_normed)
print("data_output_trans = ")
print(np.round(data_output_trans-data_output))