# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 15:56:49 2018

@author: YUN

normalize

123
"""

import numpy as np
import csv
import random
import os
import time


data_input = np.genfromtxt('data_input.csv', delimiter=',')
file1 = open('data_input_normailzed.csv', 'w', newline='')
csvCursor = csv.writer(file1)

data_output = np.genfromtxt('data_output.csv', delimiter=',')
file2 = open('data_output_normailzed.csv', 'w', newline='')
csvCursor2 = csv.writer(file2)

# 標準化_INPUT
diff = np.max(data_input, axis=0) - np.min(data_input, axis=0)
data_input_min = np.min(data_input, axis=0)
data_input_normed = ((data_input - data_input_min) / diff)

for row in data_input_normed:
    csvCursor.writerow(row)  # data_input_normailzed.csv

print("data_input shape:" + str(data_input.shape))


# 標準化_OUTPUT
diff2 = np.max(data_output, axis=0) - np.min(data_output, axis=0)
data_output_min = np.min(data_output, axis=0)
data_output_normed = ((data_output - data_output_min) / diff2)

for row in data_output_normed:
    csvCursor2.writerow(row)  # data_output_normailzed.csv

print("data_output shape:" + str(data_output.shape))

file1.close()
file2.close()

