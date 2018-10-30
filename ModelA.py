# -*- coding: utf-8 -*-
"""
Created on 2018.09.14

@author: me

ImplamtROI	BeamI	DoseCup	MutiStepSigma	BeamAngleMean	BeamAngleSpread

"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error


INPUT_SIZE = 45
HIDDEN_SIZE1 = 30
HIDDEN_SIZE2 = 15
#HIDDEN_SIZE3 = 20
OUTPUT_SIZE = 5
loss_limit = 0.004
learning_rate = 0.01
beta = 0.0001
epoch = 25000
k_number = 10


for loop in range(1):

    # region讀取數據
    f = open('data/data_input_normailzed.csv', 'r')
    X = []
    for row in csv.reader(f):
        X.append(row)
    f.close()

    f = open('data/data_output_normailzed.csv', 'r')
    y = []
    for row in csv.reader(f):
        y.append(row)
    f.close()

    # data轉換
    X = np.array(X)
    y = np.array(y)
    X = X.astype(np.float)
    y = y.astype(np.float)

    # 隨機取一定比例test
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # print(X_train.shape)
    # print(X_test.shape)
    #
    # f = open('data/X_test.csv', "w", newline='')
    # w = csv.writer(f)
    # w.writerows(X_test)
    # f.close()
    #
    # f = open('data/y_test.csv', "w", newline='')
    # w = csv.writer(f)
    # w.writerows(y_test)
    # f.close()

    test_loss_log = []
    train_loss_log = []
    test_group = 1
    k_fold = KFold(n_splits=k_number, shuffle=True, random_state=1)
    # endregion

    # 分N組資料交叉驗證
    #for train_index, validation_index in k_fold.split(X_train, y_train):
    for train_index, test_index in k_fold.split(X, y):

        time_load = time.strftime("%Y-%m-%d_%H-%M-%S")
        save_path = "result/" + time_load + '/'

        if not os.path.exists(save_path):  # 先確認資料夾是否存在
            os.makedirs(save_path)

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print(type(test_index))

        #X_train, X_validation = X[train_index], X[validation_index]
        #y_train, y_validation = y[train_index], y[validation_index]

        # 存入train,test index
        fp = open(save_path + "index.txt", "w")  # 開啟檔案
        # 寫入到檔案
        fp.write("X_train.shape = " + str(X_train.shape) + "\n")
        #fp.write("X_validation.shape = " + str(X_validation.shape) + "\n")
        fp.write("X_test.shape = " + str(X_test.shape) + "\n")
        fp.write("y_train.shape = " + str(y_train.shape) + "\n")
        #fp.write("y_validation.shape = " + str(y_validation.shape) + "\n")
        fp.write("y_test.shape = " + str(y_test.shape) + "\n")

        fp.write("train_index = " + str(train_index) + "\n")
        fp.write("test_index = " + str(test_index) + "\n")
        #fp.write("validation_index = " + str(validation_index) + "\n")
        fp.close()  # 關閉檔案

        np.savetxt(save_path + "index.csv", test_index, delimiter=",")

        # region 建立 Feeds
        with tf.name_scope('Inputs'):
            xs = tf.placeholder(tf.float32, shape = [None, INPUT_SIZE], name = 'x_inputs')
            ys = tf.placeholder(tf.float32, shape = [None, OUTPUT_SIZE], name = 'y_inputs')
        # endregion

        # region組裝神經網路

        # 添加 1 個隱藏層
        with tf.name_scope("layer1"):
            W1 = tf.Variable(tf.random_normal([INPUT_SIZE, HIDDEN_SIZE1]), name='layer1/W')
            b1 = tf.Variable(tf.zeros([1, HIDDEN_SIZE1]), name='layer1/b')
            hidden_layer1 = tf.nn.softplus(tf.add(tf.matmul(xs, W1), b1))
        # 添加 1 個隱藏層
        with tf.name_scope("layer2"):
            W2 = tf.Variable(tf.random_normal([HIDDEN_SIZE1, HIDDEN_SIZE2]), name='layer2/W')
            b2 = tf.Variable(tf.zeros([1, HIDDEN_SIZE2]), name='layer2/b')
            hidden_layer2 = tf.nn.relu(tf.add(tf.matmul(hidden_layer1, W2), b2))
        # # 添加 1 個隱藏層
        # with tf.name_scope("layer3"):
        #     W3 = tf.Variable(tf.random_normal([HIDDEN_SIZE2, HIDDEN_SIZE3]), name='layer3/W')
        #     b3 = tf.Variable(tf.zeros([1, HIDDEN_SIZE3]), name='layer3/b')
        #     hidden_layer3 = tf.nn.relu(tf.add(tf.matmul(hidden_layer2, W3), b3))
        # 添加 1 個輸出層
        with tf.name_scope("layer4"):
            W4 = tf.Variable(tf.random_normal([HIDDEN_SIZE2, OUTPUT_SIZE]), name='layer4/W')
            b4 = tf.Variable(tf.zeros([1, OUTPUT_SIZE]), name='layer4/b')
            output_layer = tf.add(tf.matmul(hidden_layer2, W4), b4)


        # 定義 `loss` 與要使用的 Optimizer
        # 定義loss function 並且選擇減低loss 的函數 這裡選擇GradientDescentOptimizer
        # 其他方法再這裡可以找到 https://www.tensorflow.org/versions/r0.10/api_docs/python/train.html
        with tf.name_scope('Loss'):
            # L2 loss
            l2_loss = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)  + tf.nn.l2_loss(W4)
            loss = tf.reduce_mean(tf.pow(ys - output_layer, 2)) / 2
            loss = loss + l2_loss * beta

        with tf.name_scope('Train'):
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
            #train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        # endregion組裝神經網路




        # 紀錄誤差
        loss_list = []
        Validation_loss_list = []
        epoch_list = []
        loss_min = 100000
        over_times = 0

        # 全部設定好了之後，初始化喔
        init = tf.global_variables_initializer()

        # 保存
        saver = tf.train.Saver({'Weights_l1': W1, 'basis_l1': b1, 'Weights_l2': W2, 'basis_l2': b2,\
                                 'Weights_l4': W4, 'basis_l4': b4})

        with tf.Session() as sess:
            # 開始運算
            sess.run(init)

            tStart = time.time()  # 計時開始
            for i in range(100000):
                # 整個訓練最核心的code , feed_dict 表示餵入 輸入與輸出
                sess.run(train_step, feed_dict={xs: X_train, ys: y_train})
                if i % 50 == 0:     #每50次顯示誤差一遍
                    # 要取出預測的數值 必須再run 一次才能取出
                    print("Weights: {0: .4f} |loop: {1},test group: {2}" .format(sess.run(l2_loss), loop, test_group))
                    loss_value = sess.run(loss, feed_dict={xs: X_train, ys: y_train})
                    #Validation_loss_value = sess.run(loss, feed_dict={xs: X_validation, ys: y_validation})
                    Validation_loss_value = sess.run(loss, feed_dict={xs: X_test, ys: y_test})

                    print('Step {0}, Train MSE: {1: .4f} | CV MSE: {2: .4f}'.format(i, loss_value, Validation_loss_value))

                    if loss_value < 0.1:
                    #if i > 10000:
                        epoch_list.append(i)
                        loss_list.append(loss_value)   # 紀錄誤差
                        Validation_loss_list.append(Validation_loss_value)  # 紀錄誤差

                    if loss_value < loss_min:
                        loss_min = loss_value
                        over_times = 0

                    if loss_value > loss_min:
                        over_times += 1
                        #print('over_times : {0}' .format(over_times))

                 # 誤差
                if loss_value < loss_limit:  # loss
                #if i > epoch:
                #if over_times > 4:
                    print('%d loss: %f' % (i, loss_value))
                    print(over_times)
                    break
                if i > epoch:
                    print('%d loss: %f' % (i, loss_value))
                    break

            tEnd = time.time()  # 計時結束
            print('it costs %d mins %f seconds.' % ((tEnd - tStart) // 60, (tEnd - tStart) % 60))  # 花費之時間顯示

            saver_path = saver.save(sess, save_path+"UMC_Model_A/model.ckpt")
            print("Saver to path: ", saver_path)


            # Weights_l1=sess.run(W1)
            #         # #print("Weights_l1:", Weights_l1)
            #         # f = open(save_path+"Weights_l1.csv", "w", newline='')
            #         # w = csv.writer(f)
            #         # w.writerows(Weights_l1)
            #         # f.close()
            #         #
            #         # Weights_l2 = sess.run(W2)
            #         # #print("Weights_l2:", sess.run(W2))
            #         # f = open(save_path+"Weights_l2.csv", "w", newline='')
            #         # w = csv.writer(f)
            #         # w.writerows(Weights_l2)
            #         # f.close()


            # testing
            y_pred = sess.run(output_layer, feed_dict={xs: X_test})
            f = open(save_path+"y_pred.csv", "w", newline='')
            w = csv.writer(f)
            w.writerows(y_pred)
            f.close()
            test_MSE = mean_squared_error(y_test, y_pred)

            # Validation
            y_pred_v = sess.run(output_layer, feed_dict={xs: X_test})
            validation_MSE = mean_squared_error(y_test, y_pred_v)

        # 為了可以可視化我們訓練的結果
        fig = plt.figure('loss',figsize=(18, 5))
        plt.subplot(2, 5, test_group)
        plt.plot(epoch_list, loss_list, color='blue', linestyle='--', label='train loss')
        plt.plot(epoch_list, Validation_loss_list, color='red', label='test loss')
        plt.title("No."+str(test_group))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(loc='upper right')

        ###############################################################################

        # region 轉換資料
        convert_path = save_path + 'convert/'

        if not os.path.exists(convert_path):    # 先確認資料夾是否存在
            os.makedirs(convert_path)

        # 原始全輸出資料(未normailzed)
        data = np.genfromtxt('data/data_output.csv', delimiter=',')

        # 測試結果


        diff = np.max(data, axis=0) - np.min(data, axis=0)   # 計算normalize
        data_min = np.min(data, axis=0)
        print('diff:', diff)
        print('data_min:', data_min)

        y_pred_convert = y_pred*diff + data_min       # convert*測試資料
        y_test_convert = y_test*diff + data_min     # convert*真實資料
        loss_convert = y_test_convert-y_pred_convert                   # convert*(真實資料-測試資料)

        f = open(convert_path + "y_pred_convert.csv", "w", newline='')
        w = csv.writer(f)
        w.writerows(y_pred_convert)
        f.close()

        f = open(convert_path + "y_test_convert.csv", "w", newline='')
        w = csv.writer(f)
        w.writerows(y_test_convert)
        f.close()

        f = open(convert_path + "loss_convert.csv", "w", newline='')
        w = csv.writer(f)
        w.writerows(loss_convert)
        f.close()

        y_loss = y_test-y_pred                   # 真實資料-測試資料

        y_loss_AVG = np.average(np.abs(y_loss), axis=0)   #各項誤差絕對值平均

        print("[test_MSE]=" + str(test_MSE))
        print("[validation_MSE]=" + str(validation_MSE))
        print("[Multi Sigma] loss average="+str(y_loss_AVG[2]))
        print("[ROI] loss average="+str(y_loss_AVG[0]))
        print("[Dose Cup] loss average="+str(y_loss_AVG[1]))
        print("[Beam Angle Mean] loss average="+str(y_loss_AVG[3]))
        print("[Beam Angle Spread] loss average="+str(y_loss_AVG[4]))
        print("[TOTAL] loss average="+str(np.mean(y_loss_AVG)))


        # 檢查檔案是否存在
        if os.path.isfile('統計.csv'):
          print("檔案存在。")
          file3 = open('統計.csv', 'a', newline='')
          csvCursor3= csv.writer(file3)
        else:
          print("檔案不存在。")
          file3 = open('統計.csv', 'a', newline='')
          csvCursor3= csv.writer(file3)
          information=['時間', '終止條件', '學習率', 'test_group','HIDDEN_SIZE1', 'HIDDEN_SIZE2',\
                        'Multi loss AVG', 'ROI_loss AVG', 'test_MSE', 'validation_MSE']
          csvCursor3.writerow(information)

        information = [time_load, str(loss_limit), str(learning_rate), str(test_group), HIDDEN_SIZE1, HIDDEN_SIZE2,\
                       str(y_loss_AVG[2]), str(y_loss_AVG[0]), test_MSE, validation_MSE]

        csvCursor3.writerow(information)
        test_group += 1

        test_loss_log.append(y_loss_AVG[2])
        train_loss_log.append(loss_value)


    final_test_loss = np.array(test_loss_log).mean()
    final_train_loss = np.average(train_loss_log).mean()

    print('K folds finished')
    print('Final validation score, MSE: {0: .4} | R_squared: {1: .4}'.format(final_test_loss, final_train_loss))

    plt.tight_layout() # 避免兩個圖重疊
    plt.savefig(convert_path+'recall.png', dpi=400)
    plt.close('all')
    #plt.show()



        # ######
        # x = np.linspace(1, OUTPUT_SIZE, OUTPUT_SIZE)
        #
        # # Recall
        # plt.figure('recall',figsize=(16, 5))
        # i = 0
        # for index in range(len(data_final2)-len(recall),len(data_final2)):
        #     plt.subplot(1,2,i+1)
        #     plt.plot(x, data_result2[index], 'ro', label='real')
        #     plt.plot(x, data_result[index], 'bx', label='testing')
        #     plt.title("recall-No."+str(recall[i]))
        #     plt.ylabel('output')
        #     #plt.xticks( np.arange(1,13), [ 'ROI', 'MutiSigma'], rotation=90 )
        #     i += 1
        # plt.tight_layout() # 避免兩個圖重疊
        # plt.savefig(convert_path+'recall.png', dpi=400)
        # plt.show()
        # plt.close('all')
        # endregion