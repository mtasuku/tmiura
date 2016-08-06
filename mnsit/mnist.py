# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 17:25:54 2016

@author: tmiura
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

####################################################################
# mnistをtensorflowで実装
# コードの分割化などを行っていないため若干見にくい
####################################################################

from __future__ import absolute_import,unicode_literals
import input_data
import tensorflow as tf

# mnistデータの読み込み
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# cross_entropyを実装
sess = tf.InteractiveSession()      # 対話型セッションを始める(長時間プログラムには向かない)
# 式に用いる変数設定
x = tf.placeholder("float", shape=[None,784])       # 入力
y_ = tf.placeholder("float", shape=[None,10])       # 誤差関数用変数 真のclass Distribution
W = tf.Variable(tf.zeros([784,10]))     # 重み
b = tf.Variable(tf.zeros([10]))         # バイアス
sess.run(tf.initialize_all_variables())     # 変数の初期化(変数使用の際必ず必要)
y = tf.nn.softmax(tf.matmul(x,W)+b)     # y=softmax(Wx+b)微分も勝手に行ってくれる
cross_entropy = -tf.reduce_sum(y_*tf.log(y))        # クロスエントロピーを定義

# 学習アルゴリズムと最小化問題
# ココでは最急降下法(勾配降下法)の最小化を解く。学習率0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 1000回学習 バッチサイズ50(この中身の処理は未だ勉強不足)
for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x:batch[0],y_:batch[1]})

# 結果の表示
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))# 尤もらしいClassすなわちargmax(y)が教師ラベルと等しいか
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))# 各サンプル毎に評価して平均を計算
print accuracy.eval(feed_dict={x: mnist.test.images,y_: mnist.test.labels})# xに画像y_にそのラベルを代入
# 出てくる結果はこの時点で精度91%前後

###############################################################
# 以下で深層畳み込みニューラルネットワークを構築する。
# 深層化することで精度99%を目指す。
###############################################################

# 重みとバイアスの初期化
# 勾配消失問題のために、小さいノイズをのせて重みを初期化する関数？どういうこと？
def weight_variable(shape):     # 重みの初期化
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):       # バイアスの初期化
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)


# convolutionとpoolingの定義
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


# 第1レイヤー 5x5パッチで32の特徴を計算
# [5,5,1,32]は、5,5でパッチサイズを、1で入力チャンネル数、32で出力チャンネル
W_conv1 = weight_variable([5,5,1,32])   # 変数定義
b_conv1 = bias_variable([32])           # 変数定義
x_image = tf.reshape(x,[-1,28,28,1])    # 画像を28*28のモノクロ画像に変換

h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)   # ReLU処理?

h_pool1 = max_pool_2x2(h_conv1)# プーリング層1の作成


# 第2レイヤー 5x5パッチで64の特徴量を計算
W_conv2 = weight_variable([5,5,32,64])  # 変数定義
b_conv2 = bias_variable([64])           # 変数定義

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)

h_pool2 = max_pool_2x2(h_conv2)# プーリング層2の作成


# 全結合層（フルコネクションレイヤー）への変換
W_fc1 = weight_variable([7*7*64,1024])  # どういう変換?
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)


# Dropoutを行う
keep_prob = tf.placeholder("float")

h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)


# 読み出しレイヤー
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)


# モデルの学習と評価
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)# Adam法を使用。

correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))  # cast?
sess.run(tf.initialize_all_variables())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob:1.0})
        print "step %d, training accuracy %g" % (i,train_accuracy)
    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})# 0.5に抑えている?

# 結果表示
print "test accuracy %g" % accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0})