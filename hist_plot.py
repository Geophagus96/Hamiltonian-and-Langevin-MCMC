# -*- coding: utf-8 -*-
"""
Created on Thu Oct 08 02:41:50 2020

@author: Yuze Zhou
"""

import numpy as np
import matplotlib.pyplot as plt

train_data = np.loadtxt("mnist_train.csv", delimiter=",")

#Transforming the mnist data from grey-scale value to binary value
fac = 0.99 / 255
train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
binary_train_imgs = (train_imgs>0.5)

train_labels = np.asfarray(train_data[:, :1])

#Categorizing the training dataset into different digits
train_imgs_0_binary = np.matrix(binary_train_imgs[np.array(train_labels==0)[:,0],:])
train_imgs_0 = np.matrix(train_data[np.array(train_labels==0)[:,0],1:])
train_imgs_1_binary = np.matrix(binary_train_imgs[np.array(train_labels==1)[:,0],:])
train_imgs_1 = np.matrix(train_data[np.array(train_labels==1)[:,0],1:])
train_imgs_2_binary = np.matrix(binary_train_imgs[np.array(train_labels==2)[:,0],:])
train_imgs_2 = np.matrix(train_data[np.array(train_labels==2)[:,0],1:])
train_imgs_3_binary = np.matrix(binary_train_imgs[np.array(train_labels==3)[:,0],:])
train_imgs_3 = np.matrix(train_data[np.array(train_labels==3)[:,0],1:])
train_imgs_4_binary = np.matrix(binary_train_imgs[np.array(train_labels==4)[:,0],:])
train_imgs_4 = np.matrix(train_data[np.array(train_labels==4)[:,0],1:])
train_imgs_5_binary = np.matrix(binary_train_imgs[np.array(train_labels==5)[:,0],:])
train_imgs_5 = np.matrix(train_data[np.array(train_labels==5)[:,0],1:])
train_imgs_6_binary = np.matrix(binary_train_imgs[np.array(train_labels==6)[:,0],:])
train_imgs_6 = np.matrix(train_data[np.array(train_labels==6)[:,0],1:])
train_imgs_7_binary = np.matrix(binary_train_imgs[np.array(train_labels==7)[:,0],:])
train_imgs_7 = np.matrix(train_data[np.array(train_labels==7)[:,0],1:])
train_imgs_8_binary = np.matrix(binary_train_imgs[np.array(train_labels==8)[:,0],:])
train_imgs_8 = np.matrix(train_data[np.array(train_labels==8)[:,0],1:])
train_imgs_9_binary = np.matrix(binary_train_imgs[np.array(train_labels==9)[:,0],:])
train_imgs_9 = np.matrix(train_data[np.array(train_labels==9)[:,0],1:])

#Plotting the marginal histogram for each digit based on grey-scale value
#Here we use digit 0 as an example
fig, axs = plt.subplots(28, 28)
for i in range(28):
    for j in range(28):
        axs[i,j].hist(train_imgs_0[:,(i*28+j)])
        axs[i,j].set_axis_off()

#Plotting the mean of marginal distribution for each digit
#Here we use digit 0 as an example
train_img_0_mean = np.mean(np.matrix(train_imgs_0),axis=0)
train_0_mean = np.array(train_img_0_mean).reshape((28,28))
plt.imshow(train_0_mean)

#Plotting the variance of marginal distribution for each digit
#Here we use digit 0 as an example
train_img_0_var = np.diag(np.cov(train_imgs_0.T))
train_0_var = np.array(train_img_0_var).reshape((28,28))
plt.imshow(train_0_var)
