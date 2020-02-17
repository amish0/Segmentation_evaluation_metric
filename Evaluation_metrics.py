# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 17:43:23 2020

@author: AMISH
"""
import tensorflow as tf
from tensorflow.keras import losses
import numpy as np
import cv2
import matplotlib.pyplot as plt
def recall(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    y_true_f = tf.round(y_true_f)
    y_pred_f = tf.round(y_pred_f)
    intersection = tf.reduce_sum(tf.multiply(y_true_f, y_pred_f))
    re=intersection/(tf.reduce_sum(y_true_f)+1)
    return re

def precision(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    y_true_f = tf.round(y_true_f)
    y_pred_f = tf.round(y_pred_f)
    intersection = tf.reduce_sum(tf.multiply(y_true_f, y_pred_f))
    pr=intersection/(tf.reduce_sum(y_pred_f)+1)
    return pr

def specificity(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    y_true_f = tf.round(y_true_f)
    y_pred_f = tf.round(y_pred_f)
    intersection = tf.reduce_sum(tf.multiply((1-y_true_f), (1-y_pred_f)))
    sp=intersection/(tf.reduce_sum((1-y_pred_f))+1)
    return sp

def mean_iou(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    y_true_f = tf.round(y_true_f)
    y_pred_f = tf.round(y_pred_f)
    intersection = tf.reduce_sum(tf.multiply(y_true_f, y_pred_f))
    miou=intersection/(tf.reduce_sum(y_pred_f)+tf.reduce_sum(y_true_f)-intersection+1)
    return miou

def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(tf.multiply(y_true_f,y_pred_f))
    score = (2. * intersection) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_coeff1(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    y_true_f = tf.round(y_true_f)
    y_pred_f = tf.round(y_pred_f)
    intersection = tf.reduce_sum(tf.multiply(y_true_f,y_pred_f))
    score = (2. * intersection) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score 

def dice_loss(y_true,y_pred):
    loss = 1-dice_coeff(y_true, y_pred)
    return loss
    
def bce_dice_loss(y_true,y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss
#### al the below code is writeen in matplab
def Hausdorff_Matrix(A,B,data=1):
    kernel = np.ones((3,3))
    Ae = cv2.erode(A,kernel,iterations = 1)
    A=A-Ae
#    print(np.min(A))
    Be = cv2.erode(B,kernel,iterations = 1)
    B=B-Be
#    plt.imshow(B)
#    print(np.min(B))
    A_corr=np.argwhere(A==data)
    B_corr=np.argwhere(B==data)
#    print(A_corr)
#    print(B_corr)
    yc=np.diagonal(np.dot(A_corr,A_corr.T))[np.newaxis].T
    xc=np.diagonal(np.dot(B_corr,B_corr.T))[np.newaxis].T
#    print(xc.shape)
#    print(yc.shape)
    Z=np.dot(A_corr,B_corr.T)
#    print(Z.shape)
    HD_matrix=yc-2*Z
    HD_matrix=np.sqrt(HD_matrix+xc.T)
#    print(HD_matrix.shape)
    return HD_matrix

def HausdorffDist(A,B,data=1):
    HDM=Hausdorff_Matrix(A,B,data)
    HD=np.max(np.array([np.max(np.min(HDM,axis=0)),np.max(np.min(HDM,axis=1))]))
    return HD

def ModifiedHausdorffDist(A,B,data=1):
    HDM=Hausdorff_Matrix(A,B,data)
    MHD=np.max(np.array([np.mean(np.min(HDM,axis=0)),np.mean(np.min(HDM,axis=1))]))
    return MHD

def ASSD(A,B,data=1):
    HDM=Hausdorff_Matrix(A,B,data)
    ASD=np.sum(np.array([np.sum(np.min(HDM,axis=0)),np.sum(np.min(HDM,axis=1))]))/(HDM.shape[0]+HDM.shape[1])
    return ASD

def dice_coeff_numpy(y_true, y_pred):
    smooth = 1.
    y_true_f = np.reshape(y_true, [-1])
    y_pred_f = np.reshape(y_pred, [-1])
    intersection = np.sum(y_true_f*y_pred_f)
    score = (2. * intersection) / (np.sum(y_true_f) + np.um(y_pred_f) + smooth)
    return score

def dice_coeff1_numpy(y_true, y_pred):
    smooth = 1.
    y_true_f = np.reshape(y_true, [-1])
    y_pred_f = np.reshape(y_pred, [-1])
    y_true_f = np.round(y_true_f)
    y_pred_f = np.round(y_pred_f)
    intersection = np.sum(y_true_f*y_pred_f)
    score = (2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    return score