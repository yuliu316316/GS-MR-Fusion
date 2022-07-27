# coding: utf-8
import cv2
import os
import glob
import numpy as np
import SimpleITK as sitk

def im2double(im):
    return im.astype(np.float) / 127.5 - 1


def rgb2ihs(im, eps=2.2204e-16):
    R = im[:, :, 0]
    G = im[:, :, 1]
    B = im[:, :, 2]

    I = (R + G + B) / 3.0
    v1 = (-np.sqrt(2) * R - np.sqrt(2) * G + 2 * np.sqrt(2) * B) / 6.0
    v2 = (R - G) / np.sqrt(2)
    H = np.arctan(v1 / (v2 + eps))
    S = np.sqrt(v1 ** 2 + v2 ** 2)
    # IHS = np.zeros(im.shape)
    # IHS[:,:,0] = I
    # IHS[:,:,1] = H
    # IHS[:,:,2] = S
    return I, v1, v2


def ihs2rgb(im, v1, v2):
    I = im[:, :, 0]
    R = I - v1 / np.sqrt(2) + v2 / np.sqrt(2)
    G = I - v1 / np.sqrt(2) - v2 / np.sqrt(2)
    B = I + np.sqrt(2) * v1
    RGB = np.zeros(im.shape)
    RGB[:, :, 0] = R
    RGB[:, :, 1] = G
    RGB[:, :, 2] = B
    return (RGB)


def prepare_data(data_path):
    """
    Args:
      data_path: choose train dataset or test dataset
      For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
    """
    data_path = os.path.join(os.getcwd(), data_path)
    images_path = glob.glob(os.path.join(data_path, "*.bmp"))
    images_path.extend(glob.glob(os.path.join(data_path, "*.tif")))
    images_path.sort(key=lambda x: int(x[len(data_path) + len(os.path.sep):-4]))
    return images_path


def get_images2(data_dir, image_size, label_size, stride):
    data = prepare_data(data_dir)
    sub_input_sequence = []
    sub_label_sequence = []
    padding = abs(image_size - label_size) // 2
    for i in range(len(data)):
        input_ = imread(data[i])
        input_ = (input_ - 127.5) / 127.5
        height, width = input_.shape[:2]
        for x in range(0, height - image_size + 1, stride):
            for y in range(0, width - image_size + 1, stride):
                sub_input = input_[x:x + image_size, y:y + image_size].reshape([image_size, image_size, 1])
                sub_label = input_[x + padding:x + padding + label_size, y + padding:y + padding + label_size].reshape(
                    [label_size, label_size, 1])
                sub_input_sequence.append(sub_input)
                sub_label_sequence.append(sub_label)
    sub_input_sequence = np.asarray(sub_input_sequence, dtype=np.float32)
    sub_label_sequence = np.asarray(sub_label_sequence, dtype=np.float32)
    return sub_input_sequence, sub_label_sequence


def imread(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img[:, :, 0]


# BraTS data
def read_img(img_path):
    return sitk.GetArrayFromImage(sitk.ReadImage(img_path))

def normalize_BraTS(x):
    ma = np.max(np.max(x))
    mi = np.min(np.min(x))
    x = 2*(x-mi)/(ma-mi) - 1
    return x

def z_score(data):
    mu = data.mean()
    std = data.std()
    return (data - mu)/(std + 1e-8)

def normalize2047(x):
    x = x/2047
    x[x > 1.0] = 1.0
    x = 2 * x - 1
    return x

def normalize_max(x):
    ma = np.max(np.max(x))
    mi = np.min(np.min(x))
    x = 2 * (x/(ma + 1e-8)) - 1
    return x

def get_BraTS_images2(image_size):
    flair = glob.glob('E:\datasets\BraTs2019\MICCAI_BraTS_2019_Data_Training/*/*/*flair.nii.gz')
    t1ce = glob.glob('E:\datasets\BraTs2019\MICCAI_BraTS_2019_Data_Training/*/*/*t1ce.nii.gz')
    label = glob.glob('E:\datasets\BraTs2019\MICCAI_BraTS_2019_Data_Training/*/*/*seg.nii.gz')
    # flair_path = flair.sort()
    # t1ce_path = t1ce.sort()
    # label_path = label.sort()

    sub_input1_sequence = []
    sub_input2_sequence = []
    sub_label1_sequence = []
    sub_label2_sequence = []

    for i in range(len(t1ce)):
        img1 = read_img(t1ce[i]).astype(np.float32)
        img2 = read_img(flair[i]).astype(np.float32)
        lab = read_img(label[i]).astype(np.float32)
        lab1 = lab.copy()
        # print(np.max(np.max(lab)))
        lab[lab == 2] = 0.
        lab[lab > 0] = 1.0
        lab1[lab1 > 0] = 1.0
        # print(np.max(np.max(lab)))
        #slice = 90
        for ii in range(40, 55):
            T1 = img1[ii*2]
            T2 = img2[ii*2]  # 240*240
            seg1 = lab[ii*2]
            seg2 = lab1[ii*2]
            # T1 = normalize_BraTS(T1)
            # T2 = normalize_BraTS(T2)
            if seg1.sum() > 200:
                # T1 = z_score(T1)
                # T2 = z_score(T2)
                T1 = normalize2047(T1)
                T2 = normalize2047(T2)
                T1 = T1.reshape([image_size, image_size, 1])
                T2 = T2.reshape([image_size, image_size, 1])
                seg1 = seg1.reshape([image_size, image_size, 1])
                seg2 = seg2.reshape([image_size, image_size, 1])
                sub_input1_sequence.append(T1)
                sub_input2_sequence.append(T2)
                sub_label1_sequence.append(seg1)
                sub_label2_sequence.append(seg2)
            #slice = slice + 2
    sub_input1_sequence = np.asarray(sub_input1_sequence, dtype=np.float32)
    sub_input2_sequence = np.asarray(sub_input2_sequence, dtype=np.float32)
    sub_label1_sequence = np.asarray(sub_label1_sequence, dtype=np.float32)
    sub_label2_sequence = np.asarray(sub_label1_sequence, dtype=np.float32)

    return sub_input1_sequence, sub_input2_sequence, sub_label1_sequence, sub_label2_sequence











