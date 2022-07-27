# coding: utf-8
import imageio
import torch
import matplotlib.pyplot as plt
import os
import SimpleITK as sitk
import glob
import time

#from model import G
from model import GAFNet as G
import cv2
from metric import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def prepare_data2(dataset):
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
    data = glob.glob(os.path.join(data_dir, "*.tif"))
    data.extend(glob.glob(os.path.join(data_dir, "*.bmp")))
    data.sort(key=lambda x: int(x[len(data_dir) + 1:-4]))
    return data


def input_setup2(data_vi, data_ir, index):
    padding = 6
    sub_ir_sequence = []
    sub_vi_sequence = []
    _ir = imread(data_ir[index])
    _vi = imread(data_vi[index])
    # input_ir = (_ir - 127.5) / 127.5
    input_ir = _ir / 255
    input_ir = np.pad(input_ir, ((padding, padding), (padding, padding)), 'edge')
    w, h = input_ir.shape
    input_ir = input_ir.reshape([w, h, 1])
    # input_vi = (_vi - 127.5) / 127.5
    input_vi = _vi / 255
    input_vi = np.pad(input_vi, ((padding, padding), (padding, padding)), 'edge')
    w, h = input_vi.shape
    input_vi = input_vi.reshape([w, h, 1])
    sub_ir_sequence.append(input_ir)
    sub_vi_sequence.append(input_vi)
    train_data_ir = np.asarray(sub_ir_sequence)
    train_data_vi = np.asarray(sub_vi_sequence)
    return train_data_ir, train_data_vi


def imread(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    return img[:, :, 0]


def imsave(image, path):
    return imageio.imwrite(path, image)

def read_img(img_path):
    return sitk.GetArrayFromImage(sitk.ReadImage(img_path))

def normalize_BraTS(x):
    ma = np.max(np.max(x))
    mi = np.min(np.min(x))
    x = 2*(x-mi)/(ma-mi) - 1.0
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


def get_BraTS_images2(image_size, path):
    flair = glob.glob(os.path.join(path, '*', '*flair.nii.gz'))
    t1ce = glob.glob(os.path.join(path, '*', '*t1ce.nii.gz'))
    #label = glob.glob(os.path.join(path, '*', '*seg.nii.gz'))
    # flair_path = flair.sort()
    # t1ce_path = t1ce.sort()

    sub_input1_sequence = []
    sub_input2_sequence = []

    for i in range(len(t1ce)):
        img1 = read_img(t1ce[i]).astype(np.float32)
        img2 = read_img(flair[i]).astype(np.float32)

        slice = 100
        for ii in range(1):
            T1 = img1[slice]
            T2 = img2[slice]  # 240*240
            # T1 = normalize_BraTS(T1)
            # T2 = normalize_BraTS(T2)
            # T1 = z_score(T1)
            # T2 = z_score(T2)
            T1 = normalize2047(T1)
            T2 = normalize2047(T2)
            T1 = T1.reshape([image_size, image_size, 1])
            T2 = T2.reshape([image_size, image_size, 1])
            sub_input1_sequence.append(T1)
            sub_input2_sequence.append(T2)
            #slice = slice + 10
    sub_input1_sequence = np.asarray(sub_input1_sequence, dtype=np.float32)
    sub_input2_sequence = np.asarray(sub_input2_sequence, dtype=np.float32)

    return sub_input1_sequence, sub_input2_sequence


def test_all(g=None, path=os.path.join(os.getcwd(), 'output', 'result'), data='data'):
    data_ir = prepare_data2(os.path.join(data, 'Test_ir'))
    data_vi = prepare_data2(os.path.join(data, 'Test_vi'))

    if g is None:
        g = G().to(device)
        weights = torch.load('output/model4/9_generator.pth')
        g.load_state_dict(weights)

    if not os.path.exists(path):
        os.makedirs(path)

    g.eval()
    with torch.no_grad():
        for i in range(len(data_ir)):
            start = time.time()
            train_data_ir, train_data_vi = input_setup2(data_vi, data_ir, i)
            train_data_ir = train_data_ir.transpose([0, 3, 1, 2])
            train_data_vi = train_data_vi.transpose([0, 3, 1, 2])

            train_data_ir = torch.tensor(train_data_ir).float().to(device)
            train_data_vi = torch.tensor(train_data_vi).float().to(device)

            result = g(train_data_ir, train_data_vi)
            result = np.squeeze(result.cpu().numpy() * 127.5 + 127.5).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            result = clahe.apply(result)
            save_path = os.path.join(path, str(i + 1) + ".bmp")
            end = time.time()
            #
            imsave(result, save_path)
            # print("Testing [%d] success,Testing time is [%f]" % (i, end - start))
            pass

def test_BaTS_all(g=None, path=None):
    T1ce, Flair = get_BraTS_images2(240, 'Data_test')

    if g is None:
        g = G().to(device)
        weights = torch.load('model/final_generator.pth')
        g.load_state_dict(weights)

    # if not os.path.exists(path):
    #     os.makedirs(path)

    g.eval()
    T1ce = T1ce.transpose([0, 3, 1, 2])
    Flair = Flair.transpose([0, 3, 1, 2])
    with torch.no_grad():
        for i in range(len(T1ce)):
            start = time.time()
            img1 = T1ce[i].reshape(1, 1, 240, 240)
            img2 = Flair[i].reshape(1, 1, 240, 240)

            train_data_t1ce = torch.tensor(img1).float().to(device)
            train_data_flair = torch.tensor(img2).float().to(device)
            print(train_data_t1ce.shape)

            result = g(train_data_t1ce, train_data_flair)
            # result = np.squeeze((result.cpu().numpy() + 1)*127.5).astype(np.uint8)
            result = np.squeeze(result.cpu().numpy())


            plt.figure('T1')
            plt.imshow(np.squeeze(((T1ce[i]+1)*127.5).astype(np.uint8)), cmap='gray')
            plt.figure('T2')
            plt.imshow(np.squeeze(((Flair[i]+1)*127.5).astype(np.uint8)), cmap='gray')
            plt.figure('result')
            plt.imshow(((result + 1)*127.5).astype(np.uint8), cmap='gray')
            plt.show()

if __name__ == '__main__':
    #test_all()
    test_BaTS_all()