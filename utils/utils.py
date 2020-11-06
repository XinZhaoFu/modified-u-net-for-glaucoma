import numpy as np
import h5py
import cv2
import datetime
import os
from random import randint, shuffle
from skimage.measure import regionprops, label
import shutil
from utils.config import config_init

config_init = config_init()


def shuffle_file(img_file_list, label_file_list):
    """
    打乱img和label的文件列表顺序 并返回两列表
    :param img_file_list:
    :param label_file_list:
    :return:
    """
    np.random.seed(10)
    index = [i for i in range(len(img_file_list))]
    np.random.shuffle(index)
    img_file_list = np.array(img_file_list)[index]
    label_file_list = np.array(label_file_list)[index]
    return img_file_list, label_file_list


def load_hdf5(infile):
    """
    载入hdf5文件
    :param infile:
    :return:
    """
    with h5py.File(infile, "r") as f:
        return f["image"][()]


def write_hdf5(data, outfile):
    """
    写入hdf5文件
    :param data:
    :param outfile:
    :return:
    """
    with h5py.File(outfile, "w") as f:
        f.create_dataset("image", data=data, dtype=data.dtype)


def fit_ellipse(prob_img):
    """
    将传入灰度图的唯一连通域进行椭圆拟合并输出
    :param prob_img:
    :return:
    """
    _, labels_num = label(prob_img, connectivity=2, return_num=True)
    if labels_num != 1:
        print("[error] can`t fit ellipse: labels_num != 1----")
        return prob_img
    contours, _ = cv2.findContours(prob_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # print(contours)
    if len(contours[0]) < 5:
        print("[error]contours nums <5------")
        return prob_img
    ellipse = cv2.fitEllipse(contours[0])
    prob_img = np.zeros([512, 512], np.uint8)
    cv2.ellipse(prob_img, ellipse, 255, -1)
    return prob_img


def print_cost_time(start_time):
    """
    计算花费时长
    :param start_time:
    :return:
    """
    end_time = datetime.datetime.now()
    print('time:\t' + str(end_time - start_time).split('.')[0])


def binary_predict_value_to_img(predict_value, img_width=512):
    """
    将独热码存储的图像数据转为常规图像值 默认该图为全0 独热码的第二个通道是该像素为1的概率 大于0.5则设定为1
    :param predict_value:
    :param img_width:
    :return:
    """
    predict_img_temp = np.zeros((512, 512, 1))
    for i in range(img_width):
        for j in range(img_width):
            if predict_value[i, j, 1] > 0.5:
                predict_img_temp[i, j, 0] = 1
    binary_predict_img = predict_img_temp * 255.
    return binary_predict_img


def create_dir(folder_name):
    """
    创建文件夹
    :param folder_name: 文件夹列表
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def recreate_dir(folder_name):
    """
    重建文件夹
    :param folder_name:
    :return:
    """
    shutil.rmtree(folder_name)
    create_dir(folder_name)


def get_offset_region_img_list(img, label, offset=64):
    """
    传入1张图像及1张标签 输出1张裁剪原图及4张偏移裁剪图和1张裁剪标签
    4张偏移图横纵坐标偏移值为offset 偏移方向为左上 右上 右下 左下 即以左上为起始的顺时针方向
    :param img:
    :param label:
    :param offset:
    :return:
    """
    img_offset_list = []
    roi_size = 512
    region_x, region_y = get_region_coordinate(label)
    img0 = get_crop(img, region_x, region_y, roi_size)
    img_offset_list.append(img0)
    img1 = get_crop(img, region_x+offset, region_y+offset, roi_size)
    img_offset_list.append(img1)
    img2 = get_crop(img, region_x+offset, region_y-offset, roi_size)
    img_offset_list.append(img2)
    img3 = get_crop(img, region_x-offset, region_y-offset, roi_size)
    img_offset_list.append(img3)
    img4 = get_crop(img, region_x-offset, region_y+offset, roi_size)
    img_offset_list.append(img4)
    crop_label = get_crop(label, region_x, region_y, roi_size)
    return img_offset_list, crop_label


def get_back_region_img_list(img_list, offset=64):
    back_img_list = []
    width, _, channel = img_list[0].shape
    back_img0 = img_list[0]
    back_img_list.append(back_img0)
    back_img1 = np.zeros((width, width, channel))
    back_img1[offset:, offset:] = img_list[1][:width-offset, :width-offset]
    back_img_list.append(back_img1)
    back_img2 = np.zeros((width, width, channel))
    back_img2[offset:, :width-offset] = img_list[2][:width-offset, offset:]
    back_img_list.append(back_img2)
    back_img3 = np.zeros((width, width, channel))
    back_img3[:width-offset, :width-offset] = img_list[3][offset:, offset:]
    back_img_list.append(back_img3)
    back_img4 = np.zeros((width, width, channel))
    back_img4[:width-offset, offset:] = img_list[4][offset:, :width-offset]
    back_img_list.append(back_img4)
    mean_back_img = np.zeros((width, width, channel))
    for back_img in back_img_list:
        mean_back_img += 0.2 * back_img
    return mean_back_img


def get_region_coordinate(gt_img, mode='', input_channel=0, background=255):
    """
    获得视杯视盘连通域的中心坐标 我目前背景色为白色 若您的数据集背景为黑 请对label函数的background自行更改
    :param gt_img:
    :param mode:
    :param input_channel:
    :param background:
    :return:
    """
    if mode == '':
        mode = config_init.get_str_info(section='experiment_info', key='mode')
    if input_channel == 0:
        input_channel = config_init.get_int_info(section='experiment_info', key='input_channel')
    if input_channel == 3:
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
    if mode == 'od':
        _, gt_img = cv2.threshold(gt_img, 128, 255, cv2.THRESH_BINARY)  # 视盘二值化
    if mode == 'oc':
        _, gt_img = cv2.threshold(gt_img, 0, 255, cv2.THRESH_BINARY)  # 视杯二值化
    labels, labels_num = label(gt_img, background=background, return_num=True)   # label输出一个联通区域 每个联通区域以一个整形数字来进行表示
    regions = regionprops(labels)   # 获取各个联通域的属性

    if labels_num == 0:
        print("[error]labels_num = 0 -----------------")
        region_x = gt_img.shape[1] * 0.5
        region_y = gt_img.shape[0] * 0.5
    elif labels_num == 1:
        region_x = regions[0].centroid[0]
        region_y = regions[0].centroid[1]
    else:
        gt_img = fill_max_region(gt_img)
        labels_temp = label(gt_img, background=255)
        regions_temp = regionprops(labels_temp)
        region_x = regions_temp[0].centroid[0]
        region_y = regions_temp[0].centroid[1]

    return region_x, region_y


def get_crop(img, region_x, region_y, size):
    """
    裁剪单张图片为正方形
    :param img:
    :param region_x:
    :param region_y:
    :param size:
    :return:
    """
    y0_border = 0
    if int(region_y - 0.5 * size) > 0:
        y0 = int(region_y - 0.5 * size)
    else:
        y0 = 0
        y0_border = int(0.5 * size - region_y)

    y1 = int(region_y + 0.5 * size) if int(region_y + 0.5 * size) < img.shape[1] else img.shape[1]
    x0 = int(region_x - 0.5 * size) if int(region_x - 0.5 * size) > 0 else 0
    x1 = int(region_x + 0.5 * size) if int(region_x + 0.5 * size) < img.shape[0] else img.shape[0]
    img = img[x0: x1, y0: y1]
    img = cv2.copyMakeBorder(img, 0, 0, y0_border, 0, cv2.BORDER_REPLICATE)
    if img.shape[1] == (size - 1):
        img = cv2.copyMakeBorder(img, 0, 0, 1, 0, cv2.BORDER_REPLICATE)

    return img


def get_od_label(label):
    """
    获得视盘二值化图像
    :param label:
    :return:
    """
    _, label = cv2.threshold(label, 128, 255, cv2.THRESH_BINARY)  # 视盘二值化
    return label


def fill_max_region(img):
    """
    将小区域进行填补 填补颜色为背景色 保留最大连通域 目前背景色为白色 若背景色为黑 可将255改为0
    :param img:
    :return:
    """
    labels, labels_num = label(img, connectivity=2, return_num=True)
    regions = regionprops(labels)
    best_area = regions[0].area
    best_region_x = regions[0].centroid[0]
    best_region_y = regions[0].centroid[1]
    for region in regions:
        if region.area > best_area:
            best_area = region.area
            best_region_x = region.centroid[0]
            best_region_y = region.centroid[1]
    for region in regions:
        if region.centroid[0] != best_region_x and region.centroid[1] != best_region_y:
            coords = region.coords
            for coord in coords:
                img[coord[0], coord[1]] = 255
    return img
