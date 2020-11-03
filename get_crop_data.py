# 用于将数据集进行裁剪,依赖label裁剪,而非模型,训练用工具
from random import randint, seed
import glob
import numpy as np
import cv2
from utils.utils import shuffle_file, recreate_dir, get_region_coordinate, get_crop
from utils.config_utils import get_info_for_get_crop_data

roi_size, refuge_img_read_path, refuge_label_read_path, train_img_save_path, train_label_save_path, val_img_save_path, val_label_save_path, test_img_save_path, test_label_save_path, crop_test_img_save_path, crop_test_label_save_path, mode = get_info_for_get_crop_data()
print(roi_size, refuge_img_read_path, refuge_label_read_path, train_img_save_path, train_label_save_path, val_img_save_path, val_label_save_path, test_img_save_path, test_label_save_path, crop_test_img_save_path, crop_test_label_save_path, mode)


def get_random_region_coordinate(region_x, region_y):
    """
    获得随机坐标
    :param region_x:
    :param region_y:
    :return:
    """
    random_offset_x = randint(-64, 64)
    random_offset_y = randint(-64, 64)

    return region_x + random_offset_x, region_y + random_offset_y


def crop_data(img_file_list, label_file_list, img_save_path, label_save_path, offset=False, mode=mode):
    recreate_dir(img_save_path)
    recreate_dir(label_save_path)
    for img_file, label_file in zip(img_file_list, label_file_list):
        print(img_file, label_file)
        label_img = cv2.imread(label_file)
        img = cv2.imread(img_file)
        img_name = img_file.split('\\')[-1]
        label_name = label_file.split('\\')[-1]

        region_x, region_y = get_region_coordinate(label_img)
        if offset:
            region_x, region_y = get_random_region_coordinate(region_x, region_y)
        label_img = get_crop(label_img, region_x, region_y, roi_size)
        img = get_crop(img, region_x, region_y, roi_size)

        if mode == 'od':
            _, label_img = cv2.threshold(label_img, 128, 255, cv2.THRESH_BINARY)  # 视盘二值化
        if mode == 'oc':
            _, label_img = cv2.threshold(label_img, 0, 255, cv2.THRESH_BINARY)  # 视杯二值化
        # if mode == 'ocd':
        #     print('[INFO]杯盘不予区分')

        height, width, _ = img.shape
        if height != roi_size or width != roi_size:
            print('[error] img shape error---------')
            print(img_name, label_name)
        cv2.imwrite(img_save_path + img_name, img)
        cv2.imwrite(label_save_path + label_name, 255-label_img)


def save_data(img_file_list, label_file_list, img_save_path, label_save_path):
    """
    此处不进行颜色翻转 否则后续定位将出现问题 将在data_loader处进行颜色翻转
    :param img_file_list:
    :param label_file_list:
    :param img_save_path:
    :param label_save_path:
    :return:
    """
    recreate_dir(img_save_path)
    recreate_dir(label_save_path)
    for img_file, label_file in zip(img_file_list, label_file_list):
        print(img_file, label_file)
        label_img = cv2.imread(label_file)
        img = cv2.imread(img_file)
        img_name = img_file.split('\\')[-1]
        label_name = label_file.split('\\')[-1]

        if mode == 'od':
            _, label_img = cv2.threshold(label_img, 128, 255, cv2.THRESH_BINARY)  # 视盘二值化
        if mode == 'oc':
            _, label_img = cv2.threshold(label_img, 0, 255, cv2.THRESH_BINARY)  # 视杯二值化

        cv2.imwrite(img_save_path + img_name, img)
        cv2.imwrite(label_save_path + label_name, label_img)


img_file_list = glob.glob(refuge_img_read_path + "*.jpg")
label_file_list = glob.glob(refuge_label_read_path + "*.bmp")
print(len(img_file_list), len(label_file_list))
assert(len(img_file_list) == len(label_file_list))

img_file_list, label_file_list = shuffle_file(img_file_list, label_file_list)

crop_data(img_file_list[:240], label_file_list[:240], train_img_save_path, train_label_save_path, offset=True)
crop_data(img_file_list[240:320], label_file_list[240:320], val_img_save_path, val_label_save_path)
save_data(img_file_list[320:], label_file_list[320:], test_img_save_path, test_label_save_path)
crop_data(img_file_list[320:], label_file_list[320:], crop_test_img_save_path, crop_test_label_save_path)

ocd_crop_test_img_save_path = './data/ocd_data/img/'
ocd_crop_test_label_save_path = './data/ocd_data/label/'

crop_data(img_file_list[320:], label_file_list[320:], ocd_crop_test_img_save_path, ocd_crop_test_label_save_path, mode='ocd')
