import time
from random import randint, shuffle, seed
import cv2
import numpy as np
from utils.config_utils import get_info_for_augmentation


def img_rotate(rot_img, rot_label, rot_num=4):
    """
    图像及标注旋转 默认含四个方向的图像 0度 90度 180度 270度
    若传入所需旋转图数大于4，则额外产生旋转图使得数量补齐
    若传入所需旋转图小于4，则在四个方向中进行随机
    若传入所需旋转图数小于1，则采用默认容量为4的四向列表
    :param rot_img:待旋转图像
    :param rot_label:待旋转标注图像
    :param rot_num:希望多少张旋转图(含原图)
    :return:所得图像列表，所得标注列表，所得名字列表(原图名+’_r角度值‘)
    """
    rot_img_list = []
    rot_label_list = []
    rotated_list = [0, 90, 180, 270]
    if rot_num > 4:
        for _ in range(rot_num - 4):
            rotated_list.append(randint(1, 360))
    if 4 > rot_num > 0:
        shuffle(rotated_list)
        rotated_list = rotated_list[:rot_num]
    if rot_num <= 0:
        return rot_img, rot_label
    col, row, _ = rot_img.shape

    for rotated in rotated_list:
        rotated_matrix = cv2.getRotationMatrix2D((col / 2, row / 2), rotated, 1)
        rot_img_temp = cv2.warpAffine(rot_img, rotated_matrix, (col, row))
        rot_img_temp = np.reshape(rot_img_temp, rot_img.shape)
        rot_img_list.append(rot_img_temp)
        rot_label_temp = cv2.warpAffine(rot_label, rotated_matrix, (col, row))
        rot_label_temp = np.reshape(rot_label_temp, rot_img.shape)
        rot_label_list.append(rot_label_temp)

    return rot_img_list, rot_label_list


def img_rotate_back(rot_img_list):
    """
    将旋转后的图片旋转回来
    :param rot_img_list:
    :return:
    """
    rotated_list = [0, -90, -180, -270]
    col, row = rot_img_list[0].shape
    back_img_list = []

    for rotated, rot_img in zip(rotated_list, rot_img_list):
        rotated_matrix = cv2.getRotationMatrix2D((col / 2, row / 2), rotated, 1)
        rot_img_temp = cv2.warpAffine(rot_img, rotated_matrix, (col, row))
        back_img_list.append(rot_img_temp)

    return back_img_list


def cutout(img, mask_rate=0.5):
    """
    对正方形图片进行cutout 遮盖位置随机
        长方形需要改一下
    遮盖长度为空时用默认值图像尺寸的一半
        依据论文作者，过拟合增大，欠拟合缩小，自行调节
    添加遮盖前 对图像一圈进行0填充
    :param img: 输入应为正方形图像
    :param mask_rate: cutout的遮盖图形设定为正方形，该变量为其边长与图像边长的比例
    :return:cutout后的图像
    """
    length, _, channel = img.shape
    mask_length = int(length * mask_rate)
    region_x, region_y = randint(0, int(length + mask_length)), randint(0, int(length + mask_length))

    fill_img = np.zeros((int(length + mask_length * 2), int(length + mask_length * 2), channel))
    fill_img[int(mask_length):int(mask_length + length), int(mask_length):int(mask_length + length)] = img
    fill_img[region_x:int(region_x + mask_length), region_y:int(region_y + mask_length)] = 0
    img = fill_img[int(mask_length):int(mask_length + length), int(mask_length):int(mask_length + length)]

    return img


def gridMask(img, rate=0.5):
    """
    对图片进行gridmask 每行每列各十个 以边均匀十等分 每一长度中包含mask长度、offset偏差和留白
        长方形需要改一下
    遮盖长度为空时用默认值图像尺寸的一半
        盲猜，过拟合增大，欠拟合缩小，自行调节
    :param img: 输入应为正方形图像
    :param rate: mask长度与十分之一边长的比值
    :return: gridmask后的图像
    """
    img_length, _, channel = img.shape
    fill_img_length = int(img_length + 0.2 * img_length)
    offset = randint(0, int(0.1 * fill_img_length))
    mask_length = int(0.1 * fill_img_length * rate)
    fill_img = np.zeros((fill_img_length, fill_img_length, channel))
    fill_img[int(0.1 * img_length):int(0.1 * img_length) + img_length, int(0.1 * img_length):int(0.1 * img_length) + img_length] = img
    for width_num in range(10):
        for length_num in range(10):
            length_base_patch = int(0.1 * fill_img_length * length_num) + offset
            width_base_patch = int(0.1 * fill_img_length * width_num) + offset
            fill_img[length_base_patch:length_base_patch + mask_length, width_base_patch:width_base_patch + mask_length, ] = 0
    img = fill_img[int(0.1 * img_length):int(0.1 * img_length) + img_length, int(0.1 * img_length):int(0.1 * img_length) + img_length]

    return img


def img_augmentation(img, label, rot_num=0, cutout_num=0, cutout_mask_rate=0.2, grid_num=0, grid_mask_rate=0.2):
    """
    数据增强 获得一组原图与翻转的旋转图 rot_num*2个
    起初是对旋转图添加cutout与gridMask掩码
    那么总数为rot_num*2*(1+cutout_num+grid_num)
    这里产生的数据集过多 训练的时候承载力有限
    后对代码进行更改 使其总数为img*8 在保证rot_num为4的情况下 对(1+cutout_num+grid_num)进行更改
    更替为普通翻转图 或一个cutout 或一个grid 即(普通翻转图||cutout_num||grid_num)
    如果不需要这种三选一的方式 删除标识代码
    :param img:
    :param label:
    :param rot_num:
    :param cutout_num:
    :param cutout_mask_rate:
    :param grid_num:
    :param grid_mask_rate:
    :return:
    """
    if rot_num == 0:
        rot_num, cutout_num, cutout_mask_rate, grid_num, grid_mask_rate = get_info_for_augmentation()
    flip_img = cv2.flip(img, 1)
    flip_img = np.reshape(flip_img, img.shape)
    flip_label = cv2.flip(label, 1)
    flip_label = np.reshape(flip_label, label.shape)
    img_list = []
    label_list = []

    nor_img_list, nor_label_list = img_rotate(img, label, rot_num=rot_num)
    flip_img_list, flip_label_list = img_rotate(flip_img, flip_label, rot_num=rot_num)

    img_list.extend(nor_img_list)
    label_list.extend(nor_label_list)
    img_list.extend(flip_img_list)
    label_list.extend(flip_label_list)

    aug_img_list = []
    aug_label_list = []

    for img, label in zip(img_list, label_list):
        # 三选一标识生成    也可是二选一 一选一  删除时删除if判别及其内容
        seed(time.time())
        random_flag_list = [1]
        if cutout_num != 0 or grid_num != 0:
            for _ in range(cutout_num+grid_num):
                random_flag_list.append(0)
                # random_flag_list.append(1)
        shuffle(random_flag_list)

        # 使用标识码判别 删除时只删除if所在行语句 不删除if内语句
        if random_flag_list[0] == 1:
            aug_img_list.append(img)
            aug_label_list.append(label)

        # 使用标识码判别 删除时只删除if所在行语句 不删除if内语句
        if cutout_num != 0:
            for i in range(1, cutout_num+1):
                if random_flag_list[i] == 1:
                    img_temp = np.reshape(img, img.shape)
                    img_temp = cutout(img_temp, mask_rate=cutout_mask_rate)
                    aug_img_list.append(img_temp)
                    aug_label_list.append(label)

        # 使用标识码判别 删除时只删除if所在行语句 不删除if内语句
        if grid_num != 0:
            for i in range(1, grid_num+1):
                if random_flag_list[cutout_num+i] == 1:
                    img_temp = np.reshape(img, img.shape)
                    img_temp = gridMask(img_temp, rate=grid_mask_rate)
                    aug_img_list.append(img_temp)
                    aug_label_list.append(label)

    return aug_img_list, aug_label_list
