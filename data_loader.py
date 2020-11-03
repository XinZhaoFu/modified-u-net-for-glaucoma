import glob
import cv2
import numpy as np
from utils.utils import load_hdf5, write_hdf5, shuffle_file, get_offset_region_img_list
from augmentation import img_augmentation
from utils.config_utils import get_info_for_data_loader


class Data_Loader:
    def __init__(self):
        """
        从config中获取数值 设置好数值及路径 rewrite为真则重新生产hdf5文件 反之则反
        hdf5文件共6个 训练图片及标签 验证图片及标签 测试图片及标签   标签以独热码形式保存
        """
        input_channel, img_width, num_class, rot_num, rewrite, data_path, hdf5_path = get_info_for_data_loader()
        print('[info]data loading----')
        print(data_path, hdf5_path)
        self.data_path = data_path
        self.hdf5_path = hdf5_path
        self.img_width = img_width
        self.num_class = num_class
        self.channel = input_channel
        self.rot_num = rot_num
        self.train_img_path = self.data_path + 'train/img/' + '*.jpg'
        self.train_label_path = self.data_path + 'train/label/' + '*.bmp'
        self.val_img_path = self.data_path + 'validation/img/' + '*.jpg'
        self.val_label_path = self.data_path + 'validation/label/' + '*.bmp'
        self.test_img_path = self.data_path + 'test/img/' + '*.jpg'
        self.test_label_path = self.data_path + 'test/label/' + '*.bmp'

        self.train_img_hdf5 = self.hdf5_path + "train_img.hdf5"
        self.train_label_hdf5 = self.hdf5_path + "train_label.hdf5"
        self.val_img_hdf5 = self.hdf5_path + "val_img.hdf5"
        self.val_label_hdf5 = self.hdf5_path + "val_label.hdf5"
        self.test_img_hdf5 = self.hdf5_path + "test_img.hdf5"
        self.test_label_hdf5 = self.hdf5_path + "test_label.hdf5"

        if rewrite:
            print("[INFO] rewrite hdf5")
            train_img, train_label = self.load_train_data(self.train_img_path, self.train_label_path)
            write_hdf5(train_img, self.train_img_hdf5)
            write_hdf5(train_label, self.train_label_hdf5)
            val_img, val_label = self.load_validation_data(self.val_img_path, self.val_label_path)
            write_hdf5(val_img, self.val_img_hdf5)
            write_hdf5(val_label, self.val_label_hdf5)
            test_img, test_label = self.load_test_data(self.test_img_path, self.test_label_path)
            write_hdf5(test_img, self.test_img_hdf5)
            write_hdf5(test_label, self.test_label_hdf5)

    def get_train_data(self):
        train_img = load_hdf5(self.train_img_hdf5)
        train_label = load_hdf5(self.train_label_hdf5)
        print('[info] reading train data')
        return train_img, train_label

    def get_val_data(self):
        val_img = load_hdf5(self.val_img_hdf5)
        val_label = load_hdf5(self.val_label_hdf5)
        print('[info] reading validation data')
        return val_img, val_label

    def get_test_data(self):
        test_img = load_hdf5(self.test_img_hdf5)
        test_label = load_hdf5(self.test_label_hdf5)
        print('[info] reading test data')
        return test_img, test_label

    def load_train_data(self, img_file_path, label_file_path):
        print("[INFO] rewrite train hdf5")
        img_file_list = glob.glob(img_file_path)
        label_file_list = glob.glob(label_file_path)
        assert (len(img_file_list) == len(label_file_list))

        img_file_list, label_file_list = shuffle_file(img_file_list, label_file_list)

        dataset_img = np.empty((len(img_file_list)*self.rot_num*2, self.img_width, self.img_width, self.channel), dtype=np.float16)
        dataset_label = np.zeros((len(label_file_list)*self.rot_num*2, self.img_width, self.img_width, self.num_class), dtype=np.uint8)

        index = 0
        for img_file, label_file in zip(img_file_list, label_file_list):
            if self.channel == 1:
                img = cv2.imread(img_file, 0) / 255.
                img = np.reshape(img, (self.img_width, self.img_width, 1))
            else:
                img = cv2.imread(img_file) / 255.
            label = cv2.imread(label_file, 0)
            label = np.reshape(label, (self.img_width, self.img_width, 1))
            aug_img_list, aug_label_list = img_augmentation(img, label)
            for aug_img, aug_label in zip(aug_img_list, aug_label_list):
                aug_img = aug_img/255.
                if self.channel == 1:
                    aug_img = np.reshape(aug_img, (self.img_width, self.img_width, 1))
                aug_label = aug_label/255.
                aug_label = np.reshape(aug_label, (self.img_width, self.img_width, 1))
                dataset_img[index, :, :, :] = aug_img
                for i in range(self.num_class):
                    dataset_label[index, :, :, i] = (aug_label[:, :, 0] == i).astype(int)
                index += 1
        return dataset_img, dataset_label

    def load_validation_data(self, img_file_path, label_file_path):
        print("[INFO] rewrite validation hdf5")
        img_file_list = glob.glob(img_file_path)
        label_file_list = glob.glob(label_file_path)
        assert (len(img_file_list) == len(label_file_list))

        dataset_img = np.empty((len(img_file_list), self.img_width, self.img_width, self.channel), dtype=np.float16)
        dataset_label = np.zeros((len(label_file_list), self.img_width, self.img_width, self.num_class), dtype=np.uint8)

        index = 0
        for img_file, label_file in zip(img_file_list, label_file_list):
            if self.channel == 1:
                img = cv2.imread(img_file, 0) / 255.
                img = np.reshape(img, (self.img_width, self.img_width, 1))
            else:
                img = cv2.imread(img_file) / 255.
            label = cv2.imread(label_file, 0) / 255.
            label = np.reshape(label, (self.img_width, self.img_width, 1))
            dataset_img[index, :, :, :] = img
            for i in range(self.num_class):
                dataset_label[index, :, :, i] = (label[:, :, 0] == i).astype(int)
            index += 1
        return dataset_img, dataset_label

    def load_test_data(self, img_file_path, label_file_path):
        print("[INFO] rewrite test hdf5")
        img_file_list = glob.glob(img_file_path)
        label_file_list = glob.glob(label_file_path)
        assert (len(img_file_list) == len(label_file_list))

        dataset_img = np.empty((len(img_file_list)*5, self.img_width, self.img_width, self.channel), dtype=np.float16)
        dataset_label = np.zeros((len(label_file_list), self.img_width, self.img_width, self.num_class), dtype=np.uint8)

        index = 0
        for img_file, label_file in zip(img_file_list, label_file_list):
            if self.channel == 1:
                img = cv2.imread(img_file, 0)
            else:
                img = cv2.imread(img_file)
            label = cv2.imread(label_file, 0)

            img_offset_list, crop_label = get_offset_region_img_list(img, label, offset=64)
            crop_label = (255-crop_label)/255.
            label = np.reshape(crop_label, (self.img_width, self.img_width, 1))

            index_temp = 0
            for img_offset in img_offset_list:
                img_offset_temp = np.reshape(img_offset, (self.img_width, self.img_width, self.channel))
                dataset_img[index*5+index_temp, :, :, :] = img_offset_temp / 255.
                index_temp += 1

            for i in range(self.num_class):
                dataset_label[index, :, :, i] = (label[:, :, 0] == i).astype(int)
            index += 1
        return dataset_img, dataset_label
