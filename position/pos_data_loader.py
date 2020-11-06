import glob
import cv2
import numpy as np
from utils.utils import load_hdf5, write_hdf5, shuffle_file, get_od_label
from augmentation import img_augmentation


class Data_Loader:
    def __init__(self, rewrite_hdf5=False, data_path='../data/refuge/', hdf5_path='../data/hdf5/pos_hdf5/', img_width=512, channel=1, num_class=2, rot_num=4):
        """
        写注释是一个程序员的本职 但是我要先去吃饭了
        :param rewrite_hdf5:
        :param data_path:
        :param hdf5_path:
        :param img_width:
        :param channel:
        :param num_class:
        :param rot_num:
        """
        self.data_path = data_path
        self.hdf5_path = hdf5_path
        self.img_width = img_width
        self.channel = channel
        self.num_class = num_class
        self.rot_num = rot_num
        self.dataset_img_path = self.data_path + "img/"
        self.dataset_label_path = self.data_path + "label/"
        self.train_img_hdf5 = self.hdf5_path + "train_img.hdf5"
        self.train_label_hdf5 = self.hdf5_path + "train_label.hdf5"
        self.val_img_hdf5 = self.hdf5_path + "val_img.hdf5"
        self.val_label_hdf5 = self.hdf5_path + "val_label.hdf5"
        self.test_img_hdf5 = self.hdf5_path + "test_img.hdf5"
        self.test_label_hdf5 = self.hdf5_path + "test_label.hdf5"

        if rewrite_hdf5:
            print("[INFO] rewrite hdf5")
            train_img_file_list, train_label_file_list, val_img_file_list, val_label_file_list, test_img_file_list, test_label_file_list = self.load_data()
            train_img, train_label = self.load_train_data(train_img_file_list, train_label_file_list)
            write_hdf5(train_img, self.train_img_hdf5)
            write_hdf5(train_label, self.train_label_hdf5)
            val_img, val_label = self.load_validation_data(val_img_file_list, val_label_file_list)
            write_hdf5(val_img, self.val_img_hdf5)
            write_hdf5(val_label, self.val_label_hdf5)
            test_img, test_label = self.load_test_data(test_img_file_list, test_label_file_list)
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

    def load_train_data(self, train_img_file_list, train_label_file_list):
        print("[INFO] rewrite train hdf5")

        dataset_img = np.empty((len(train_img_file_list)*self.rot_num*2, self.img_width, self.img_width, self.channel), dtype=np.float16)
        dataset_label = np.empty((len(train_label_file_list)*self.rot_num*2, self.img_width, self.img_width, self.num_class), dtype=np.uint8)

        index = 0
        for img_file, label_file in zip(train_img_file_list, train_label_file_list):
            if self.channel == 1:
                img = cv2.imread(img_file, 0)
            else:
                img = cv2.imread(img_file)
            img = cv2.resize(img, dsize=(self.img_width, self.img_width))
            img = np.reshape(img, (self.img_width, self.img_width, self.channel))
            label = cv2.imread(label_file, 0)
            label = cv2.resize(label, dsize=(self.img_width, self.img_width))
            label = get_od_label(label)
            label = 255 - label
            label = np.reshape(label, (self.img_width, self.img_width, 1))

            aug_img_list, aug_label_list = img_augmentation(img, label, rot_num=self.rot_num, cutout_num=1, cutout_mask_rate=0.5, grid_num=1, grid_mask_rate=0.5)
            for aug_img, aug_label in zip(aug_img_list, aug_label_list):
                aug_img = aug_img/255.
                if self.channel == 1:
                    aug_img = np.reshape(aug_img, (self.img_width, self.img_width, 1))
                aug_label = aug_label / 255.
                aug_label = np.reshape(aug_label, (self.img_width, self.img_width, 1))
                dataset_img[index, :, :, :] = aug_img
                for i in range(self.num_class):
                    dataset_label[index, :, :, i] = (aug_label[:, :, 0] == i).astype(int)
                index += 1
        return dataset_img, dataset_label

    def load_validation_data(self, val_img_file_list, val_label_file_list):
        print("[INFO] rewrite validation hdf5")

        dataset_img = np.empty((len(val_img_file_list), self.img_width, self.img_width, self.channel), dtype=np.float16)
        dataset_label = np.zeros((len(val_label_file_list), self.img_width, self.img_width, self.num_class), dtype=np.uint8)

        index = 0
        for img_file, label_file in zip(val_img_file_list, val_label_file_list):
            if self.channel == 1:
                img = cv2.imread(img_file, 0)
            else:
                img = cv2.imread(img_file)
            img = cv2.resize(img, dsize=(self.img_width, self.img_width))
            img = img / 255.
            img = np.reshape(img, (self.img_width, self.img_width, self.channel))
            label = cv2.imread(label_file, 0)
            label = cv2.resize(label, dsize=(self.img_width, self.img_width))
            label = get_od_label(label)
            label = (255-label) / 255.
            label = np.reshape(label, (self.img_width, self.img_width, 1))
            dataset_img[index, :, :, :] = img
            for i in range(self.num_class):
                dataset_label[index, :, :, i] = (label[:, :, 0] == i).astype(int)
            index += 1
        return dataset_img, dataset_label

    def load_test_data(self, test_img_file_list, test_label_file_list):
        print("[INFO] rewrite test hdf5")

        dataset_img = np.empty((len(test_img_file_list), self.img_width, self.img_width, self.channel), dtype=np.float16)
        dataset_label = np.zeros((len(test_label_file_list), self.img_width, self.img_width, 1), dtype=np.float16)

        index = 0
        for img_file, label_file in zip(test_img_file_list, test_label_file_list):
            if self.channel == 1:
                img = cv2.imread(img_file, 0)
            else:
                img = cv2.imread(img_file)
            img = cv2.resize(img, dsize=(self.img_width, self.img_width))
            img = img / 255.
            img = np.reshape(img, (self.img_width, self.img_width, self.channel))
            label = cv2.imread(label_file, 0)
            label = cv2.resize(label, dsize=(self.img_width, self.img_width))
            label = get_od_label(label)
            label = np.reshape(label, (self.img_width, self.img_width, 1))
            dataset_img[index, :, :, :] = img
            dataset_label[index, :, :, :] = label
            index += 1
        return dataset_img, dataset_label

    def load_data(self):
        print("[INFO] reload refuge data")
        img_file_list = glob.glob(self.dataset_img_path+'*.jpg')
        label_file_list = glob.glob(self.dataset_label_path+'*.bmp')
        assert (len(img_file_list) == len(label_file_list))

        shuffle_file(img_file_list, label_file_list)
        train_img_file_list, train_label_file_list = img_file_list[:240], label_file_list[:240]
        val_img_file_list, val_label_file_list = img_file_list[240:320], label_file_list[240:320]
        test_img_file_list, test_label_file_list = img_file_list[320:], label_file_list[320:]

        return train_img_file_list, train_label_file_list, val_img_file_list, val_label_file_list, test_img_file_list, test_label_file_list

