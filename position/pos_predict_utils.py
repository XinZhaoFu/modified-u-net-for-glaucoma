from model.unet_pos import UNet_pos
from position.pos_data_loader import Data_Loader
import tensorflow as tf
import numpy as np
import cv2
from utils.utils import binary_predict_value_to_img, get_region_coordinate


class Pos_Predict_Utils:
    """
    将传入图片通过unet_pos获得坐标 同时坐标进行相应的放缩
    """
    def __init__(self, checkpoint_save_path='./checkpoint/pos_checkpoint/unet_pos.ckpt'):
        self.checkpoint_save_path = checkpoint_save_path
        self.data_loader = Data_Loader()
        self.model = UNet_pos()
        self.img_width = 512

        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy']
        )
        self.model.load_weights(checkpoint_save_path)

    def predict(self, img):
        channel = 1
        if img.ndim == 3:
            width, _, channel = img.shape
        else:
            width, _ = img.shape

        img = cv2.resize(img, dsize=(self.img_width, self.img_width))
        img = img / 255.
        img = np.reshape(img, (self.img_width, self.img_width, channel))
        img_temp = np.empty((1, self.img_width, self.img_width, 1))
        img_temp[0, :, :, :] = img
        predict = self.model.predict(img_temp)

        predict_temp = np.reshape(predict, (512, 512, 2))
        predict_img = binary_predict_value_to_img(predict_temp)
        region_x, region_y = get_region_coordinate(predict_img, background=0)

        region_x = region_x * width / self.img_width
        region_y = region_y * width / self.img_width
        return region_x, region_y
