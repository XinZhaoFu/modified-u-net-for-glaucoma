import datetime
from utils.utils import print_cost_time
from model.unet_pos import UNet_pos
import numpy as np
import cv2
from position.pos_data_loader import Data_Loader
import tensorflow as tf
from utils.utils import binary_predict_value_to_img, get_region_coordinate

np.set_printoptions(threshold=np.inf)
start_time = datetime.datetime.now()
img_width = 512
checkpoint_save_path = '../checkpoint/pos_checkpoint/unet_pos.ckpt'

data_loader = Data_Loader()
test_img, test_label = data_loader.get_test_data()

# 加载模型
model = UNet_pos()
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

model.load_weights(checkpoint_save_path)

# 获取预测图
index_temp = 0
predict_temp_list = []
for img in test_img:
    img_temp = np.empty((1, img_width, img_width, 1))
    img_temp[0, :, :, :] = img
    predict = model.predict(img_temp)
    predict_temp = np.reshape(predict, (512, 512, 2))
    predict_img = binary_predict_value_to_img(predict_temp)
    region_x, region_y = get_region_coordinate(255-predict_img)
    label = np.empty((512, 512, 1))
    label_temp = test_label[index_temp]
    label[:, :, :] = label_temp[:, :, :]
    region_x_temp, region_y_temp = get_region_coordinate(label)
    print(index_temp, region_x-region_x_temp, region_y-region_y_temp)
    index_temp += 1

print_cost_time(start_time)
