import datetime
from utils.utils import print_cost_time
from model.unet_seg import UNet_seg
import numpy as np
from data_loader import Data_Loader
import tensorflow as tf
import cv2
import glob
from utils.utils import binary_predict_value_to_img, get_back_region_img_list, recreate_dir
from utils.config_utils import get_info_for_predict

np.set_printoptions(threshold=np.inf)
start_time = datetime.datetime.now()

test_label_file_path, test_predict_file_path, checkpoint_save_path = get_info_for_predict()
recreate_dir(test_predict_file_path)
data_loader = Data_Loader()
test_img, _ = data_loader.get_test_data()

# 加载模型
model = UNet_seg()
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

model.load_weights(checkpoint_save_path)

# 获取预测图的文件名
test_label_list = glob.glob(test_label_file_path + '*.bmp')
img_name_list = []
for img_file in test_label_list:
    img_name = (img_file.split('\\')[-1])
    img_name_list.append(img_name)

# 获取预测图
index_temp1 = 0
index_temp2 = 0
predict_temp_list = []
for img in test_img:
    img_temp = np.empty((1, 512, 512, 1))
    img_temp[0, :, :, :] = img
    predict = model.predict(img_temp)
    predict_temp = np.reshape(predict, (512, 512, 2))
    predict_img = binary_predict_value_to_img(predict_temp)
    predict_temp_list.append(predict_img)
    index_temp1 += 1
    if index_temp1 % 5 == 0:
        print(index_temp2, img_name_list[index_temp2])
        mean_predict = get_back_region_img_list(predict_temp_list)
        _, mean_predict = cv2.threshold(mean_predict, 125, 255, cv2.THRESH_BINARY)
        cv2.imwrite(test_predict_file_path+img_name_list[index_temp2], mean_predict)
        predict_temp_list.clear()
        index_temp2 += 1

# 获取预测图
# index = 0
# for img in test_img:
#     if index % 5 == 0:
#         img_name = img_name_list[int(index/5)]
#         img_temp = np.empty((1, 512, 512, 1))
#         img_temp[0, :, :, :] = img
#         predict = model.predict(img_temp)
#         predict_temp = np.reshape(predict, (512, 512, 2))
#         predict_img = binary_predict_value_to_img(predict_temp)
#         cv2.imwrite(test_predict_file_path+img_name, predict_img)
#         print(index/5, img_name)
#     index += 1

print_cost_time(start_time)
