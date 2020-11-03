import numpy as np
import cv2
import glob
from data_loader import Data_Loader
from position.pos_data_loader import Data_Loader as Pos_Data_Loader
# from data_loader import Data_Loader
from utils.config import config_init
from utils.utils import get_region_coordinate, get_crop
from position.pos_predict_utils import Pos_Predict_Utils

np.set_printoptions(threshold=np.inf)

img = cv2.imread('./data/V0009.jpg', 0)
cv2.imwrite('./data/V0009.bmp', img)

# img = cv2.imread('d:/71623f75611151183531842637f2201.jpg')
# img = cv2.resize(img, (413, 626))
# cv2.imwrite('d:/img.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 50])

# pos_data_loader = Pos_Data_Loader(data_path='./data/refuge/', hdf5_path='./data/hdf5/pos_hdf5/')
# train_img, train_label = pos_data_loader.get_train_data()
# val_img, val_label = pos_data_loader.get_val_data()
#
# img = np.empty((512, 512, 1))
# img_temp = train_img[0] * 255
# print(img_temp)
# img[:, :, :] = img_temp[:, :, :]
# cv2.imwrite('./train_img0.jpg', img)

# pos_predict_utils = Pos_Predict_Utils()
# img = cv2.imread('./data/refuge/img/V0001.jpg', 0)
# img = img / 255.
# region_x, region_y = pos_predict_utils.predict(img)
# print(region_x, region_y)
# img = img * 255
# img1 = get_crop(img, region_x, region_y, 512)
# img2 = get_crop(img, region_y, region_x, 512)
# cv2.imwrite('./demo1.jpg', img1)
# cv2.imwrite('./demo2.jpg', img2)


# img = cv2.imread('./data/refuge/img/V0001.jpg', 0)
# img = cv2.resize(img, (512, 512))
# print(img.shape)
# cv2.imshow('', img)
# cv2.waitKey(0)
# data_loader = Data_Loader()
# train_img, train_label = data_loader.get_test_data()
#
# print(train_label[0].shape)
# img_temp = train_img[52] * 255.
# img_temp = np.reshape(img_temp, (512, 512))
# # print(img_temp)
# img = np.zeros((512, 512))
# img[:, :] = img_temp[:, :]
# # img_temp = np.reshape(img_temp, (512, 512, 1))
# cv2.imwrite('../01.jpg', img)

# label_list_path = './data/refuge/label/'
# label_file_list = glob.glob(label_list_path+'*.bmp')
# print(len(label_file_list))
#
# for label_file in label_file_list:
#     name = label_file.split('\\')[-1]
#     label = cv2.imread(label_file)
#     region_x, region_y = get_region_coordinate(label)
#     print(name, region_x, region_y)

# data_loader = Data_Loader()
# test_img, _ = data_loader.get_val_data()
# print(len(test_img))
# img_temp = np.empty((1, 512, 512, 1))
# img_temp[0, :, :, :] = test_img[0] * 255.
# img_temp = np.reshape(img_temp, (512, 512, 1))
# print(img_temp)
# cv2.imshow('', img_temp)
# cv2.waitKey(0)
# create_folder('./logs/logs/logs')

# img_width = 512
# data_path = './data/'
# hdf5_path = './hdf5/'
# data_loader = Data_Loader(channel=1, rot_num=4, data_path=data_path, hdf5_path=hdf5_path, img_width=img_width, rewrite=True)
# train_img, train_label = data_loader.get_train_data()
#
# print(train_label[0])

# config_init = config_init()
# # # label_file_path = config_init.get_str_info(section="optic_cup_data_path", key="oc_data_test_label_path")
# # mode = config_init.get_str_info(section="experiment_info", key="mode")
# img_width = config_init.get_int_info(section="experiment_info", key="img_width")
# print(img_width)
# # print(mode)
# print(config_init.get_section())

# config = configparser.ConfigParser()
# config.read("config.ini")
# print(config.sections())

# model = UNet(filters=32, input_width=512)
# model.compile(
#     optimizer='adam',
#     loss=tf.keras.losses.MeanSquaredError(),
#     metrics=['accuracy']
# )
# checkpoint_save_path = './checkpoint/unet.ckpt'
# model.load_weights(checkpoint_save_path)
#
# img_demo = cv2.imread('./data/train/img/V0001.jpg')
# img_temp = np.empty((1, 512, 512, 3))
# img_temp[0, :, :, :] = img_demo/255.
# predict = model.predict(img_temp)
# # print(predict)
# predict_temp = np.empty((512, 512, 1))
# predict_temp[:, :, 0] = np.squeeze(predict*255.)
# cv2.imwrite('./data/test/predict/demo.bmp', predict_temp)













# train_img, train_label = data_loader.get_train_data()
#
# train_img_temp = np.empty((512, 512, 3), dtype=np.uint8)
# train_img_temp[:, :, :] = train_img[0]*255.
# cv2.imshow('train_img0', train_img_temp)
# cv2.waitKey(0)
#
# train_label_temp = np.empty((512, 512, 1), dtype=np.uint8)
# train_label_temp[:, :, ] = train_label[0]*255.
# print(train_label_temp)
# cv2.imshow('train_label0', train_label_temp)
# cv2.waitKey(0)

# img_read_path = './refuge/img/'
# label_read_path = './refuge/label/'
#
# img_file_list = glob.glob(img_read_path + '*.jpg')
# label_file_list = glob.glob(label_read_path + '*.bmp')
#
# index = [i for i in range(len(img_file_list))]
# np.random.shuffle(index)
# img_file_list = np.array(img_file_list)[index]
# label_file_list = np.array(label_file_list)[index]
#
# data_list = zip(img_file_list, label_file_list)
#
# for img_file, label_file in data_list:
#     print(img_file, label_file)


#
# data_path = './data/'
# hdf5_path = './hdf5/'
# data_loader = Data_Loader(data_path=data_path, hdf5_path=hdf5_path)
# data_loader.dataset_to_hdf5(is_rewrite=False)
# train_img, train_label = data_loader.get_train_data()
# test_img, test_label = data_loader.get_test_data()
# model = SegmentionModel()
# model.compile(
#     optimizer='adam',
#     loss=tf.keras.losses.BinaryCrossentropy(),
#     metrics=['accuracy']
# )
#
# checkpoint_save_path = './checkpoint/unet.ckpt'
#
# if os.path.exists(checkpoint_save_path+'.index'):
#     print("[INFO] loading weights")
#     model.load_weights(checkpoint_save_path)
#
# checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_save_path,
#     save_weights_only=True,
#     save_best_only=True
# )
#
# history = model.fit(
#     train_img, train_label, batch_size=32, epochs=5,
#     validation_data=(test_img, test_label), validation_freq=1,
#     callbacks=[checkpoint_callback]
# )
#
# model.summary()

# cifar10 = tf.keras.datasets.cifar10
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train.shape)
#
# f = h5py.File("./hdf5/train_img.hdf5", "w")
# f = h5py.File("./hdf5/train_label.hdf5", "w")
# f = h5py.File("./hdf5/test_img.hdf5", "w")
# f = h5py.File("./hdf5/test_label.hdf5", "w")

# np.set_printoptions(threshold=np.inf)
#
# data_loader = Data_Loader(path='./data/')
# x_train, y_train = data_loader.load_train_data()
# print(x_train.shape, y_train.shape)

# label = cv2.imread('./org_data/label/V0003.bmp')
# print(label[256:257, :, ])

# img_read_path = './org_data/img/'
# label_read_path = './org_data/label/'
# train_img_save_path = './data/train/img/'
# train_label_save_path = './data/train/label/'
# test_img_save_path = './data/test/img/'
# test_label_save_path = './data/test/label/'

# img_file_list = glob.glob(img_read_path+'*.jpg')
# label_file_list = glob.glob(label_read_path+'*.bmp')
# print(len(img_file_list), len(label_file_list))

# index = 0
# for img_file, label_file in zip(img_file_list, label_file_list):
#     index += 1
#     img_name = (img_file.split('\\')[-1]).split('.')[0]
#     org_img = cv2.imread(img_file)
#     org_label = cv2.imread(label_file)
#     up_img = cv2.pyrDown(cv2.pyrDown(org_img))
#     up_label = cv2.pyrDown(cv2.pyrDown(org_label))
#     if index <= 60:
#         cv2.imwrite(train_img_save_path+img_name+'.jpg', up_img)
#         cv2.imwrite(train_label_save_path+img_name+'.bmp', up_label)
#     else:
#         cv2.imwrite(test_img_save_path+img_name+'.jpg', up_img)
#         cv2.imwrite(test_label_save_path+img_name+'.bmp', up_label)
