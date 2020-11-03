import time
from glob import glob
import cv2
import numpy as np
from tensorflow.keras.metrics import AUC

oc_predict_file_path = './data/oc_data/test/predict/'
od_predict_file_path = './data/od_data/test/predict/'
ocd_label_file_path = './data/ocd_data/label/'
ocd_predict_file_path = './data/ocd_data/predict/'
ocd_predict_img_size = 512

oc_predict_file_list = glob(oc_predict_file_path+'*.bmp')
od_predict_file_list = glob(od_predict_file_path+'*.bmp')
ocd_label_file_list = glob(ocd_label_file_path+'*.bmp')

assert(len(oc_predict_file_list) == len(od_predict_file_list) == len(ocd_label_file_list))


def get_auc(label_value, predict_value):
    m = AUC()
    _ = m.update_state(y_true=label_value, y_pred=predict_value)
    auc = m.result().numpy()
    return auc


auc_sum = 0
index = 0
for oc_predict_path, od_predict_path, ocd_label_path in zip(oc_predict_file_list, od_predict_file_list, ocd_label_file_list):
    oc_predict_name = (oc_predict_path.split('\\')[-1]).split('.')[0]
    od_predict_name = (od_predict_path.split('\\')[-1]).split('.')[0]

    if oc_predict_name != od_predict_name:
        print(oc_predict_name, od_predict_name)
        print('[info]oc_predict_name != od_predict_name')
        continue

    oc_predict_img = cv2.imread(oc_predict_path, 0)
    od_predict_img = cv2.imread(od_predict_path, 0)
    ocd_label_img = cv2.imread(ocd_label_path, 0)
    ocd_predict_img = np.zeros(shape=(ocd_predict_img_size, ocd_predict_img_size))

    oc_predict_img_temp = np.zeros(shape=(ocd_predict_img_size, ocd_predict_img_size))
    oc_predict_img_temp[:, :] = (oc_predict_img[:, :] / 255) * 128

    od_predict_img_temp = np.zeros(shape=(ocd_predict_img_size, ocd_predict_img_size))
    od_predict_img_temp[:, :] = (od_predict_img[:, :] / 255) * 127

    ocd_predict_img[:, :] = oc_predict_img_temp[:, :] + od_predict_img_temp[:, :]
    cv2.imwrite(ocd_predict_file_path+oc_predict_name+'.bmp', ocd_predict_img)

    auc = get_auc(ocd_label_img/255., ocd_predict_img/255.)
    auc_sum += auc
    index += 1
    print(str(index) + '\t' + oc_predict_name + '\tauc:\t' + str(auc))


print('mean_auc:\t' + str(auc_sum/len(ocd_label_file_list)))

info = '\t' + '测试'
log = open("./log/exp_log.txt", 'a')
print("时间:\t", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), file=log)
print("备注信息:\t", info, file=log)
print("整体AUC:\t", str(auc_sum/len(ocd_label_file_list)), file=log)
print("\n", file=log)
log.close()
