import time
from tensorflow.keras.metrics import AUC, MeanIoU
import glob
import cv2
import datetime
import numpy as np
from utils.utils import print_cost_time, binary_predict_value_to_img
from utils.config_utils import get_info_for_metrics
from data_loader import Data_Loader


def get_mean_iou(label_value, predict_value, num_classes=2):
    m = MeanIoU(num_classes=num_classes)
    _ = m.update_state(y_true=label_value, y_pred=predict_value)
    iou = m.result().numpy()
    return iou


def get_dice(label_value, predict_value):
    iou = get_mean_iou(label_value, predict_value)
    dice = (2 * iou) / (1 + iou)
    return dice


def get_auc(label_value, predict_value):
    m = AUC()
    _ = m.update_state(y_true=label_value, y_pred=predict_value)
    auc = m.result().numpy()
    return auc


# 进行计时
start_time = datetime.datetime.now()

data_loader = Data_Loader()
_, test_label_list = data_loader.get_test_data()
test_predict_file_path, mode = get_info_for_metrics()
print(test_predict_file_path)
test_predict_file_list = glob.glob(test_predict_file_path + '*.bmp')
print(len(test_label_list), len(test_predict_file_list))
assert(len(test_label_list) == len(test_predict_file_list))

# 计算每张图的指标 并统计平均值
auc_sum, iou_sum, dice_sum = 0, 0, 0
for test_label, predict_file in zip(test_label_list, test_predict_file_list):
    img_name = (predict_file.split('\\')[-1]).split('.')[0]

    predict = cv2.imread(predict_file, 0)/255
    label = binary_predict_value_to_img(test_label)
    label = np.reshape(label, predict.shape)
    label = label / 255.

    auc = get_auc(label, predict)
    iou = get_mean_iou(label, predict, num_classes=2)
    dice = get_dice(label, predict)

    print('name: ' + img_name + '\t\tauc: ' + str(auc) + '\t\tiou: ' + str(iou) + '\t\tdice: ' + str(dice))

    auc_sum += auc
    iou_sum += iou
    dice_sum += dice


print('[info]-----')
print('auc: ' + str(auc_sum/len(test_predict_file_list)) + '\tiou: ' +
      str(iou_sum/len(test_predict_file_list)) + '\tdice: ' + str(dice_sum/len(test_predict_file_list)))

info = mode + '\t' + '整合测试'
log = open("./log/exp_log.txt", 'a')
print("时间:\t", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), file=log)
print("备注信息:\t", info, file=log)
print("AUC:\t", auc_sum/len(test_predict_file_list), file=log)
print("IoU:\t", iou_sum/len(test_predict_file_list), file=log)
print("dice:\t", dice_sum/len(test_predict_file_list), file=log)
print("\n", file=log)
log.close()

print_cost_time(start_time)
