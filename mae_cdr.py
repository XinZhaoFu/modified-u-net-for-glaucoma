import cv2
from glob import glob
from skimage.measure import regionprops, label

"""
    我要去玩游戏了 不写具体注释了 这是计算垂直cdr的代码
    注释这种事情下次一定
    债见
"""

oc_predict_file_path = './data/oc_data/test/predict/'
od_predict_file_path = './data/od_data/test/predict/'
oc_label_file_path = './data/oc_data/test/crop_label/'
od_label_file_path = './data/od_data/test/crop_label/'

oc_predict_file_list = glob(oc_predict_file_path + '*.bmp')
od_predict_file_list = glob(od_predict_file_path + '*.bmp')
oc_label_file_list = glob(oc_label_file_path + '*.bmp')
od_label_file_list = glob(od_label_file_path + '*.bmp')

# print(len(oc_predict_file_list), len(od_predict_file_list), len(oc_label_file_list), len(od_label_file_list))
assert(len(oc_predict_file_list) == len(od_predict_file_list)
       == len(oc_label_file_list) == len(od_label_file_list))


def get_row_value(get_row_value_img):
    labels = label(get_row_value_img, connectivity=2)
    regions = regionprops(labels)
    (min_row, _, max_row, _) = regions[0].bbox
    if max_row >= min_row:
        return max_row - min_row + 1
    else:
        return min_row - max_row + 1


sum_mae = 0
for oc_predict_path, od_predict_path, oc_label_path, od_label_path in zip(oc_predict_file_list, od_predict_file_list, oc_label_file_list, od_label_file_list):
    oc_predict_name = (oc_predict_path.split('\\')[-1]).split('.')[0]
    od_predict_name = (od_predict_path.split('\\')[-1]).split('.')[0]
    oc_label_name = (oc_label_path.split('\\')[-1]).split('.')[0]
    od_label_name = (od_label_path.split('\\')[-1]).split('.')[0]

    assert(oc_predict_name == od_predict_name == oc_label_name == od_label_name)

    oc_predict_img = cv2.imread(oc_predict_path, 0)
    od_predict_img = cv2.imread(od_predict_path, 0)
    oc_label_img = cv2.imread(oc_label_path, 0)
    od_label_img = cv2.imread(od_label_path, 0)

    predict_oc_row = get_row_value(oc_predict_img)
    predict_od_row = get_row_value(od_predict_img)
    label_oc_row = get_row_value(oc_label_img)
    label_od_row = get_row_value(od_label_img)

    mae_cdr = abs((predict_oc_row/predict_od_row) - (label_oc_row/label_od_row))
    sum_mae += mae_cdr
    print('mae:\t' + str(mae_cdr)
          + '\t\tpredict_oc_row:\t' + str(predict_oc_row)
          + '\t\tpredict_od_row:\t' + str(predict_od_row)
          + '\t\tlabel_oc_row:\t' + str(label_oc_row)
          + '\t\tlabel_od_row:\t' + str(label_od_row))

mean_mae = sum_mae / len(oc_label_file_list)
print('mean_mae:\t' + str(mean_mae))
