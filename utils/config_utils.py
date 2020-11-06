from utils.config import config_init
from utils.utils import create_dir

config_init = config_init()


def get_path_for_data():
    mode = config_init.get_str_info(section='experiment_info', key='mode')
    img_data_path = ''
    if mode == 'oc':
        img_data_path = config_init.get_str_info(section='data_path', key='oc_data_path')
    if mode == 'od':
        img_data_path = config_init.get_str_info(section='data_path', key='od_data_path')
    train_img_file_path = img_data_path + 'train/img/'
    create_dir(train_img_file_path)
    train_label_file_path = img_data_path + 'train/label/'
    create_dir(train_label_file_path)
    validation_img_file_path = img_data_path + 'validation/img/'
    create_dir(validation_img_file_path)
    validation_label_file_path = img_data_path + 'validation/label/'
    create_dir(validation_label_file_path)
    test_img_file_path = img_data_path + 'test/img/'
    create_dir(test_img_file_path)
    test_label_file_path = img_data_path + 'test/label/'
    create_dir(test_label_file_path)
    crop_test_img_file_path = img_data_path + 'test/crop_img/'
    create_dir(crop_test_img_file_path)
    crop_test_label_file_path = img_data_path + 'test/crop_label/'
    create_dir(crop_test_label_file_path)
    test_predict_file_path = img_data_path + 'test/predict/'
    create_dir(test_predict_file_path)
    return train_img_file_path, train_label_file_path, validation_img_file_path, validation_label_file_path, test_img_file_path, test_label_file_path, crop_test_img_file_path, crop_test_label_file_path, test_predict_file_path


def get_path_for_refuge():
    refuge_path = config_init.get_str_info(section='data_path', key='refuge_path')
    refuge_img_path = refuge_path + 'img/'
    refuge_label_path = refuge_path + 'label/'
    return refuge_img_path, refuge_label_path


def get_path_for_checkpoint():
    mode = config_init.get_str_info(section='experiment_info', key='mode')
    checkpoint_save_path = ''
    if mode == 'oc':
        checkpoint_save_path = config_init.get_str_info(section='data_path', key='oc_checkpoint_save_path')
    if mode == 'od':
        checkpoint_save_path = config_init.get_str_info(section='data_path', key='od_checkpoint_save_path')
    return checkpoint_save_path


def get_info_for_get_crop_data():
    img_width = config_init.get_int_info(section='experiment_info', key='img_width')
    train_img_file_path, train_label_file_path, validation_img_file_path, validation_label_file_path, test_img_file_path, test_label_file_path, crop_test_img_file_path, crop_test_label_file_path, _ = get_path_for_data()
    refuge_img_path, refuge_label_path = get_path_for_refuge()
    mode = config_init.get_str_info(section='experiment_info', key='mode')
    return img_width, refuge_img_path, refuge_label_path, train_img_file_path, train_label_file_path, validation_img_file_path, validation_label_file_path, test_img_file_path, test_label_file_path, crop_test_img_file_path, crop_test_label_file_path, mode


def get_info_for_metrics():
    _, _, _, _, _, _, _, _, test_predict_file_path = get_path_for_data()
    mode = config_init.get_str_info(section='experiment_info', key='mode')
    return test_predict_file_path, mode


def get_info_for_predict():
    _, _, _, _, _, test_label_file_path, _, _, test_predict_file_path = get_path_for_data()
    checkpoint_save_path = get_path_for_checkpoint()
    return test_label_file_path, test_predict_file_path, checkpoint_save_path


def get_info_for_train():
    load_weights = config_init.get_bool_info(section='experiment_info', key='load_weights')
    checkpoint_save_path = get_path_for_checkpoint()
    batch_size = config_init.get_int_info(section='experiment_info', key='batch_size')
    epochs = config_init.get_int_info(section='experiment_info', key='epochs')
    return load_weights, checkpoint_save_path, batch_size, epochs


def get_info_for_unet_seg():
    filters = config_init.get_int_info(section='experiment_info', key='filters')
    img_width = config_init.get_int_info(section='experiment_info', key='img_width')
    input_channel = config_init.get_int_info(section='experiment_info', key='input_channel')
    num_class = config_init.get_int_info(section='experiment_info', key='num_class')
    num_con_unit = config_init.get_int_info(section='experiment_info', key='num_con_unit')
    return filters, img_width, input_channel, num_class, num_con_unit


def get_info_for_data_loader():
    mode = config_init.get_str_info(section='experiment_info', key='mode')
    input_channel = config_init.get_int_info(section='experiment_info', key='input_channel')
    img_width = config_init.get_int_info(section='experiment_info', key='img_width')
    num_class = config_init.get_int_info(section='experiment_info', key='num_class')
    rot_num = config_init.get_int_info(section='augmentation_info', key='rot_num')
    rewrite = config_init.get_bool_info(section='experiment_info', key='rewrite')
    data_path, hdf5_path = '', ''
    if mode == 'oc':
        data_path = config_init.get_str_info(section='data_path', key='oc_data_path')
        hdf5_path = config_init.get_str_info(section='data_path', key='oc_hdf5_path')
    if mode == 'od':
        data_path = config_init.get_str_info(section='data_path', key='od_data_path')
        hdf5_path = config_init.get_str_info(section='data_path', key='od_hdf5_path')
    create_dir(hdf5_path)
    return input_channel, img_width, num_class, rot_num, rewrite, data_path, hdf5_path


def get_info_for_augmentation():
    rot_num = config_init.get_int_info(section='augmentation_info', key='rot_num')
    cutout_num = config_init.get_int_info(section='augmentation_info', key='cutout_num')
    cutout_mask_rate = config_init.get_float_info(section='augmentation_info', key='cutout_mask_rate')
    grid_num = config_init.get_int_info(section='augmentation_info', key='grid_num')
    grid_mask_rate = config_init.get_float_info(section='augmentation_info', key='grid_mask_rate')
    return rot_num, cutout_num, cutout_mask_rate, grid_num, grid_mask_rate
