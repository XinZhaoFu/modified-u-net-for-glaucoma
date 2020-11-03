from model.unet_seg import UNet_seg
from model.xnet import X_Net
import tensorflow as tf
import os
from data_loader import Data_Loader
from utils.config_utils import get_info_for_train

load_weights, checkpoint_save_path, batch_size, epochs = get_info_for_train()
data_loader = Data_Loader()

train_img, train_label = data_loader.get_train_data()
val_img, val_label = data_loader.get_val_data()

model = UNet_seg()
# model = X_Net()
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)
# model.compile(
#     optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
#     loss=tf.keras.losses.BinaryCrossentropy(),
#     metrics=['accuracy']
# )

if os.path.exists(checkpoint_save_path+'.index') and load_weights:
    print("[INFO] loading weights")
    model.load_weights(checkpoint_save_path)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_save_path,
    monitor='val_loss',
    save_weights_only=True,
    save_best_only=True,
    mode='auto',
    save_freq='epoch'
)

history = model.fit(
    train_img, train_label, batch_size=batch_size, epochs=epochs,
    validation_data=(val_img, val_label), validation_freq=1,
    callbacks=[checkpoint_callback]
)

model.summary()


