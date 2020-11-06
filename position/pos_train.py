from model.unet_pos import UNet_pos
import tensorflow as tf
import os
from position.pos_data_loader import Data_Loader


load_weights = False
checkpoint_save_path = '../checkpoint/pos_checkpoint/unet_pos.ckpt'
batch_size = 4
epochs = 0
rewrite_hdf5 = False
data_loader = Data_Loader(rewrite_hdf5=rewrite_hdf5)

train_img, train_label = data_loader.get_train_data()
val_img, val_label = data_loader.get_val_data()

model = UNet_pos()
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy()
)
# model.compile(
#     optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
#     loss=tf.keras.losses.BinaryCrossentropy()
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


