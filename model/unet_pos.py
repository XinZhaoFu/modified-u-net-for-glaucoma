from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from utils.config_utils import get_info_for_unet_seg


class Con_Unit(Model):
    def __init__(self, filters, init_input_shape, activation='relu'):
        super(Con_Unit, self).__init__()
        self.filters = filters
        self.init_input_shape = init_input_shape
        width, _, _ = init_input_shape
        self.follow_input_shape = (width, width, self.filters)
        self.activation = activation

        self.con = Conv2D(filters=self.filters, kernel_size=(3, 3), input_shape=self.init_input_shape, padding='same', use_bias=False, kernel_regularizer=regularizers.l2())
        self.bn = BatchNormalization(input_shape=self.follow_input_shape)
        self.act = Activation(self.activation)

    def call(self, inputs):
        con = self.con(inputs)
        bn = self.bn(con)
        out = self.act(bn)
        return out


class Con_Block(Model):
    def __init__(self, filters, input_width, input_channel, num_con_unit=1):
        super(Con_Block, self).__init__()
        self.filters = filters
        self.input_width = input_width
        self.input_channel = input_channel
        self.num_con_unit = num_con_unit
        self.init_input_shape = (self.input_width, self.input_width, self.input_channel)
        self.follow_input_shape = (self.input_width, self.input_width, self.filters)

        self.con_blocks = Sequential()
        for id_unit in range(self.num_con_unit):
            if id_unit == 0:
                block = Con_Unit(filters=self.filters, init_input_shape=self.init_input_shape)
            else:
                block = Con_Unit(filters=self.filters, init_input_shape=self.follow_input_shape)
            self.con_blocks.add(block)

    def call(self, inputs):
        out = self.con_blocks(inputs)
        return out


class Up_Block(Model):
    def __init__(self, filters, input_width, input_channel, num_con_unit=1):
        super().__init__()
        self.filters = filters
        self.input_width = input_width
        self.input_channel = input_channel
        self.num_con_unit = num_con_unit
        self.init_input_shape = (self.input_width, self.input_width, self.input_channel)
        self.follow_input_shape = (self.input_width, self.input_width, self.filters)

        self.con_blocks = Sequential()
        for id_unit in range(self.num_con_unit):
            if id_unit == 0:
                block = Con_Unit(filters=self.filters, init_input_shape=self.init_input_shape)
            else:
                block = Con_Unit(filters=self.filters, init_input_shape=self.follow_input_shape)
            self.con_blocks.add(block)
        self.up = UpSampling2D()

    def call(self, inputs):
        con = self.con_blocks(inputs)
        out = self.up(con)
        return out


class UNet_pos(Model):
    def __init__(self, filters=32, img_width=512, input_channel=1, num_class=2, num_con_unit=1):
        super(UNet_pos, self).__init__()
        self.filters = filters
        self.input_width = img_width
        self.input_channel = input_channel
        self.num_class = num_class
        self.num_con_unit = num_con_unit

        self.con_block1 = Con_Block(filters=self.filters, input_width=self.input_width, input_channel=self.input_channel, num_con_unit=self.num_con_unit)
        self.con_block2 = Con_Block(filters=self.filters, input_width=self.input_width/2, input_channel=self.filters, num_con_unit=self.num_con_unit)
        self.con_block3 = Con_Block(filters=self.filters, input_width=self.input_width/4, input_channel=self.filters, num_con_unit=self.num_con_unit)
        self.con_block4 = Con_Block(filters=self.filters, input_width=self.input_width/8, input_channel=self.filters, num_con_unit=self.num_con_unit)
        self.con_block5 = Con_Block(filters=self.filters, input_width=self.input_width/16, input_channel=self.filters, num_con_unit=self.num_con_unit)
        self.con_block6 = Con_Block(filters=self.filters, input_width=self.input_width/32, input_channel=self.filters, num_con_unit=self.num_con_unit)
        self.con_block7 = Con_Block(filters=self.filters, input_width=self.input_width/64, input_channel=self.filters, num_con_unit=self.num_con_unit)

        self.con_up7 = Up_Block(filters=self.filters, input_width=self.input_width/64, input_channel=self.filters, num_con_unit=self.num_con_unit)
        self.con_up6 = Up_Block(filters=self.filters, input_width=self.input_width/32, input_channel=self.filters*2, num_con_unit=self.num_con_unit)
        self.con_up5 = Up_Block(filters=self.filters, input_width=self.input_width/16, input_channel=self.filters*2, num_con_unit=self.num_con_unit)
        self.con_up4 = Up_Block(filters=self.filters, input_width=self.input_width/8, input_channel=self.filters*2, num_con_unit=self.num_con_unit)
        self.con_up3 = Up_Block(filters=self.filters, input_width=self.input_width/4, input_channel=self.filters*2, num_con_unit=self.num_con_unit)
        self.con_up2 = Up_Block(filters=self.filters, input_width=self.input_width/2, input_channel=self.filters*2, num_con_unit=self.num_con_unit)
        self.con_up1 = Con_Block(filters=self.filters, input_width=self.input_width, input_channel=self.filters*2, num_con_unit=self.num_con_unit)

        self.con_end = Con_Unit(filters=self.num_class, init_input_shape=(self.input_width, self.input_width, self.filters), activation='softmax')

        self.pool = MaxPooling2D(padding='same')

    def call(self, inputs):
        con1 = self.con_block1(inputs)

        pool2 = self.pool(con1)
        con2 = self.con_block2(pool2)

        pool3 = self.pool(con2)
        con3 = self.con_block3(pool3)

        pool4 = self.pool(con3)
        con4 = self.con_block4(pool4)

        pool5 = self.pool(con4)
        con5 = self.con_block5(pool5)

        pool6 = self.pool(con5)
        con6 = self.con_block6(pool6)

        pool7 = self.pool(con6)
        con7 = self.con_block7(pool7)

        up7 = self.con_up7(con7)

        merge6 = concatenate([up7, con6], axis=3)
        up6 = self.con_up6(merge6)
        # up6 = self.con_up6(con6)

        merge5 = concatenate([up6, con5], axis=3)
        up5 = self.con_up5(merge5)

        merge4 = concatenate([up5, con4], axis=3)
        up4 = self.con_up4(merge4)

        merge3 = concatenate([up4, con3], axis=3)
        up3 = self.con_up3(merge3)

        merge2 = concatenate([up3, con2], axis=3)
        up2 = self.con_up2(merge2)

        merge1 = concatenate([up2, con1], axis=3)
        up1 = self.con_up1(merge1)

        out = self.con_end(up1)

        return out
