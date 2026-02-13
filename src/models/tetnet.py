import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, Conv3D, BatchNormalization, Activation, Add,
                                     Dropout, GlobalAveragePooling2D, GlobalAveragePooling3D,
                                     Concatenate, Dense)
from tensorflow.keras.models import Model
from .cbam import cbam_block

def _cls_branch(name, windowSize_2D, bands, ks):
    inp = Input((windowSize_2D, windowSize_2D, 1), name=f"{name}_in")
    x1 = Conv2D(bands, kernel_size=(1,1), padding='same')(inp)
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)

    x2 = Conv2D(bands, kernel_size=(ks,ks), padding='same')(x1)
    x2 = BatchNormalization()(x2)
    x2 = Activation("relu")(x2)

    x3 = Conv2D(bands, kernel_size=(ks,ks), padding='same')(x2)
    x3 = BatchNormalization()(x3)
    x3 = Activation("relu")(x3)

    add = Add()([x1, x3])
    add = Conv2D(16, kernel_size=(ks,ks), padding='valid', activation='relu')(add)
    add = Dropout(0.5)(add)
    feat = GlobalAveragePooling2D()(add)
    return inp, feat

def build_tetnet(windowSize_2D=15, windowSize_3D=15, bands=16, dbands=16):
    ks = [5,7,5,3,5,7,7,7,7,7,3,5,3,3,7,5]
    inputs=[]; feats=[]
    for i,k in enumerate(ks, start=1):
        inp, feat = _cls_branch(f"cls{i}", windowSize_2D, bands, k)
        inputs.append(inp); feats.append(feat)

    # 3D branch
    in_3D = Input((windowSize_3D, windowSize_3D, dbands, 1), name="in_3D")
    x = Conv3D(64, kernel_size=(3,3,3), padding='valid')(in_3D)
    x = BatchNormalization()(x); x=Activation("relu")(x)

    x = Conv3D(32, kernel_size=(3,3,7), padding='valid')(x)
    x = BatchNormalization()(x); x=Activation("relu")(x)

    x = Conv3D(16, kernel_size=(3,3,5), padding='valid')(x)
    x = BatchNormalization()(x); x=Activation("relu")(x)

    x = Conv3D(8, kernel_size=(3,3,3), padding='valid')(x)
    x = BatchNormalization()(x); x=Activation("relu")(x)

    flatten3d = GlobalAveragePooling3D()(x)

    # cbam2D branch (2D+attention)
    in_cbam2D = Input((windowSize_2D, windowSize_2D, bands), name="in_cbam2D")
    c1 = Conv2D(64, kernel_size=(3,3), padding='same')(in_cbam2D)
    c1 = BatchNormalization()(c1); c1 = Activation("relu")(c1)
    c2 = Conv2D(64, kernel_size=(1,1), padding='same')(c1)
    c2 = BatchNormalization()(c2); c2 = Activation("relu")(c2)
    cb1 = cbam_block(c1)
    a1 = Add()([cb1, c2])

    c3 = Conv2D(32, kernel_size=(3,3), padding='same')(a1)
    c3 = BatchNormalization()(c3); c3 = Activation("relu")(c3)
    c4 = Conv2D(32, kernel_size=(1,1), padding='same')(c3)
    c4 = BatchNormalization()(c4); c4 = Activation("relu")(c4)
    cb2 = cbam_block(c3)
    a2 = Add()([cb2, c4])

    c5 = Conv2D(16, kernel_size=(3,3), padding='same')(a2)
    c5 = BatchNormalization()(c5); c5 = Activation("relu")(c5)
    c6 = Conv2D(16, kernel_size=(1,1), padding='same')(c5)
    c6 = BatchNormalization()(c6); c6 = Activation("relu")(c6)
    cb3 = cbam_block(c5)
    a3 = Add()([cb3, c6])

    c7 = Conv2D(8, kernel_size=(3,3), padding='same')(a3)
    c7 = BatchNormalization()(c7); c7 = Activation("relu")(c7)
    flatten2d2 = GlobalAveragePooling2D()(c7)

    # classifier head
    out = Concatenate()(feats + [flatten3d, flatten2d2])
    out = Dense(256, activation='relu', use_bias=True)(out)
    out = Dropout(0.25)(out)
    out = Dense(128, activation='relu', use_bias=True)(out)
    out = Dropout(0.25)(out)
    out = Dense(bands, activation='softmax', use_bias=True)(out)

    model = Model(inputs=inputs + [in_3D, in_cbam2D], outputs=[out], name="TETNet_SA")
    return model
