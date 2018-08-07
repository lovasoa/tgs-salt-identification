from pathlib import Path

import numpy as np
from keras import Input, Model
from keras.callbacks import ModelCheckpoint, ProgbarLogger
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization, UpSampling2D, Reshape, \
    Activation
from keras.utils import Sequence
from scipy.misc import imread
from skimage.transform import resize


class SaltImageSequence(Sequence):

    def __init__(self, img_root: Path, batch_size: int = 10):
        self.batch_size = batch_size
        self.img_root = img_root
        self.ids = [f.stem for f in (img_root / "images").glob("*png")]
        assert len(self.ids) > 0, f"The images folder in {img_root} is empty"

    def __len__(self):
        return int(np.ceil(len(self.ids) / float(self.batch_size)))

    def img(self, folder: str, img_id: str) -> np.ndarray:
        return imread(self.img_root / folder / f"{img_id}.png", mode='L')  # 2d array

    def read_image(self, img_id: str) -> np.ndarray:
        img = self.img("images", img_id)  # 2d array
        img_with_channels = np.expand_dims(img, axis=0)  # 3d array
        return img_with_channels

    def read_mask(self, img_id: str) -> np.ndarray:
        img = self.img("images", img_id)  # 2d array
        img_resized = resize(img, (96, 96), mode='reflect', anti_aliasing=False)
        flat = img_resized.flatten().astype('float16') / 256
        return flat

    def __getitem__(self, idx):
        ids = self.ids[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.array([self.read_image(i) for i in ids])
        batch_y = np.array([self.read_mask(i) for i in ids])
        return batch_x, batch_y


def VGGSegnet(vgg_level=3):
    """Create a VGG segnet keras model.
    Code from https://github.com/divamgupta/image-segmentation-keras/
    """
    input_height = 101
    input_width = 101
    img_input = Input(shape=(1, input_height, input_width))

    x = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        input_shape=(1, input_height, input_width),
        data_format='channels_first',
        activation='relu',
        padding='same',
        name='block1_conv1'
    )(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format='channels_first')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format='channels_first')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format='channels_first')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format='channels_first')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format='channels_first')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format='channels_first')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format='channels_first')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format='channels_first')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format='channels_first')(x)

    o = x

    o = (ZeroPadding2D((1, 1), data_format='channels_first'))(o)
    o = (Conv2D(512, (3, 3), padding='valid', data_format='channels_first'))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format='channels_first'))(o)
    o = (ZeroPadding2D((1, 1), data_format='channels_first'))(o)
    o = (Conv2D(256, (3, 3), padding='valid', data_format='channels_first'))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format='channels_first'))(o)
    o = (ZeroPadding2D((1, 1), data_format='channels_first'))(o)
    o = (Conv2D(128, (3, 3), padding='valid', data_format='channels_first'))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format='channels_first'))(o)
    o = (ZeroPadding2D((1, 1), data_format='channels_first'))(o)
    o = (Conv2D(64, (3, 3), padding='valid', data_format='channels_first'))(o)
    o = (BatchNormalization())(o)

    o = Conv2D(1, (3, 3), padding='same', data_format='channels_first')(o)

    (batch_size, channels, output_height, output_width) = Model(img_input, o).output_shape

    o = (Reshape((output_height * output_width,)))(o)
    o = (Activation('softmax'))(o)
    return Model(img_input, o)


img_root = Path(__file__).resolve().parents[2] / "tgs_data" / "train"
imageSeq = SaltImageSequence(img_root=img_root)

model = VGGSegnet()
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit_generator(
    generator=imageSeq,
    callbacks=[ModelCheckpoint(filepath='weights'), ProgbarLogger(count_mode='steps')]
)
