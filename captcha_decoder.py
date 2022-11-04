import os
import random
import time

from PIL import Image, ImageChops
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import cv2 as cv
from wand.image import Image as Im

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def mnist_cnn_model():
    image_size = 28
    num_channels = 1  # 1 for grayscale images
    num_classes = 10  # Number of outputs
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                     padding='same', input_shape=(image_size, image_size, num_channels)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    # Densely connected layers
    model.add(Dense(128, activation='relu'))
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def mnist_cnn_train(model):
    (train_digits, train_labels), (test_digits, test_labels) = keras.datasets.mnist.load_data()

    # Get image size
    image_size = 28
    num_channels = 1  # 1 for grayscale images

    # re-shape and re-scale the images data
    train_data = np.reshape(train_digits, (train_digits.shape[0], image_size, image_size, num_channels))
    train_data = train_data.astype('float32') / 255.0
    # encode the labels - we have 10 output classes
    # 3 -> [0 0 0 1 0 0 0 0 0 0], 5 -> [0 0 0 0 0 1 0 0 0 0]
    num_classes = 10
    train_labels_cat = keras.utils.to_categorical(train_labels, num_classes)

    # re-shape and re-scale the images validation data
    val_data = np.reshape(test_digits, (test_digits.shape[0], image_size, image_size, num_channels))
    val_data = val_data.astype('float32') / 255.0
    # encode the labels - we have 10 output classes
    val_labels_cat = keras.utils.to_categorical(test_labels, num_classes)

    print("Training the network...")
    t_start = time.time()

    # Start training the network
    model.fit(train_data, train_labels_cat, epochs=8, batch_size=64,
              validation_data=(val_data, val_labels_cat))

    print("Done, dT:", time.time() - t_start)

    return model


def cnn_digits_predict(model, image_file):
    image_size = 28
    img = keras.preprocessing.image.load_img(image_file,
                                             target_size=(image_size, image_size), color_mode='grayscale')
    img_arr = np.expand_dims(img, axis=0)
    img_arr = 1 - img_arr / 255.0
    img_arr = img_arr.reshape((1, 28, 28, 1))

    result = model.predict([img_arr])
    return result[0]


def learn():
    model = mnist_cnn_model()
    mnist_cnn_train(model)
    model.save('cnn_digits_28x28.h5')


def predict():
    model = tf.keras.models.load_model('cnn_digits_28x28.h5')
    prediction = cnn_digits_predict(model, 'digit_1.png')
    print(prediction)
    m = 0
    i = 0
    for v in prediction:
        if prediction[m] < v:
            m = i
        i += 1
    print(m)


def convert():
    image = Image.open('captcha.gif')
    image = image.convert('1')
    image.save("captcha.png")


def dilrode():
    img = cv.imread('captcha.png', 0)
    kernel = np.ones((3, 3), np.uint8)
    kernel_er = np.ones((2, 2), np.uint8)
    img_dilation = cv.erode(img, kernel, iterations=1)
    img_erosion = cv.dilate(img_dilation, kernel_er, iterations=1)
    cv.imwrite('captcha_eroded.png', img_erosion)


def prepare():
    convert()
    dilrode()
    crop()


def crop():
    img = Image.open('captcha_eroded.png')
    image_cropped = trim(img)
    image_cropped.save('captcha_cropped.png')


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    else:
        # Failed to find the borders, convert to "RGB"
        return trim(im.convert('RGB'))


def random_number():
    img = cv.imread('letterus.png', cv.IMREAD_GRAYSCALE)
    # num = random.randint(0, 9)
    num = 2
    crop_img = img[0:44, num * 23 + num:num * 23 + num + 23].copy()
    size = crop_img.shape[:2]
    max_dim = max(size) + 18
    delta_w = max_dim - size[1]
    delta_h = max_dim - size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [255, 255, 255]
    image_out = cv.copyMakeBorder(crop_img, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)
    cv.imwrite('number.png', image_out)
    # cv.imwrite('number.png', crop_img)


def distortion(img):
    img = cv.imread('number.png', cv.IMREAD_GRAYSCALE)

    A = img.shape[0] / 3.0
    w = 1.0 / img.shape[1]
    r1 = random.randint(1, 4) * 0.2
    r2 = random.randint(1, 4) * 0.2
    r3 = random.randint(1, 4) * 0.2
    r4 = random.randint(1, 4) * 0.2

    shift = lambda x: A * np.sin(0.1 * np.pi * x * w)

    for i in range(img.shape[0]):
        img[:, i] = np.roll(img[:, i], int(shift(i) * 0.5/r1))
    for i in range(img.shape[0]):
        img[i, :] = np.roll(img[i, :], int(shift(i) * 0.5/r2))
    for i in range(img.shape[0]):
        img[:, i] = np.roll(img[:, i], int(shift(i) * -0.5/r3))
    for i in range(img.shape[0]):
        img[i, :] = np.roll(img[i, :], int(shift(i) * -0.5/r4))
    cv.imwrite('number_distorted.png', img)


if __name__ == '__main__':
    random_number()
    distortion()
