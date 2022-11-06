import os
import random
import time

import elasticdeform
import numpy
from PIL import Image, ImageChops
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import cv2 as cv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def my_cnn_model():
    image_size = 59
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


def my_cnn_train(model, data):
    (train_digits, train_labels), (test_digits, test_labels) = data

    # Get image size
    image_size = 59
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
    model.fit(train_data, train_labels_cat, epochs=3, batch_size=64,
              validation_data=(val_data, val_labels_cat))

    print("Done, dT:", time.time() - t_start)

    return model


def cnn_digits_predict(model, image_file):
    image_size = 59
    img = keras.preprocessing.image.load_img(image_file,
                                             target_size=(image_size, image_size), color_mode='grayscale')
    img_arr = np.expand_dims(img, axis=0)
    img_arr = 1 - img_arr / 255.0
    img_arr = img_arr.reshape((1, 59, 59, 1))

    result = model.predict([img_arr])
    return result[0]


def learn():
    size = 1000
    digits = np.zeros(shape=(size, 59, 59))
    labels = np.zeros(shape=(size,))
    for i in range(size):
        img, num = generate_data()
        digits[i] = img
        labels[i] = num
    split_data(digits, labels)
    model = my_cnn_model()
    model = my_cnn_train(model, split_data(digits, labels))
    model.save('cnn_digits_59x59.h5')


def predict(img):
    model = tf.keras.models.load_model('cnn_digits_59x59.h5')
    prediction = cnn_digits_predict(model, img)
    print(prediction)
    m = 0
    i = 0
    for v in prediction:
        if prediction[m] < v:
            m = i
        i += 1
    print(m)
    return m


def convert(img_path):
    image = Image.open(img_path)
    image = image.convert('L')
    img = pil_to_cv(image)
    return img


def erode(img):
    kernel = np.ones((3, 3), np.uint8)
    return cv.erode(img, kernel, iterations=1)


def dilate(img):
    kernel = np.ones((2, 2), np.uint8)
    return cv.dilate(img, kernel, iterations=1)


def dilrode(img):
    img_dilation = erode(img)
    img_erosion = dilate(img_dilation)
    return img_erosion


def prepare(img_path):
    img = convert(img_path)
    img = dilrode(img)
    img = cv_to_pil(img)
    img = crop(img)
    img = pil_to_cv(img)
    return img


def crop(img):
    image_cropped = trim(img)
    return image_cropped


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
    num = random.randint(0, 9)
    crop_img = img[0:44, num * 23 + num:num * 23 + num + 23].copy()
    size = crop_img.shape[:2]
    max_dim = max(size) + 15
    image_out = resize(crop_img, max_dim)

    return image_out, num


def distortion(img):
    return elasticdeform.deform_random_grid(img, sigma=5, order=0, points=2, prefilter=False, mode='constant', cval=255)


def make_holes(img):
    for x in range(np.shape(img)[0]):
        for y in range(np.shape(img)[1]):
            img[x][y] = 255 if random.randint(0, 10) >= 8 else img[x][y]
    return img


def cv_to_pil(img):
    return Image.fromarray(img)


def pil_to_cv(img):
    return np.asarray(img)


def generate_data():
    img, num = random_number()
    img = distortion(img)
    img = make_holes(img)
    img = dilrode(img)
    return img, num


def resize(img, p):
    size = img.shape[:2]
    if max(size) > p:
        h = max(size) - p
        half1 = h/2 if h % 2 == 1 else h/2+1
        half2 = h/2
        image_out = img[half1:max(size)-half2, half1:max(size)-half2]
        return image_out
    else:
        max_dim = p
        delta_w = max_dim - size[1]
        delta_h = max_dim - size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        color = [255, 255, 255]
        image_out = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)
        return image_out


def split_data(digits, labels):
    overall = 1000
    size = 300
    r = random.randint(0, size)
    m = overall - (size - r)
    train_digits = digits[r:m]
    train_labels = labels[r:m]
    test_digits = np.zeros(shape=(0, 59, 59))
    test_labels = np.zeros(shape=(0, ))
    test_digits = np.append(test_digits, digits[0:r], axis=0)
    test_labels = np.append(test_labels, labels[0:r], axis=0)
    test_digits = np.append(test_digits, digits[m:overall], axis=0)
    test_labels = np.append(test_labels, labels[m:overall], axis=0)
    return (train_digits, train_labels), (test_digits, test_labels)


def split_input(img):
    size = int(img.shape[1]/6)
    print(img.shape)
    tr = img.transpose()
    print(tr.shape)
    digits = np.zeros(shape=(6, img.shape[0], size))
    for i in range(6):
        digits[i] = tr[size*i:size*(i+1)].transpose()
    return digits


if __name__ == '__main__':

    # img = prepare('captcha4.gif')
    # arr = split_input(img)
    # predicted = numpy.array([])
    # for i in range(6):
    #     img_path = 'digit_'+str(i)+'.png'
    #     cv.imwrite(img_path, arr[i])
    #     predicted = np.append(predicted, predict(img_path))
    # print(predicted)
