# This file includes basic functionality for image processing
# including i/o handling, image augmentation and model input pre-processing
# Author: Stefan Kahl, 2018, Chemnitz University of Technology

import sys

sys.path.append("..")

import copy

import numpy as np
import cv2


###################### AUGMENTATION ######################
def augment(img, augmentation={}, count=3, probability=0.5):
    # Make working copy
    augmentations = copy.deepcopy(augmentation)

    # Choose number of augmentations according to count
    # Count = 3 means either 0, 1, 2 or 3 different augmentations
    while (count > 0 and len(augmentations) > 0):

        # Roll the dice if we do augment or not
        if RANDOM.choice([True, False], p=[probability, 1 - probability]):

            # Choose one method
            aug = RANDOM.choice(augmentations.keys())

            # Call augementation methods
            if aug == 'flip':
                img = flip(img, augmentations[aug])
            elif aug == 'rotate':
                img = rotate(img, augmentations[aug])
            elif aug == 'zoom':
                img = zoom(img, augmentations[aug])
            elif aug == 'crop':
                if isinstance(augmentations[aug], float):
                    img = crop(img, top=augmentations[aug], left=augmentations[aug], right=augmentations[aug],
                               bottom=augmentations[aug])
                else:
                    img = crop(img, top=augmentations[aug][0], left=augmentations[aug][1], bottom=augmentations[aug][2],
                               right=augmentations[aug][3])
            elif aug == 'roll':
                img = roll(img, vertical=augmentations[aug], horizontal=augmentations[aug])
            elif aug == 'roll_v':
                img = roll(img, vertical=augmentations[aug], horizontal=0)
            elif aug == 'roll_h':
                img = roll(img, vertical=0, horizontal=augmentations[aug])
            elif aug == 'mean':
                img = mean(img, augmentations[aug])
            elif aug == 'noise':
                img = noise(img, augmentations[aug])
            elif aug == 'dropout':
                img = dropout(img, augmentations[aug])
            elif aug == 'blackout':
                img = blackout(img, augmentations[aug])
            elif aug == 'blur':
                img = blur(img, augmentations[aug])
            elif aug == 'brightness':
                img = brightness(img, augmentations[aug])
            elif aug == 'multiply':
                img = randomMultiply(img, augmentations[aug])
            elif aug == 'hue':
                img = hue(img, augmentations[aug])
            elif aug == 'lightness':
                img = lightness(img, augmentations[aug])
            elif aug == 'add':
                img = add(img, augmentations[aug])
            else:
                pass

            # Remove key so we avoid duplicate augmentations
            del augmentations[aug]

        # Count (even if we did not augment)
        count -= 1

    return img


def flip(img, flip_axis=1):
    return cv2.flip(img, flip_axis)


def rotate(img, angle, zoom=1.0):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), RANDOM.uniform(-angle, angle), zoom)

    return cv2.warpAffine(img, M, (w, h))


def zoom(img, amount=0.33):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1 + RANDOM.uniform(0, amount))

    return cv2.warpAffine(img, M, (w, h))


def crop(img, top=0.1, left=0.1, bottom=0.1, right=0.1):
    h, w = img.shape[:2]

    t_crop = max(1, int(h * RANDOM.uniform(0, top)))
    l_crop = max(1, int(w * RANDOM.uniform(0, left)))
    b_crop = max(1, int(h * RANDOM.uniform(0, bottom)))
    r_crop = max(1, int(w * RANDOM.uniform(0, right)))

    img = img[t_crop:-b_crop, l_crop:-r_crop]
    img = squeeze(img, w, h)

    return img


def roll(img, vertical=0.1, horizontal=0.1):
    # Vertical Roll
    img = np.roll(img, int(img.shape[0] * RANDOM.uniform(-vertical, vertical)), axis=0)

    # Horizontal Roll
    img = np.roll(img, int(img.shape[1] * RANDOM.uniform(-horizontal, horizontal)), axis=1)

    return img


def mean(img, normalize=True):
    img = substractMean(img, True)

    if normalize and not img.max() == 0:
        img /= img.max()

    return img


def noise(img, amount=0.05):
    img += RANDOM.normal(0.0, RANDOM.uniform(0, amount ** 0.5), img.shape)
    img = np.clip(img, 0.0, 1.0)

    return img


def dropout(img, amount=0.25):
    d = RANDOM.uniform(0, 1, img.shape)
    d[d <= amount] = 0
    d[d > 0] = 1

    return img * d


def blackout(img, amount=0.25):
    b_width = int(img.shape[1] * amount)
    b_start = RANDOM.randint(0, img.shape[1] - b_width)

    img[:, b_start:b_start + b_width] = 0

    return img


def blur(img, kernel_size=3):
    return cv2.blur(img, (kernel_size, kernel_size))


def brightness(img, amount=0.25):
    img *= RANDOM.uniform(1 - amount, 1 + amount)
    img = np.clip(img, 0.0, 1.0)

    return img


def randomMultiply(img, amount=0.25):
    img *= RANDOM.uniform(1 - amount, 1 + amount, size=img.shape)
    img = np.clip(img, 0.0, 1.0)

    return img


def hue(img, amount=0.1):
    try:
        # Only works with BGR images
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] *= RANDOM.uniform(1 - amount, 1 + amount)
        hsv[:, :, 0].clip(0, 360)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    except:
        pass

    return img


def lightness(img, amount=0.25):
    try:
        # Only works with BGR images
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] *= RANDOM.uniform(1 - amount, 1 + amount)
        lab[:, :, 0].clip(0, 255)
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    except:
        pass

    return img


def add(img, items):
    # Choose one item from List
    index = RANDOM.randint(len(items))

    # Open and resize image
    img2 = openImage(items[index], cfg.IM_DIM)
    img2 = resize(img2, img.shape[1], img.shape[0])

    # Generate random weights
    w1 = RANDOM.uniform(1, 2)
    w2 = RANDOM.uniform(1, 2)

    # Add images and calculate average
    img = (img * w1 + img2 * w2) / (w1 + w2)

    return img


if __name__ == '__main__':
    im_path = '../example/Acadian Flycatcher.png'

    img = openImage(im_path, 1)
    img = resize(img, 256, 256, mode='fill')

    showImage(img)
    img = augment(img, {'flip': 1}, 3)
