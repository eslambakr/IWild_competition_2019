# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 00:38:04 2019

@author: meltahan
"""

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import cv2
import numpy as np
import copy

image = cv2.imread('1_cam_0_frame_0_0.JPG', cv2.IMREAD_COLOR)

ia.seed(1)


def scale_transalate(image, bbs):
    translate_x = int(np.random.randint(30, 50, size=1, dtype='int8'))
    translate_y = int(np.random.randint(40, 70, size=1, dtype='int8'))
    seq = iaa.Sequential([
        iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect BBs
        iaa.Affine(
            translate_px={"x": translate_x, "y": translate_y},
            scale=(0.5, 0.7)
        )  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
    ])
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
    return image_aug, bbs_aug


def rotate(image, bbs, rotate):
    image_aug, bbs_aug = iaa.Affine(rotate=rotate)(image=image, bounding_boxes=bbs)
    return image_aug, bbs_aug


def noise_rotate(image, bbs, rotate):
    noise_scale = int(np.random.randint(5, 15, size=1, dtype='int8'))
    seq = iaa.Sequential([
        iaa.Affine(rotate=(-rotate, rotate)),
        iaa.AdditiveGaussianNoise(scale=(noise_scale, noise_scale)),
    ], random_order=True)
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
    return image_aug, bbs_aug


def sharp_noise(image, bbs):
    alpha = np.random.rand()*0.15
    image_aug, bbs_aug = iaa.Sequential([
        iaa.Add(10, per_channel=True),
        iaa.Sharpen(alpha=alpha)
    ])(image=image, bounding_boxes=bbs)
    return image_aug, bbs_aug


def some_of(image, bbs, rotate):
    alpha = np.random.rand()*0.15
    image_aug, bbs_aug = iaa.SomeOf(2, [
        iaa.Affine(rotate=rotate),
        iaa.AdditiveGaussianNoise(scale=0.01 * 255),
        iaa.Add(15, per_channel=True),
        iaa.Sharpen(alpha=alpha)
    ], random_order=True)(image=image, bounding_boxes=bbs)
    return image_aug, bbs_aug


def crop_pad(image, bbs):
    image_aug, bbs_aug = iaa.CropAndPad(percent=(-0.35, 0.35))(image=image, bounding_boxes=bbs)
    return image_aug, bbs_aug


def crop_add(image, bbs):
    pad = int(np.random.randint(32, 128, size=1, dtype='int8'))
    image_aug, bbs_aug = iaa.CropAndPad(
        percent=(0, 0.05),
        pad_mode=["constant", "edge"],
        pad_cval=(0, pad)
    )(image=image, bounding_boxes=bbs)
    return image_aug, bbs_aug


def flip(image, bbs):
    image_aug, bbs_aug = iaa.Fliplr(0.5)(image=image, bounding_boxes=bbs)
    return image_aug, bbs_aug


def translate(image, bbs, x, y):
    image_aug, bbs_aug = iaa.Affine(translate_px={"x": (-x, x), "y": (-y, y)})(image=image, bounding_boxes=bbs)
    return image_aug, bbs_aug


def shear(image, bbs, angle):
    image_aug, bbs_aug = iaa.Affine(shear=(-angle, angle))(image=image, bounding_boxes=bbs)
    return image_aug, bbs_aug


def super_pixel(image):
    n_segments = int(np.random.randint(5, 20, size=1, dtype='int8'))
    image_aug = iaa.Superpixels(p_replace=0.01, n_segments=n_segments)(image=image)
    return image_aug


def change_color_space(image):
    x1 = int(np.random.randint(20, 50, size=1, dtype='int8'))
    x2 = int(np.random.randint(60, 100, size=1, dtype='int8'))
    image_aug = iaa.Sequential([
        iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
        iaa.WithChannels(0, iaa.Add((x1, x2))),
        iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")
    ])(image=image)
    return image_aug


def grey(image):
    image_aug = iaa.Grayscale(alpha=(0.0, 1.0))(image=image)
    return image_aug


def emboss(image):
    image_aug = iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5))(image=image)
    return image_aug


def merge_edges(image):
    image_aug = iaa.EdgeDetect(alpha=(0.0, .5))(image=image)
    return image_aug


def add(image):
    x1 = int(np.random.randint(-40, -20, size=1, dtype='int8'))
    x2 = int(np.random.randint(30, 50, size=1, dtype='int8'))
    image_aug = iaa.Add((x1, x2), per_channel=0.5)(image=image)
    return image_aug


def dropout(image):
    alpha = np.random.rand() * 0.25
    image_aug = iaa.Dropout(p=(0, alpha))(image=image)
    return image_aug


def coarse_dropout(image):
    size_percent = np.random.rand() * 0.2
    image_aug = iaa.CoarseDropout(0.02, size_percent=size_percent, per_channel=0.5)(image=image)
    return image_aug


def invert(image):
    image_aug = iaa.Invert(0.25, per_channel=0.1)(image=image)
    return image_aug


def contrast_normalization(image):
    image_aug = iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)(image=image)
    return image_aug


"""
Numpy augmentation functions 
"""


def crop_img(img, box):
    crop_img = img[box[1] - box[3] * 3:box[1] + box[3] * 3, box[0] - box[2] * 3:box[0] + box[2] * 3, :]
    return crop_img


def brightness_distortion(img):
    # brightness distortion
    noise = np.random.randint(15, 40, size=(img.shape[0], img.shape[1], 3), dtype='int32')
    img_m = np.copy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(3):
                img_m[i][j][k] += noise[i][j][k]
    return img_m


def s_p_noise(image):
    # salt and pepper noise
    row, col, ch = image.shape
    s_vs_p = 0.5
    amount = 0.1
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    out[tuple(coords)] = 1

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    out[tuple(coords)] = 0
    return out


def r_g_distortion(image):
    out = np.copy(image)
    temp = out[:, :, 0]
    out[:, :, 0] = out[:, :, 1]
    out[:, :, 1] = temp
    return out


def add_blur(image):
    blur = cv2.GaussianBlur(image, (15, 15), 0)
    return blur


def add_median(image):
    median = cv2.medianBlur(image, 5)
    return median


"""
box = np.array([[688, 260, 30, 25],[634,255,45,65]])   

bbs = BoundingBoxesOnImage([
    BoundingBox(x1=box[0][0], x2=box[0][0]+box[0][2], y1=box[0][1], y2=box[0][1]+box[0][3]),
    BoundingBox(x1=box[1][0], x2=box[1][0]+box[1][2], y1=box[1][1], y2=box[1][1]+box[1][3]),
], shape=image.shape)
"""


# image_aug, bbs_aug = scale_transalate(image, bbs)
# image_aug, bbs_aug = rotate(image,bbs,7)
# image_aug, bbs_aug = noise_rotate(image,bbs,15)
# image_aug, bbs_aug =  sharp_noise(image,bbs)
# image_aug, bbs_aug =  some_of(image,bbs,0)
# image_aug, bbs_aug =  crop_pad(image,bbs)
# image_aug, bbs_aug =  crop_add(image,bbs)
# image_aug, bbs_aug = flip(image,bbs)
# image_aug, bbs_aug = translate(image,bbs,190,500)
# image_aug, bbs_aug = shear(image,bbs,25)


# no box change
# image_aug = super_pixel(image)
# image_aug = change_color_space(image)
# image_aug = grey(image)
# image_aug = emboss(image)
# image_aug = merge_edges(image)
# image_aug = add(image)
# image_aug = dropout(image)
# image_aug = coarse_dropout(image)
# image_aug = invert(image)
# image_aug = contrast_normalization(image)

# ia.imshow(bbs.draw_on_image(image, size=1))
# ia.imshow(bbs_aug.draw_on_image(image_aug, size=1))

def augmentation_handler_50(image, box):
    info = []
    element = {}

    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=box[0], x2=box[0] + box[2], y1=box[1], y2=box[1] + box[3]),
    ], shape=image.shape)

    for i in range(10):
        image_aug, bbs_aug = scale_transalate(image, bbs)
        element['image'] = image_aug
        box = bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y1, bbs_aug.bounding_boxes[0].x2 - \
              bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y2 - bbs_aug.bounding_boxes[0].y1
        element['box'] = box
        info.append(copy.deepcopy(element))

    image_aug, bbs_aug = rotate(image, bbs, int(np.random.randint(-45, 45, size=1, dtype='int8')))
    element['image'] = image_aug
    box = bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y1, bbs_aug.bounding_boxes[0].x2 - \
          bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y2 - bbs_aug.bounding_boxes[0].y1
    element['box'] = box
    info.append(copy.deepcopy(element))

    for i in range(5):
        angle = int(np.random.randint(-45, 45, size=1, dtype='int8'))
        angle_tuple = (angle)
        image_aug, bbs_aug = noise_rotate(image, bbs, angle_tuple)
        element['image'] = image_aug
        box = bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y1, bbs_aug.bounding_boxes[0].x2 - \
              bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y2 - bbs_aug.bounding_boxes[0].y1
        element['box'] = box
        info.append(copy.deepcopy(element))

    image_aug, bbs_aug = sharp_noise(image, bbs)
    element['image'] = image_aug
    box = bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y1, bbs_aug.bounding_boxes[0].x2 - \
          bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y2 - bbs_aug.bounding_boxes[0].y1
    element['box'] = box
    info.append(copy.deepcopy(element))

    for i in range(5):
        angle = int(np.random.randint(-25, 25, size=1, dtype='int8'))
        angle_tuple = (angle)
        image_aug, bbs_aug = some_of(image, bbs, angle_tuple)
        element['image'] = image_aug
        box = bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y1, bbs_aug.bounding_boxes[0].x2 - \
              bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y2 - bbs_aug.bounding_boxes[0].y1
        element['box'] = box
        info.append(copy.deepcopy(element))

    image_aug, bbs_aug = crop_pad(image, bbs)
    element['image'] = image_aug
    box = bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y1, bbs_aug.bounding_boxes[0].x2 - \
          bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y2 - bbs_aug.bounding_boxes[0].y1
    element['box'] = box
    info.append(copy.deepcopy(element))

    for i in range(3):
        image_aug, bbs_aug = crop_add(image, bbs)
        element['image'] = image_aug
        box = bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y1, bbs_aug.bounding_boxes[0].x2 - \
              bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y2 - bbs_aug.bounding_boxes[0].y1
        element['box'] = box
        info.append(copy.deepcopy(element))

    image_aug, bbs_aug = flip(image, bbs)
    element['image'] = image_aug
    box = bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y1, bbs_aug.bounding_boxes[0].x2 - \
          bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y2 - bbs_aug.bounding_boxes[0].y1
    element['box'] = box
    info.append(copy.deepcopy(element))

    for i in range(5):
        x = int(np.random.rand() * 300)
        x_tuple = (x)
        y = int(np.random.rand() * 300)
        y_tuple = (y)
        image_aug, bbs_aug = translate(image, bbs, x_tuple, y_tuple)
        element['image'] = image_aug
        box = bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y1, bbs_aug.bounding_boxes[0].x2 - \
              bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y2 - bbs_aug.bounding_boxes[0].y1
        element['box'] = box
        info.append(copy.deepcopy(element))

    for i in range(3):
        angle = int(np.random.randint(-25, 30, size=1, dtype='int8'))
        angle_tuple = (angle)
        image_aug, bbs_aug = shear(image, bbs, angle_tuple)
        element['image'] = image_aug
        box = bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y1, bbs_aug.bounding_boxes[0].x2 - \
              bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y2 - bbs_aug.bounding_boxes[0].y1
        element['box'] = box
        info.append(copy.deepcopy(element))

    image_aug = super_pixel(image)
    element['image'] = image_aug
    element['box'] = box
    info.append(copy.deepcopy(element))

    image_aug = change_color_space(image)
    element['image'] = image_aug
    element['box'] = box
    info.append(copy.deepcopy(element))

    image_aug = grey(image)
    element['image'] = image_aug
    element['box'] = box
    info.append(copy.deepcopy(element))

    image_aug = emboss(image)
    element['image'] = image_aug
    element['box'] = box
    info.append(copy.deepcopy(element))

    image_aug = merge_edges(image)
    element['image'] = image_aug
    element['box'] = box
    info.append(copy.deepcopy(element))

    image_aug = add(image)
    element['image'] = image_aug
    element['box'] = box
    info.append(copy.deepcopy(element))

    image_aug = dropout(image)
    element['image'] = image_aug
    element['box'] = box
    info.append(copy.deepcopy(element))

    image_aug = coarse_dropout(image)
    element['image'] = image_aug
    element['box'] = box
    info.append(copy.deepcopy(element))

    image_aug = invert(image)
    element['image'] = image_aug
    element['box'] = box
    info.append(copy.deepcopy(element))

    image_aug = contrast_normalization(image)
    element['image'] = image_aug
    element['box'] = box
    info.append(copy.deepcopy(element))

    image_aug = brightness_distortion(image)
    element['image'] = image_aug
    element['box'] = box
    info.append(copy.deepcopy(element))

    image_aug = s_p_noise(image)
    element['image'] = image_aug
    element['box'] = box
    info.append(copy.deepcopy(element))

    image_aug = r_g_distortion(image)
    element['image'] = image_aug
    element['box'] = box
    info.append(copy.deepcopy(element))

    image_aug = add_blur(image)
    element['image'] = image_aug
    element['box'] = box
    info.append(copy.deepcopy(element))

    image_aug = add_median(image)
    element['image'] = image_aug
    element['box'] = box
    info.append(copy.deepcopy(element))

    return info


def augmentation_handler_10(image, box):
    info = []
    element = {}

    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=box[0], x2=box[0] + box[2], y1=box[1], y2=box[1] + box[3]),
    ], shape=image.shape)

    for i in range(3):
        image_aug, bbs_aug = scale_transalate(image, bbs)
        image_aug = add_median(image_aug)
        element['image'] = image_aug
        box = bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y1, bbs_aug.bounding_boxes[0].x2 - \
              bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y2 - bbs_aug.bounding_boxes[0].y1
        element['box'] = box
        info.append(copy.deepcopy(element))

    for i in range(2):
        angle = int(np.random.randint(-45, 45, size=1, dtype='int8'))
        angle_tuple = (angle)
        image_aug, bbs_aug = noise_rotate(image, bbs, angle_tuple)
        element['image'] = image_aug
        box = bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y1, bbs_aug.bounding_boxes[0].x2 - \
              bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y2 - bbs_aug.bounding_boxes[0].y1
        element['box'] = box
        info.append(copy.deepcopy(element))

    for i in range(1):
        angle = int(np.random.randint(-25, 25, size=1, dtype='int8'))
        angle_tuple = (angle)
        image_aug, bbs_aug = some_of(image, bbs, angle_tuple)
        element['image'] = image_aug
        box = bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y1, bbs_aug.bounding_boxes[0].x2 - \
              bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y2 - bbs_aug.bounding_boxes[0].y1
        element['box'] = box
        info.append(copy.deepcopy(element))

    image_aug, bbs_aug = crop_pad(image, bbs)
    image_aug = s_p_noise(image_aug)
    element['image'] = image_aug
    box = bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y1, bbs_aug.bounding_boxes[0].x2 - \
          bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y2 - bbs_aug.bounding_boxes[0].y1
    element['box'] = box
    info.append(copy.deepcopy(element))

    for i in range(1):
        image_aug, bbs_aug = crop_add(image, bbs)
        element['image'] = image_aug
        box = bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y1, bbs_aug.bounding_boxes[0].x2 - \
              bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y2 - bbs_aug.bounding_boxes[0].y1
        element['box'] = box
        info.append(copy.deepcopy(element))

    image_aug, bbs_aug = flip(image, bbs)
    element['image'] = image_aug
    box = bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y1, bbs_aug.bounding_boxes[0].x2 - \
          bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y2 - bbs_aug.bounding_boxes[0].y1
    element['box'] = box
    info.append(copy.deepcopy(element))

    for i in range(1):
        angle = int(np.random.randint(-25, 30, size=1, dtype='int8'))
        angle_tuple = (angle)
        image_aug, bbs_aug = shear(image, bbs, angle_tuple)
        image_aug = add_blur(image_aug)
        element['image'] = image_aug
        box = bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y1, bbs_aug.bounding_boxes[0].x2 - \
              bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y2 - bbs_aug.bounding_boxes[0].y1
        element['box'] = box
        info.append(copy.deepcopy(element))

    return info


def augmentation_handler_5(image, box):
    info = []
    element = {}

    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=box[0], x2=box[0] + box[2], y1=box[1], y2=box[1] + box[3]),
    ], shape=image.shape)

    for i in range(1):
        image_aug, bbs_aug = scale_transalate(image, bbs)
        image_aug = add_median(image_aug)
        element['image'] = image_aug
        box = bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y1, bbs_aug.bounding_boxes[0].x2 - \
              bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y2 - bbs_aug.bounding_boxes[0].y1
        element['box'] = box
        info.append(copy.deepcopy(element))

    for i in range(1):
        angle = int(np.random.randint(-45, 45, size=1, dtype='int8'))
        angle_tuple = (angle)
        image_aug, bbs_aug = noise_rotate(image, bbs, angle_tuple)
        element['image'] = image_aug
        box = bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y1, bbs_aug.bounding_boxes[0].x2 - \
              bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y2 - bbs_aug.bounding_boxes[0].y1
        element['box'] = box
        info.append(copy.deepcopy(element))

    for i in range(1):
        angle = int(np.random.randint(-25, 25, size=1, dtype='int8'))
        angle_tuple = (angle)
        image_aug, bbs_aug = some_of(image, bbs, angle_tuple)
        element['image'] = image_aug
        box = bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y1, bbs_aug.bounding_boxes[0].x2 - \
              bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y2 - bbs_aug.bounding_boxes[0].y1
        element['box'] = box
        info.append(copy.deepcopy(element))

    image_aug, bbs_aug = crop_pad(image, bbs)
    image_aug = s_p_noise(image_aug)
    element['image'] = image_aug
    box = bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y1, bbs_aug.bounding_boxes[0].x2 - \
          bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y2 - bbs_aug.bounding_boxes[0].y1
    element['box'] = box
    info.append(copy.deepcopy(element))

    for i in range(1):
        image_aug, bbs_aug = crop_add(image, bbs)
        element['image'] = image_aug
        box = bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y1, bbs_aug.bounding_boxes[0].x2 - \
              bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y2 - bbs_aug.bounding_boxes[0].y1
        element['box'] = box
        info.append(copy.deepcopy(element))

    return info
"""
box = np.array([688, 260, 30, 25])
augmented_imgs = augmentation_handler(image, box)

# ia.imshow(bbs.draw_on_image(image, size=1))
# ia.imshow(bbs_aug.draw_on_image(image_aug, size=1))

# cv2.imshow('image_aug',image_aug)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
