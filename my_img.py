#  Copyright (c) 2024 by Tomoya Konishi
#  All rights reserved.
#
#  License:
#  This program is permitted under the principle of "NO WARRANTY" and
#  "NO RESPONSIBILITY". The author shall not be liable for any event
#  arising in any way out of the use of these resources.
#
#  Redistribution in source and binary forms, with or without
#  modification, is also permitted provided that the above copyright
#  notice, disclaimer and this condition are retained.
#
#
#
import numpy as np
import cv2

def is_binary_image(image):
    unique_values = np.unique(image)
    return len(unique_values) == 2 and set(unique_values).issubset({0, 255})


def is_gray_image(image):
    img_blue, img_green, img_red = cv2.split(image)
    return (img_blue==img_green).all() & (img_green==img_red).all()


def is_color_image(image):
    return len(image.shape) == 3 and image.shape[2] == 3


def remove_lines(image):
    """
    画像から線状の領域を削除します。

    Args:
        image (numpy.ndarray): 入力画像。

    Returns:
        numpy.ndarray: 線状の領域が削除された画像。
    """
    img = image.copy()

    vp = np.sum((img != 0).astype(np.uint8), axis=0)
    loc_x_spike = np.where(vp > np.max(vp[:])*0.9)[0]
    for x in loc_x_spike:
        line_color = (0, 0, 0)
        cv2.line(img, (x, 0), (x, img.shape[0]), line_color, 3)

    hp = np.sum((img != 0).astype(np.uint8), axis=1)
    loc_y_spike = np.where(hp > np.max(hp[:])*0.9)[0]
    for y in loc_y_spike:
        line_color = (0, 0, 0)
        cv2.line(img, (0, y), (img.shape[1], y), line_color, 3)

    return img


def count_pixels_in_rect(image, rect):
    mask = np.zeros(image.shape, dtype="uint8")
    cv2.rectangle(mask, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), 255, -1)
    mask = cv2.bitwise_and(image, image, mask=mask)
    return cv2.countNonZero(mask)


def extract_monotone(image):
    """
    画像からモノクロ領域を抽出し、他の領域を白色に置き換えます（着色領域の除去）。

    Args:
        image (numpy.ndarray): 入力画像。

    Returns:
        numpy.ndarray: 単色領域が抽出され、他の領域が白色に置き換えられた画像。
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white_image = np.ones_like(image) * 255

    # Define the range for the black color in HSV
    lower_mono = np.array([0, 0, 0])
    upper_mono = np.array([255, 50, 255])

    # Create masks using the defined ranges
    mask_mono = cv2.inRange(hsv_image, lower_mono, upper_mono)
    mask_mono = cv2.medianBlur(mask_mono, 3)
    inv_mask_mono = cv2.bitwise_not(mask_mono)

    # Apply the mask to the original image
    unchanged_img = cv2.bitwise_and(image, image, mask=mask_mono)
    white_img = cv2.bitwise_and(white_image, white_image, mask=inv_mask_mono)
    return cv2.add(unchanged_img, white_img)


def split_image_vertical_and_choose_greatest(image):
    """
    画像を縦線で垂直方向に分割し、最も大きな領域を選択します。

    Args:
        image (numpy.ndarray): 入力画像。

    Returns:
        numpy.ndarray: 最も大きな領域を含む画像の部分。
    """
    vp = np.sum((image != 0).astype(np.uint8), axis=0)
    loc_x_spike = np.where(vp == np.max(vp[:]))[0]
    diff_loc_x_spike = np.diff(loc_x_spike)
    idx = np.where(diff_loc_x_spike == max(diff_loc_x_spike))[0][0]
    return image[:, loc_x_spike[idx]:loc_x_spike[idx+1]]