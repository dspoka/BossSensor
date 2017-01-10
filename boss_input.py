# -*- coding: utf-8 -*-
import os

import numpy as np
import cv2

from os.path import expanduser

IMAGE_SIZE = 64


def resize_with_pad(image, height=IMAGE_SIZE, width=IMAGE_SIZE):

    def get_padding_size(image):
        h, w, _ = image.shape
        longest_edge = max(h, w)
        top, bottom, left, right = (0, 0, 0, 0)
        if h < longest_edge:
            dh = longest_edge - h
            top = dh // 2
            bottom = dh - top
        elif w < longest_edge:
            dw = longest_edge - w
            left = dw // 2
            right = dw - left
        else:
            pass
        return top, bottom, left, right

    top, bottom, left, right = get_padding_size(image)
    BLACK = [0, 0, 0]
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    resized_image = cv2.resize(constant, (height, width))

    return resized_image


images = []
labels = []
def traverse_dir(path):
    for file_or_dir in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, file_or_dir))
        # print(abs_path)
        if os.path.isdir(abs_path):  # dir
            traverse_dir(abs_path)
        else:                        # file
            if file_or_dir.endswith('.jpg') or file_or_dir.endswith('.png'):
                image = read_image(abs_path)
                images.append(image)
                labels.append(path)

    return images, labels


def read_image(file_path):
    image = cv2.imread(file_path)
    image = resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE)

    return image


def extract_data(path):
    images, labels = traverse_dir(path)
    images = np.array(images)
    labels = np.array([0 if label.endswith('boss') else 1 for label in labels])
    return images, labels

def read_face_scrub_csv():
    home = expanduser("~")
    path = home + '/datasets/face_scrub/download'
    print path

    actor_id = 0
    dict_actor_id = {}
    dict_id_actor = {}
    images = []
    labels = []

    for actor in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, actor))
        # print(abs_path)
        thumnails_path = abs_path + '/face'
        if actor_id == 5:
            images = np.array(images)
            labels = np.array(labels)
            return images, labels, dict_actor_id, dict_id_actor
        if os.path.isdir(thumnails_path):
            dict_actor_id[actor] = actor_id
            dict_id_actor[actor_id] = actor
            for actor_picture in os.listdir(thumnails_path):
                image_path = thumnails_path + '/'+ actor_picture
                if image_path.endswith('.jpg') or file_or_dir.endswith('.png') or file_or_dir.endswith('.JPG'):
                    # print actor, image_path
                    image = cv2.imread(image_path)

                    # cv2.imshow('image',image)
                    # cv2.waitKey()
                    image = cv2.resize(image, (IMAGE_SIZE,IMAGE_SIZE))

                    # cv2.imshow('image',image)
                    # cv2.waitKey()

                    images.append(image)
                    labels.append(actor_id)
            actor_id += 1
    # print dict_actor_id
    # print dict_id_actor

    # images = np.array(images)
    # labels = np.array(labels)
    return images, labels, dict_actor_id, dict_id_actor

        # if os.path.isdir(abs_path):  # dir
        #     traverse_dir(abs_path)
        # else:                        # file
        #     if file_or_dir.endswith('.jpg') or file_or_dir.endswith('.png'):
        #         image = read_image(abs_path)
        #         images.append(image)
        #         labels.append(path)
