
import cv2
import os
from boss_train import Model
from image_show import show_image

def traverse_dir(path):
    images = []
    for file_or_dir in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, file_or_dir))
        # print(abs_path)
        if os.path.isdir(abs_path):  # dir
            traverse_dir(abs_path)
        else:                        # file
            if file_or_dir.endswith('.jpg') or file_or_dir.endswith('.png') or file_or_dir.endswith('.JPG'):
                frame = cv2.imread(abs_path)
                # cv2.imshow('image',frame)
                # print frame[0][0]
                crops = face_crop(frame)
                # images.append(face_crop(frame))
                i = 0
                for crop in crops:
                    if crop != None:
                        i += 1
                        write_crop(path, str(i)+file_or_dir, crop)


def face_crop(frame):
    images = []
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cascade_path)
    facerect = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(10, 10))
    if len(facerect) > 0:
        print('face detected')
        color = (255, 255, 255)
        for rect in facerect:
            # cv2.rectangle(frame, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]))
            # print "rect :", rect
            x, y = rect[0:2]
            width, height = rect[2:4]
            image = frame[y - 10: y + height, x: x + width]
            images.append(image)
            # cv2.imshow('image',image)
            # cv2.waitKey(0)
    return images

def write_crop(path, file_name, image):
    new_file = './data/boss_1_crops/' + file_name
    abs_path = os.path.abspath(new_file)
    print abs_path
    cv2.imwrite(abs_path, image)


if __name__ == '__main__':
    cascade_path = "/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
    traverse_dir('./data/boss_1/')
