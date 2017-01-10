
import cv2

from boss_train import Model
from image_show import show_image
import os

def write_crop(file_name, image):
    new_file = './data/boss/' + file_name
    abs_path = os.path.abspath(new_file)
    print abs_path
    cv2.imwrite(abs_path, image)


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cascade_path = "/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
    model = Model()
    model.load('boss_save')
    i = 0
    while True:
        _, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cascade_path)

        facerect = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(10, 10))

        if len(facerect) > 0:
            print('face detected')
            color = (255, 255, 255)
            for rect in facerect:
                i += 1
                # cv2.rectangle(frame, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), color, thickness=2)
                # print "rect :", rect

                x, y = rect[0:2]
                width, height = rect[2:4]
                image = frame[y - 10: y + height, x: x + width]
                # cv2.imshow('image',image)
                file_name = str(i) + '.jpg'

                write_crop(file_name,image)
                # result = model.predict(image)
                # print result
                # if result == 0:
                #     print('Boss is approaching')
                # else:
                #     print('Not boss')

        k = cv2.waitKey(1000)
        # returns k is -1

        if k == 27:
            # k is ascii escape
            break

    cap.release()
    cv2.destroyAllWindows()
