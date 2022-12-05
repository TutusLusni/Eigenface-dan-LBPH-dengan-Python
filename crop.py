import urllib.request
from tkinter import messagebox, Tk

import numpy as np
import cv2
import sys
import os

FREQ_DIV = 3  # frequency divider for capturing training images
RESIZE_FACTOR = 4
# NUM_TRAINING = 2

NUM_TRAINING = 100


class AddPerson:
    def __init__(self):
        cascPath = "haarcascades/haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascPath)
        self.face_dir = 'face_data'
        self.face_name = 'z4'
        self.path = os.path.join(self.face_dir, self.face_name)
        if not os.path.isdir(self.face_dir):
            os.mkdir(self.face_dir)
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        # self.count_captures = 0
        # self.count_timer = 0

    def capture_training_images(self):
        # video_capture = cv2.VideoCapture(1)
        # # url="http://192.168.43.1:8080/shot.jpg"
        dir_tes = 'frontal'
        list = os.listdir('Person17')  # dir is your directory path
        number_files = len(list)
        # print(number_files)
        i = 0
        while i < number_files:
            # self.count_timer += 1

            # imgResp = urllib.request.urlopen(url)
            # imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
            # frame = cv2.imdecode(imgNp, -1)
            no = number_files - (i)
            # print(no)
            frame = cv2.imread('Person17/gambar ('+str(i+1)+').jpg')
            inImg = np.array(frame)
            self.process_image(inImg)
            i+=1
            # outImg = self.process_image(inImg)
            # cv2.imshow('Video', outImg)

            # When everything is done, release the capture on pressing 'q'
            # if cv2.waitKey(1) & 0xFF == ord('q'):
                # video_capture.release()
        cv2.waitKey()
        cv2.destroyAllWindows()
        return

    def process_image(self, inImg):
        frame = cv2.flip(inImg, 1)
        resized_width, resized_height = (112, 92)
        # if self.count_captures < NUM_TRAINING:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, (int(gray.shape[1] / RESIZE_FACTOR), int(gray.shape[0] / RESIZE_FACTOR)))
        faces = self.face_cascade.detectMultiScale(
            gray_resized,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) > 0:
            areas = []
            for (x, y, w, h) in faces:
                areas.append(w * h)
            max_area, idx = max([(val, idx) for idx, val in enumerate(areas)])
            face_sel = faces[idx]

            x = face_sel[0] * RESIZE_FACTOR
            y = face_sel[1] * RESIZE_FACTOR
            w = face_sel[2] * RESIZE_FACTOR
            h = face_sel[3] * RESIZE_FACTOR

            face = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (resized_width, resized_height))
            img_no = sorted([int(fn[:fn.find('.')]) for fn in os.listdir(self.path) if fn[0] != '.'] + [0])[-1] + 1

            # if self.count_timer % FREQ_DIV == 0:
            cv2.imwrite('%s/%s.png' % (self.path, img_no), face_resized)
                # self.count_captures += 1
                # print ("Captured image: ", self.count_captures)

            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # cv2.putText(frame, '%s-%s' % (self.face_name, self.count_captures), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        # elif self.count_captures == NUM_TRAINING:
        #     root = Tk()
        #     root.withdraw()
        #     cv2.imwrite('%s/%s.png' % (self.face_dir,self.face_name), frame)
        #     messagebox.showinfo("Konfirmasi","Training selesai, klik 'OK' kemudian tekan 'q' ")
        #     root.destroy()
        #     root.quit()
        #     cv2.destroyAllWindows()
        #     print ("Training data captured. Press 'q' to exit.")
        #     self.count_captures += 1

        return


if __name__ == '__main__':
    trainer = AddPerson()
    trainer.capture_training_images()
    print ("Type in next user to train, or press Recognize")

