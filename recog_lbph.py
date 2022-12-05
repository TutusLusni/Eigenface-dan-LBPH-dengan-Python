import datetime
from tkinter import messagebox, Tk

import numpy as np
import cv2
import os

import pymongo

RESIZE_FACTOR = 4

class RecogLBPH:
    def __init__(self):
        myclient = pymongo.MongoClient("mongodb://localhost:27017/")
        mydb = myclient["presensi"]
        self.mycol = mydb["hasil"]
        cascPath = "haarcascades/haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascPath)
        self.face_dir = 'asik'
        self.model = cv2.face.LBPHFaceRecognizer_create()
        self.face_names = []
        self.wk = datetime.datetime.now()
        self.orang = []
        self.data = []
        self.root = Tk()
        self.root.withdraw()

    def load_trained_data(self):
        names = {}
        key = 0
        for (subdirs, dirs, files) in os.walk(self.face_dir):
            for subdir in dirs:
                names[key] = subdir
                key += 1
        self.names = names
        self.model.read('trained_data/lbph_trained_data.xml')

    def show_video(self):
        video_capture = cv2.VideoCapture(0)
        while True:
            ret, frame = video_capture.read()
            inImg = np.array(frame)
            outImg, self.face_names = self.process_image(inImg)
            cv2.imshow('Video', outImg)

            # When everything is done, release the capture on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # img_no = sorted([int(fn[:fn.find('.')]) for fn in os.listdir('ttg') if fn[0] != '.'] + [0])[-1] + 1
                # cv2.imwrite('ttg/%s.png' % (img_no), frame)
                video_capture.release()
                cv2.destroyAllWindows()
                return

    def process_image(self, inImg):
        frame = cv2.flip(inImg,1)
        resized_width, resized_height = (112, 92)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, (int(gray.shape[1]/RESIZE_FACTOR), int(gray.shape[0]/RESIZE_FACTOR)))
        faces = self.face_cascade.detectMultiScale(
                gray_resized,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
                )
        persons = []
        for i in range(len(faces)):
            face_i = faces[i]
            x = face_i[0] * RESIZE_FACTOR
            y = face_i[1] * RESIZE_FACTOR
            w = face_i[2] * RESIZE_FACTOR
            h = face_i[3] * RESIZE_FACTOR
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (resized_width, resized_height))
            confidence = self.model.predict(face_resized)
            if confidence[1]<80:
                person = self.names[confidence[0]]
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 3)
                cv2.putText(frame, '%s - %.0f' % (person, confidence[1]), (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 2,(0, 255, 0), 2)

                self.orang.append(person)
                if (len(self.orang) == 3):
                    if (self.orang[0] == self.orang[1] and self.orang[0] == self.orang[2]):
                        for db in self.mycol.find(
                                {'nama': self.orang[0], 'tanggal': str(self.wk.strftime("%d/%m/%Y")), 'tipe' : 'LBPH'}):
                            self.data.append(db)

                        if len(self.data) == 0:
                            self.mycol.insert_one({"nama": self.orang[0], "waktu": str(self.wk.strftime("%X")),
                                                   "tanggal": str(self.wk.strftime("%d/%m/%Y")),
                                                   "tipe": "LBPH"})
                            messagebox.showinfo('Informasi', '' + self.orang[
                                0] + ' Telah Presensi Tekan OK dan Tekan q untuk keluar')
                            self.data = []
                            self.root.destroy()
                            # self.root.quit()
                if (len(self.orang) > 3):
                    self.orang = []
            else:
                person = 'Tidak Dikenali !'
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 3)
                cv2.putText(frame, '%s - %.0f' % (person, confidence[1]), (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 2,(0, 255, 0), 2)
            persons.append(person)
        return (frame, persons)


if __name__ == '__main__':
    recognizer = RecogLBPH()
    recognizer.load_trained_data()
    print ("Press 'q' to quit video")
    recognizer.show_video()

