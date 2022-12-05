from sklearn.metrics import *
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

RESIZE_FACTOR = 4


class RecogEigenFaces:
    def __init__(self):
        cascPath = "haarcascades/haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascPath)
        self.face_dir = 'face_data'
        self.model = cv2.face.LBPHFaceRecognizer_create()
        self.face_names = []
        self.data = []
        self.dir_test = 'serong'

    def load_trained_data(self):
        names = {}
        key = 0
        for (subdirs, dirs, files) in os.walk(self.face_dir):
            for subdir in dirs:
                names[key] = subdir
                key += 1
        self.names = names
        self.model.read('trained_data/many_lbph_trained_data.xml')

    def show_video(self):
        # dir_tes = 'frontal'
        list = os.listdir('Test/'+self.dir_test)  # dir is your directory path
        number_files = len(list)
        # print(number_files)

        for eachfile in os.listdir('Test/'+self.dir_test):
            if eachfile.lower().endswith(('.png', '.jpg', '.jpeg')) == False:
                continue
            # data = eachfile.split('.',)
            # print(eachfile)
            frame = cv2.imread('Test/'+self.dir_test+'/'+eachfile)
            # frame = cv2.imread('Test/'+self.dir_test+'/z.png')
            inImg = np.array(frame)
            self.process_image(inImg)
            print(self.data)

            # i += 1
        print(len(self.data))
        cv2.waitKey()
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
        # print(range(len(faces)))
        if(range(len(faces)) == range(0,0)) :
            person = 'Tidak Dikenali !'
            self.data.append(person)

        for i in range(len(faces)):
            face_i = faces[i]
            # print(face_i)
            x = face_i[0] * RESIZE_FACTOR
            y = face_i[1] * RESIZE_FACTOR
            w = face_i[2] * RESIZE_FACTOR
            h = face_i[3] * RESIZE_FACTOR
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (resized_width, resized_height))
            confidence = self.model.predict(face_resized)
            if confidence[1]<80:
                person = self.names[confidence[0]]
                self.data.append(person)

            else:
                person = 'Tidak Dikenali !'
                self.data.append(person)
        # print(self.data)
        return (frame, persons)

    def confusion(self):
        # self.data.append('pred false !')
        print('Hasil Prediksi',self.data)
        act = []
        y_true = []
        y_pred = []
        for name in os.listdir('face_data'):
            # if name.lower().endswith(('.png', '.jpg', '.jpeg')) == False:
            #     continue
            # data = name.split('.')
            act.append(name)
        act.append('false')
        act.append('false')
        print("data aktual : ",act)

        x = 0
        while x < len(act):
            if act[x] != 'false' :
                y_true.append(1)
            else:
                y_true.append(0)
            x += 1

        i=0
        while i<len(self.data):
            if self.data[i] == act[i]:
                y_pred.append(1)
            else:
                y_pred.append(0)
            i += 1

        print("data aktual biner : ",y_true)
        print("data aktual pred : ", y_pred)
        print('panjang aktual : ',len(y_true))
        print('panjang pred : ',len(y_pred))

        accuracy = accuracy_score(y_true, y_pred)
        prfs = precision_recall_fscore_support(y_true,y_pred)
        precision = prfs[0][1]
        recall = prfs[1][1]
        fscore = prfs[2][1]
        print("confusion matrix  ne: \n", confusion_matrix(y_true, y_pred))
        print("accuracy : ",accuracy)
        print("precission : ", precision)
        print("recal : ",recall)
        print("f1-score : ",fscore)
        print(classification_report(y_true,y_pred))

        y = [accuracy, precision,recall,fscore]
        x = ['akurasi', 'presisi', 'recal', 'f-measure']
        width = 0.35

        fig, ax = plt.subplots()
        idn = np.arange(len(y))
        rects = ax.bar(idn, y, width, color="yellow")
        ax.set_ylabel("Nilai")
        ax.set_title('LBPH '+self.dir_test)
        ax.set_yticks(idn)
        ax.set_xlabel("Confusion Matrix")
        ax.set_xticks(idn)
        ax.set_xticklabels(x)
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2, 1.05 * height, "{:.3f}".format(height))
        autolabel(rects)
        plt.show()

        # plt.figure()
        # plt.title('LBPH '+self.dir_test)
        # plt.xlabel('Confusion Matrix')
        # plt.ylabel('Nilai')
        # plt.bar(('akurasi', 'presisi', 'recal', 'f-measure'), (accuracy, precision,recall,fscore))


if __name__ == '__main__':
    recognizer = RecogEigenFaces()
    recognizer.load_trained_data()
    # print ("Press 'q' to quit video")
    recognizer.show_video()
    recognizer.confusion()
