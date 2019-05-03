import sys 
from PyQt5 import QtWidgets
import cv2
import numpy as np
import os 
from PIL import Image


#Arayuz İçin Sınıf 
class Pencere(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()#miras aldık

        self.init_ui()
    def init_ui(self):
        #Input,yazı ve buton ekledik
        self.kisi_ekleme = QtWidgets.QLabel("Kişi Ekleme (ID=1)")
        self.id_girisi = QtWidgets.QLineEdit()
        self.buton1 = QtWidgets.QPushButton("Başlat")
        
        self.kisileri_kaydetme = QtWidgets.QLabel("Kişileri Kaydet")
        self.buton2 = QtWidgets.QPushButton("Başlat")

        self.kisileri_tanıma = QtWidgets.QLabel("Kişileri Tanı")
        self.buton3 = QtWidgets.QPushButton("Başlat")

        #Yatay ve dikey eksen olustur
        v_box = QtWidgets.QVBoxLayout()

        v_box.addWidget(self.kisi_ekleme)
        v_box.addWidget(self.id_girisi)
        v_box.addWidget(self.buton1)
        v_box.addStretch()

        v_box.addWidget(self.kisileri_kaydetme)
        v_box.addWidget(self.buton2)
        v_box.addStretch()

        v_box.addWidget(self.kisileri_tanıma)
        v_box.addWidget(self.buton3)

        h_box = QtWidgets.QHBoxLayout()

        h_box.addStretch()
        h_box.addLayout(v_box)
        h_box.addStretch()


        self.setLayout(h_box)
        #Butonlara click event verdi
        self.buton1.clicked.connect(self.click1)
        self.buton2.clicked.connect(self.click2)
        self.buton3.clicked.connect(self.click3)
        self.show()
    #1.Butonun eventi
    def click1(self):
        cam = cv2.VideoCapture(0)
        cam.set(3, 640) # video genişligi
        cam.set(4, 480) # video yüksekliği

        face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        # Her kişinin id numarası
        #face_id = input('\n kişi id girip enter basınız <return> ==>  ')
        #print(self.id_girisi.text())
        face_id = self.id_girisi.text()
        
        print("\n [INFO] Initializing face capture. Look the camera and wait ...")
        # Fotoğraf sayıcı
        count = 0

        while(True):

            ret, img = cam.read()
            img = cv2.flip(img, 1) # Çevirici
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)

            for (x,y,w,h) in faces:

                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
                count += 1

                # Kaydedilecek datasetlerin isim ve konumu
                cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

                cv2.imshow('image', img)

            k = cv2.waitKey(100) & 0xff # ESC basınca kapansın
            if k == 27:
                break
            elif count >= 100: # 100 tana fotoğraf çekmesi
                break

        # Temizleme
        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()
    #2.Butonun eventi
    def click2(self):
        path = 'dataset'

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        #görüntüyü label yapan fonksiyon
        def getImagesAndLabels(path):

            imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
            faceSamples=[]
            ids = []

            for imagePath in imagePaths:

                PIL_img = Image.open(imagePath).convert('L') # griye çevirme
                img_numpy = np.array(PIL_img,'uint8')

                id = int(os.path.split(imagePath)[-1].split(".")[1])
                faces = detector.detectMultiScale(img_numpy)

                for (x,y,w,h) in faces:
                    faceSamples.append(img_numpy[y:y+h,x:x+w])
                    ids.append(id)

            return faceSamples,ids

        print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
        faces,ids = getImagesAndLabels(path)
        recognizer.train(faces, np.array(ids))

        # Kayıt trainer/trainer.yml
        recognizer.write('trainer/trainer.yml')

        print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
    #3.Butonun eventi
    def click3(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer/trainer.yml')
        cascadePath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascadePath)

        font = cv2.FONT_HERSHEY_SIMPLEX

        #id sayıcı
        id = 0

        # idnin isimde karsılıgı: örnek ==> Kişi1: id=1,  etc
        names = ['None', 'Kişi1', 'Kişi2', 'Kişi3', 'Z', 'Unknown'] 

        # Video Boyutları ve Kamera başlatmsı
        cam = cv2.VideoCapture(0)
        cam.set(3, 640) # Video Genisligi
        cam.set(4, 480) # Video Yüksekligi

        # Minumum yüz tanıyacak pencere boyutu
        minW = 0.1*cam.get(3)
        minH = 0.1*cam.get(4)

        while True:

            ret, img =cam.read()
            img = cv2.flip(img, 1) # Görüntüyü dödürme

            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale( 
                gray,
                scaleFactor = 1.2,
                minNeighbors = 5,
                minSize = (int(minW), int(minH)),
            )

            for(x,y,w,h) in faces:

                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

                id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

                # Uyusma eğer %100 olmasını istiyorsak 0 bizim icin ideal confidence.
                if (confidence < 35):
                    id = names[id]
                    confidence = "  {0}%".format(round(100 - confidence))
                else:
                    id = "Unknown"
                    confidence = "  {0}%".format(round(100 - confidence))
                
                cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
                cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
            
            cv2.imshow('camera',img) 

            k = cv2.waitKey(10) & 0xff # Çıkmak için ESC basınız
            if k == 27:
                break

        # Temizleme
        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()
#Pencere oluşturması
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    Pencere = Pencere()
    sys.exit(app.exec_())

