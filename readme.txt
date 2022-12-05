Program ini dibuat dengan bahasa pemrograman python dengan database MongoDb

############ Struktur database MongoDb ##############
	
	Nama Database 	: presensi
	collection 	: hasil
	field		: nama, waktu, tanggal, tipe
	
#####################################################

*********** Struktur Program ***********

1. add_person.py 	: untuk menambahkan orang
2. recog_eigen/lbph.py	: untuk tes program
3. train_eigen/lbph.py	: untuk training program
4. table.py		: untuk menampilkan hasil presensi
5. many.py		: untuk tes program dengan hasil berupa diagram batang

*******************************************

     ?????? Cara Menjalankan Program ????????
 	
	jalankan gui.py untuk pertama kalinya

     ????????????????????????????????????????

/////catatan

1. ganti sumber camera 
   [video_capture = cv2.VideoCapture(1)]

2. patikan gambar untuk tes/training sudah grayscale

3. ganti direktori gambar untuk tes/training
   [self.face_dir = 'asik'] 