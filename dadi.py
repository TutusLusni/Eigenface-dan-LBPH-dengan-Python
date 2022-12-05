from tkinter import *
import os

root = Tk(className = 'face_recognition_gui')
root.title('Pengenalan Wajah');
# root.wm_attributes("-transparentcolor", 'grey')

img = PhotoImage(file="assets/bgrnd.png")
logo = PhotoImage(file="assets/logoo.png")
btn_t_e = PhotoImage(file="assets/deteksi.png")
btn_t_l = PhotoImage(file="assets/keluar.png")
btn_r_e = PhotoImage(file="assets/btn_r_e.png")
btn_r_l = PhotoImage(file="assets/btn_r_l.png")
btn_hasil = PhotoImage(file="assets/hasil.png")
btn_add = PhotoImage(file="assets/add.png")

# root.config(background="black")

svalue = StringVar() # defines the widget state as string
image = Label(root, image=img)
image.place(x=-10,y=-10)

main = Frame(root,bg='white', height=400, width=350)
main.place (x=325, y=140)


# lg = Label(main, image=logo, bg="white")
# lg.place(x=110,y=10)

# l = Label(main, text="Tambahkan Nama :")
# l.config(font=("Times New Roman", 12), bg='white')
# l.place(x=20,y=100)

# w = Entry(main,textvariable=svalue) # adds a textarea widget
# w.place(x=150,y=102)


def train_eigen_btn_load():
    name = svalue.get()
    os.system('python train_eigen.py %s'%name)

def train_lbph_btn_load():
    name = svalue.get()
    os.system('python train_lbph.py %s'%name)

def recog_eigen_btn_load():
    os.system('python recog_eigen.py')

def recog_lbph_btn_load():
    os.system('python recog_lbph.py')

def hasil_presensi():
    root.destroy()
    os.system('python table.py')

def add_person():
    name = svalue.get()
    w.delete(0, END)
    os.system('python add_person.py %s'%name)


# add_btn = Button(main, image=btn_add, borderwidth=0, bg="white", command=add_person)
# add_btn.place(x=280,y=98)

f=Frame(main,height=1, width=250, bg="black")
f.place(x=50, y=100)

l = Label(main, text="Main Menu")
l.config(font=("Forte", 26), bg='white')
l.place(x=80, y= 50)

trainE_btn = Button(main, image=btn_t_e, borderwidth=0, bg="white", command=train_eigen_btn_load)
trainE_btn.place(x=60, y = 150)

recogL_btn = Button(main,image=btn_t_l, borderwidth=0, bg="white", command=root.destroy)
recogL_btn.place(x=60, y = 250)

# f=Frame(main,height=1, width=250, bg="black")
# f.place(x=50, y = 260)

# l = Label(main, text="Testing")
# l.config(font=("forte", 16), bg='white')
# l.place(x = 140, y= 270)

# recogE_btn = Button(main,image=btn_r_e, borderwidth=0, bg="white", command=recog_eigen_btn_load)
# recogE_btn.place(x = 20, y=310)

# recogL_btn = Button(main, image=btn_r_l, borderwidth=0, bg="white", command=recog_lbph_btn_load)
# recogL_btn.place(x=185, y = 310)

# f=Frame(main,height=1, width=250, bg="black")
# f.place(x=50, y = 360)

# l = Label(main, text="Hasil Presensi")
# l.config(font=("Forte", 16), bg = 'white')
# l.place(x = 110, y = 370)
#
# presensi_btn = Button(main, image=btn_hasil, borderwidth=0, bg="white", command=hasil_presensi)
# presensi_btn.place(x= 105, y = 410)

width_window = 1000
height_window = 600

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width/2) - (width_window/2)
y = (screen_height/2) - (height_window/2)

root.geometry("%dx%d+%d+%d" % (width_window,height_window,x,y))
root.resizable(0,0)
root.mainloop()