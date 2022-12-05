import os
from tkinter import *
import tkinter as tk
from tkinter import ttk
import pymongo

myclient = pymongo.MongoClient("mongodb://localhost:27017/")

#use database named "organisation"
mydb = myclient["presensi"]

#use collection named "developers"
mycol = mydb["hasil"]

root = Tk(className="hasil_presensi")
root.title("Tabel Hasil")

data = []
q = StringVar()
var = StringVar()
img = PhotoImage(file="assets/bgrnd.png")
image = Label(root, image=img)
image.place(x=0,y=0)

def show(data):
    trv.delete(*trv.get_children())
    # print(len(data))
    a = 1
    i = (len(data) - 1)
    while i < len(data) and i != -1:
        trv.insert('','end',values=((a),data[i]['nama'],data[i]['waktu'],data[i]['tanggal'],data[i]['tipe']))
        a+=1
        i -= 1

def awal() :
    for db in mycol.find():
        data.append(db)
    show(data)

def search():
    data =[]
    q2 = q.get()
    var2 = var.get()
    # print(q2)

    if not q2 or q2=='dd/mm/yyyy':
        for db in mycol.find({'tipe': var2}):
            data.append(db)
    else:
        for db in mycol.find({'tanggal': q2, 'tipe': var2}):
            data.append(db)
    if not data:
        data =[{'_id': '', 'nama': 'Data tidak  ditemukan / tanggal salah !!', 'waktu': '', 'tanggal': '', 'tipe': ''}]

    # print('ini search :',data)
    show(data)

def clear():
    data =[]
    for db in mycol.find():
        data.append(db)

    show(data)
    ent.delete(0,END)
    ent.insert(0,"dd/mm/yyyy")


def back():
    root.destroy()
    os.system('gui.py')

def entry_text(event):
    ent.delete(0,END)

wrapper1 = LabelFrame(image, bg='#99e6ff', text="search", font="courier", height="1")
wrapper2 = LabelFrame(image, bg='#99e6ff', text="Hasil Presensi", font="courier")

wrapper1.pack(fill="both", expand="yes",padx=20, pady=10)
wrapper2.pack(fill="both", expand="yes",padx=20, pady=10)



#isi wrapper 2
trv = ttk.Treeview(wrapper2,column=(1,2,3,4,5), show="headings", height="14")
trv.pack()
scrl = ttk.Scrollbar(wrapper2, orient="vertical", command=trv.yview)
scrl.place(x=935, y=23, height=282)
trv.configure(yscrollcommand=scrl.set)


trv.heading(1, text="No")
trv.column(1, width=50)
trv.heading(2, text="Nama")
trv.column(2, width=350)
trv.heading(3, text="Pukul")
trv.column(3, width=200)
trv.heading(4, text="Tanggal")
trv.column(4, width=200)
trv.heading(5, text="Tipe")
trv.column(5, width=150)

# query="SELECT * FROM hasil"
# cursor.execute(query)
# data = cursor.fetchall()
for db in mycol.find():
    data.append(db)

# print(data)
show(data)


#isi wrapper 1
lbl =Label(wrapper1, bg='#99e6ff', text="Masukkan Tanggal")
lbl.pack(side=tk.LEFT, padx=10, pady=20)
ent =Entry(wrapper1, textvariable =q)
ent.pack(side=tk.LEFT, padx=6)
ent.insert(0,"dd/mm/yyyy")
ent.bind("<Button>", entry_text)

var.set('LBPH')
r = ('LBPH','Eigenface')
tipe=OptionMenu(wrapper1,var,*r)
tipe.pack(side=tk.LEFT, padx=10)

btn =Button(wrapper1, text= "Cari",command=search)
btn.pack(side=tk.LEFT, padx=20)
btn_clear =Button(wrapper1, text="Clear", command=clear)
btn_clear.pack(side=tk.LEFT, padx=6)


back_btn = Button(image,text="Kembali",command=back)
back_btn.pack(side=tk.LEFT, pady=10, padx=40)


width_window = 1000
height_window = 500

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width/2) - (width_window/2)
y = (screen_height/2) - (height_window/2)

root.geometry("%dx%d+%d+%d" % (width_window,height_window,x,y))
root.resizable(0,0)
root.mainloop()
