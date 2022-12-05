import pymongo

myclient = pymongo.MongoClient("mongodb://localhost:27017/")

#use database named "organisation"
mydb = myclient["presensi"]

#use collection named "developers"
mycol = mydb["hasil"]

#a document
developer = [{"nama": "wina", "waktu":"19.30", "tanggal" :"26/05/2021", "tipe": "LBPH"},
             {"nama": "royu", "waktu":"20.30", "tanggal" :"29/05/2021", "tipe": "LBPH"},
             {"nama": "ida", "waktu":"20.30", "tanggal" :"29/05/2021", "tipe": "LBPH"}]

#insert a document to the collection
x = mycol.delete_many({'nama':'tutus', 'tipe':'LBPH'})

# data = mycol.find()
# data = []
# #list the databases
# tgl = '29/05/2021'
# tp = 'lbph'
#
# for db in mycol.find({'tanggal' : tgl, 'tipe': tp }):
#     data.append(db)
#
# print(data)
# # i=0
# # a=0
# # while a < len(data):
# #     print(data[a])
# #     a +=1