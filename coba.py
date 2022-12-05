import datetime
import os

import PIL
import numpy as np
import cv2
from PIL import Image
from PIL import ImageEnhance
from tkinter import messagebox
from tkinter import *

# x = datetime.datetime.now()
# root = Tk()
# print(x.strftime("%d/%m/%Y"))
# print(x.strftime("%X"))
#
# root.withdraw()
# messagebox.askyesno('tanya','coba????')
# # root.deiconify()
# root.destroy()
# root.quit()
# import cv2
#
# for eachfile in os.listdir('Test/frontal'):
#             if eachfile.lower().endswith(('.png', '.jpg', '.jpeg')) == False:
#                 continue
#             # data = eachfile.split('.',)
#             print(eachfile)


for eachfile in os.listdir('Test/frontal'):
    if not eachfile.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    # data = eachfile.split('.',)
    # print(eachfile)
    img = cv2.imread('Test/frontal/'+eachfile)
    intensity = np.ones(img.shape, dtype ="uint8") * 50
    file = cv2.subtract(img, intensity)
    cv2.imwrite('Test/cahaya/'+eachfile,file)
cv2.waitKey()
cv2.destroyAllWindows()

# print(np.arange(3))