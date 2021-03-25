#!/usr/bin/python3
import cv2
import numpy as np
import glob
import os
 
img_array = []
#size=0

#sorted(glob.glob('*.png'), key=os.path.getmtime)
for filename in sorted(glob.glob('../img/*.png'), key=os.path.getmtime):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 

out = cv2.VideoWriter('angle_computation.avi', cv2.VideoWriter_fourcc(*'DIVX'), 3, size)


 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
