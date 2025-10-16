import cv2
import numpy as np
import os

for home, dirs, files in os.walk('output/y/'):
	for filename in files:
		file_path = os.path.join(home, filename)
		if filename.endswith('png'):
			img = cv2.imread(file_path)
			gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			contours,hierarchy = cv2.findContours(gray_image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

			maxarea = 0
			for c in contours:
				area = cv2.contourArea(c)
				if area > maxarea:
					maxarea = area
					cmax = c
			x,y,w,h = cv2.boundingRect(cmax)
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)

			cv2.imshow('img', img)
			cv2.waitKey(0)

