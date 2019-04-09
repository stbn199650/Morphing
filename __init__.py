from face_landmark_detection import makeCorrespondence
from delaunay import makeDelaunay
from faceMorph import makeMorphs
import subprocess
import argparse
import shutil
import os

def doMorphing(thePredictor,theImage1,theImage2,theDuration,theFrameRate,theResult):
	[size,img1,img2,list1,list2,list3]=makeCorrespondence(thePredictor,theImage1,theImage2)
	if(size[0]==0):
		print("Couldn't find a face in the image "+size[1])
		return
	list4=makeDelaunay(size[1],size[0],list3)
	makeMorphs(theDuration,theFrameRate,img1,img2,list1,list2,list4,size,theResult)

if __name__ == "__main__":

	parser=argparse.ArgumentParser()
	parser.add_argument("img1", help="The First Image")
	parser.add_argument("img2", help="The Second Image")
	parser.add_argument("dur",type=int, help="The Duration")
	parser.add_argument("fr",type=int, help="The Frame Rate")
	parser.add_argument("res", help="The Resultant Video")
	args=parser.parse_args()

	with open(args.img1,'rb') as image1, open(args.img2,'rb') as image2:
		doMorphing('shape_predictor_68_face_landmarks.dat',image1,image2,args.dur,args.fr,args.res)

"""import cv2
import numpy as np
from matplotlib import pyplot as plt
 
img = cv2.imread('ben.jpg', 0)
plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
"""
