from face_landmark_detection import makeCorrespondence
from delaunay import makeDelaunay
from faceMorph import makeMorphs,makeMorphs_three
from sys import argv
import subprocess
import argparse
import shutil
import os

def doMorphing(thePredictor,theImage1,theImage2,theDuration,theFrameRate,theResult):
	[size,img1,img2,list1,list2,array]=makeCorrespondence(thePredictor,theImage1,theImage2)
	if(size[0]==0):
		print("Cannot find a face in the image "+size[1])
		return
	list4=makeDelaunay(size[1],size[0],array)
	makeMorphs(theDuration,theFrameRate,img1,img2,list1,list2,list4,size,theResult)


def doMorphing_three(Predictor,Image1,Image2,Image3,Image1_2,Image2_2,Image3_2,Duration,FrameRate,Result):
	[size,img1,img2,list1,list2,array]=makeCorrespondence(Predictor,Image1,Image2)
	[size1,img3,img4,list3,list4,array2]=makeCorrespondence(Predictor,Image2_2,Image3)
	[size2,img5,img6,list5,list6,array3]=makeCorrespondence(Predictor,Image3_2,Image1_2)
	
	if(size[0]==0 or size1[0]==0 or size2[0]==0):
		print("Cannot find a face in the image ")
		return
	list_m1=makeDelaunay(size[1],size[0],array)
	list_m2=makeDelaunay(size1[1],size1[0],array2)
	list_m3=makeDelaunay(size2[1],size2[0],array3)
	makeMorphs_three(Duration,FrameRate,img1,img2,img3,img4,img5,img6,list1,list2,list3,list4,list5,list6,list_m1,list_m2,list_m3,size,Result)
	

if __name__ == "__main__":

	parser=argparse.ArgumentParser()
	parser.add_argument("num", type=int, help="The Image Number")
	if argv[1]=='2':
		parser.add_argument("img1", help="The First Image")
		parser.add_argument("img2", help="The Second Image")
	elif argv[1]=='3':
		parser.add_argument("img1", help="The First Image")
		parser.add_argument("img2", help="The Second Image")
		parser.add_argument("img3", help="The Third Image")
	else:
		print("Wrong Image Number")
		exit(0)

	parser.add_argument("res", help="The Resultant Video")
	args=parser.parse_args()
	dur=7		#video take 7 seconds
	frame=20	#video rate 20 image per second
	
	if args.num==2:
		with open(args.img1,'rb') as image1, open(args.img2,'rb') as image2:
			doMorphing('shape_predictor_68_face_landmarks.dat',image1,image2,dur,frame,args.res)
	elif args.num==3:
		with open(args.img1,'rb') as img1, open(args.img2,'rb') as img2, open(args.img3,'rb') as img3, open(args.img1,'rb') as img1_2, open(args.img2,'rb') as img2_2, open(args.img3,'rb') as img3_2:
			doMorphing_three('shape_predictor_68_face_landmarks.dat',img1,img2,img3,img1_2,img2_2,img3_2,dur,frame,args.res)


