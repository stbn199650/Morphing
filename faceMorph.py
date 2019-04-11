import numpy as np
import cv2
import sys
import os
import math
from subprocess import Popen, PIPE
from PIL import Image

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def morphTriangle(img1, img2, img, t1, t2, t, alpha) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))


    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []

    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask


def makeMorphs(theDuration,theFrameRate,theImage1,theImage2,theList1,theList2,theList4,size,theResult):

	totalImages=int(theDuration*theFrameRate)

	p = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-r', str(theFrameRate),'-s',str(size[1])+'x'+str(size[0]), '-i', '-', '-c:v', 'libx264', '-crf', '25','-vf','scale=trunc(iw/2)*2:trunc(ih/2)*2','-pix_fmt','yuv420p', theResult], stdin=PIPE)
	for j in range(0,totalImages):
        
        # Read images
		img1 = theImage1
		img2 = theImage2

        # Convert Mat to float data type
		img1 = np.float32(img1)
		img2 = np.float32(img2)

        # Read array of corresponding points
		points1 = theList1
		points2 = theList2
		points = [];
		alpha=j/(totalImages-1)

        # Compute weighted average point coordinates
		for i in range(0, len(points1)):
			x = ( 1 - alpha ) * points1[i][0] + alpha * points2[i][0]
			y = ( 1 - alpha ) * points1[i][1] + alpha * points2[i][1]
			points.append((x,y))

        # Allocate space for final output
		imgMorph = np.zeros(img1.shape, dtype = img1.dtype)

        # Read triangles from delaunay_output.txt
		for i in range(len(theList4)):    
			x = int(theList4[i][0])
			y = int(theList4[i][1])
			z = int(theList4[i][2])
            
			t1 = [points1[x], points1[y], points1[z]]
			t2 = [points2[x], points2[y], points2[z]]
			t = [ points[x], points[y], points[z] ]

            # Morph one triangle at a time.
			morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha)

		temp_res=cv2.cvtColor(np.uint8(imgMorph),cv2.COLOR_BGR2RGB)
		res=Image.fromarray(temp_res)
		res.save(p.stdin,'JPEG')

	p.stdin.close()
	p.wait()

def doMorph(j,p,theImg1,theImg2,theList1,theList2,array,totalImages):

	# Read images
	img1 = theImg1
	img2 = theImg2

	# Convert Mat to float data type
	img1 = np.float32(img1)
	img2 = np.float32(img2)

	# Read array of corresponding points
	points1 = theList1
	points2 = theList2
	points = [];
	alpha=j/(totalImages-1)

	# Compute weighted average point coordinates
	for i in range(0, len(points1)):
		x = ( 1 - alpha ) * points1[i][0] + alpha * points2[i][0]
		y = ( 1 - alpha ) * points1[i][1] + alpha * points2[i][1]
		points.append((x,y))

	# Allocate space for final output
	imgMorph = np.zeros(img1.shape, dtype = img1.dtype)

	# Read triangles from delaunay_output.txt
	for i in range(len(array)):    
		x = int(array[i][0])
		y = int(array[i][1])
		z = int(array[i][2])
            
		t1 = [points1[x], points1[y], points1[z]]
		t2 = [points2[x], points2[y], points2[z]]
		t = [ points[x], points[y], points[z] ]

		# Morph one triangle at a time.
		morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha)
	
	temp_res=cv2.cvtColor(np.uint8(imgMorph),cv2.COLOR_BGR2RGB)

	return temp_res
#    res=Image.fromarray(temp_res)
#    res.save(p.stdin,'JPEG')


def makeMorphs_three(Duration,FrameRate,theImg1,theImg2,theImg3,theImg4,theImg5,theImg6,theList1,theList2,theList3,theList4,theList5,theList6,array,array2,array3,size,Result):

	totalImages=int(Duration*FrameRate)

	p = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-r', str(FrameRate),'-s',str(size[1])+'x'+str(size[0]), '-i', '-', '-c:v', 'libx264', '-crf', '25','-vf','scale=trunc(iw/2)*2:trunc(ih/2)*2','-pix_fmt','yuv420p', Result], stdin=PIPE)
	for j in range(0,totalImages):
		temp_res=doMorph(j,p,theImg1,theImg2,theList1,theList2,array,totalImages)
		res=Image.fromarray(temp_res)
		res.save(p.stdin,'JPEG')
       	 
	for j in range(0,totalImages):
		temp_res=doMorph(j,p,theImg3,theImg4,theList3,theList4,array2,totalImages)
		res=Image.fromarray(temp_res)
		res.save(p.stdin,'JPEG')

	for j in range(0,totalImages):
		temp_res=doMorph(j,p,theImg5,theImg6,theList5,theList6,array3,totalImages)
		res=Image.fromarray(temp_res)
		res.save(p.stdin,'JPEG')

	p.stdin.close()
	p.wait()
#  makeMorphs(0.5,60,'2.jpg','3.jpg')

