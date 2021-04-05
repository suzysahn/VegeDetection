# import the necessary packages
from __future__ import print_function
from imutils import perspective
from imutils import contours
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import os
import cv2


def get_objCoord(pts):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost

	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]

	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([tl, tr, br, bl], dtype="float32")

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# remove the edges of the box where there might be objects we do not want
def edged_box(box, height, width):
	for p in box:
		if p[0] < 10 or p[1] < 10:
			return True
		if p[0] > (width - 10) or p[1] > (height - 10):
			return True
	return False

# get the starting and ending coordinates in order to move motor there
def rect_edges(box):
	(tl, tr, br, bl) = box
	top = tl if tl[1] < tr[1] else tr
	left = tl if tl[0] < bl[0] else bl
	right = tr[0] if tr[0] > br else br
	bottom = bl if bl[1] > br[1] else br
	return left, right


def display_meas(image_file, ref_width):
	# preprocess image and run it through canny edge
	img = cv2.imread(image_file)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (7, 7), 0) 
	canny = cv2.Canny(gray, 0, 100)
	canny = cv2.dilate(canny, None, iterations=1)
	canny = cv2.erode(canny, None, iterations=1)
	edgedout = cv2.imwrite("testedge.jpg", img=canny)

	# find contours in the edge map
	cnts = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	(cnts, _) = contours.sort_contours(cnts)

	# check edged contours and remove it
	height, width, channels = img.shape
	cntl = list(cnts)
	i = 0
	dS = 0
	while i < len(cntl):
		c = cntl[i]
		box = cv2.minAreaRect(c)
		box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
		box = np.array(box, dtype="int")
		cv2.drawContours(img, [box], -1, (0, 255, 0), 2)

		if edged_box(box, height, width):
			del cntl[i]
			continue

		# calc size of the rect
		(tl, tr, br, bl) = box
		(tltrX, tltrY) = midpoint(tl, tr)
		(blbrX, blbrY) = midpoint(bl, br)
		(tlblX, tlblY) = midpoint(tl, bl)
		(trbrX, trbrY) = midpoint(tr, br)
		dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
		dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

		# if first item, let dS as size
		if i == 0:
			dS = dA * dB
		# if size is smallest, move to front
		elif dS > (dA * dB):
			dS = dA * dB
			cx = cntl[i]
			del cntl[i]
			cntl.insert(0, cx)
		i = i + 1
	cnts = tuple(cntl)

	# initialize variables for further processing
	colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))
	pixelsPerMetric = None
	refObj = None

	# loop over the contours one at a time
	for (i, c) in enumerate(cnts):
		if cv2.contourArea(c) < 100:
			continue

		# compute the rotated bounding box of the contour, then
		box = cv2.minAreaRect(c)
		box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
		box = np.array(box, dtype="int")

		# draw the contours
		cv2.drawContours(img, [box], -1, (0, 255, 0), 2)
		cX = np.average(box[:,0])
		cY = np.average(box[:,1])
		print("Object #{}:".format(i + 1))
		print(box)

		# ordering the coordinates and draw the in the points
		rect = get_objCoord(box)
		for ((x, y), color) in zip(rect, colors):
			cv2.circle(img, (int(x), int(y)), 5, color, -1)

		# draw the object number
		cv2.putText(img, "Object #{}".format(i + 1),
			(int(rect[0][0] - 15), int(rect[0][1] - 15)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

		# compute the box midpoints
		(tl, tr, br, bl) = box
		(tltrX, tltrY) = midpoint(tl, tr)
		(blbrX, blbrY) = midpoint(bl, br)

		# compute the midpoint between the tl and bl, then tr and br
		(tlblX, tlblY) = midpoint(tl, bl)
		(trbrX, trbrY) = midpoint(tr, br)

		# draw the midpoints 
		cv2.circle(img, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
		cv2.circle(img, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
		cv2.circle(img, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
		cv2.circle(img, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

		# draw lines between the midpoints
		cv2.line(img, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
			(255, 0, 255), 2)
		cv2.line(img, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
			(255, 0, 255), 2)

		# compute the Euclidean distance between the midpoints
		dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
		dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

		# if the pixels per metric has not been initialized, then
		# compute it as the ratio of pixels to supplied metric
		# (in this case, inches)
		if refObj is None:
			(tl, tr, br, bl) = box
			(tltrX, tltrY) = midpoint(tl, tr)
			(blbrX, blbrY) = midpoint(bl, br)
			refObj = (box, (cX, cY), dB / ref_width)
		print(refObj)

		if pixelsPerMetric is None:
			D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
			pixelsPerMetric = D / ref_width

		# if this is the object that is the size of a quarter then 
		# draw the contours on the image
		orig = img.copy()
		cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
		cv2.drawContours(orig, [refObj[0].astype("int")], -1, (0, 255, 0), 2)

		# stack the reference coordinates and the object coordinates
		# to include the object center
		refCoords = np.vstack([refObj[0], refObj[1]])
		objCoords = np.vstack([box, (cX, cY)])

		for ((xA, yA), (xB, yB), color) in zip(refCoords, objCoords, colors):
			# draw circles corresponding to the current points and
			# connect them with a line
			cv2.circle(orig, (int(xA), int(yA)), 5, color, -1)
			cv2.circle(orig, (int(xB), int(yB)), 5, color, -1)
			cv2.line(orig, (int(xA), int(yA)), (int(xB), int(yB)),
				color, 2)

			D = dist.euclidean((xA, yA), (xB, yB)) / refObj[2]
			(mX, mY) = midpoint((xA, yA), (xB, yB))
			cv2.putText(orig, "{:.2f}in".format(D), (int(mX), int(mY - 10)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
		
		# compute the size of the object
		dimA = dA / pixelsPerMetric
		dimB = dB / pixelsPerMetric

		# draw the object sizes on the image
		cv2.putText(img, "{:.2f}in".format(dimA),
			(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
			0.65, (255, 255, 255), 2)
		cv2.putText(img, "{:.2f}in".format(dimB),
			(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
			0.65, (255, 255, 255), 2)

		# draw lines to the midpoints from left and top
		if (i > 0):
			cv2.line(img, (int(tltrX), int(tltrY)), (int(0), int(tltrY)),
				(255, 0, 0), 2)
			cv2.line(img, (int(trbrX), int(0)), (int(trbrX), int(trbrY)),
				(255, 0, 0), 2)

			dL = dist.euclidean((tltrX, tltrY), (0, tltrY))
			dT = dist.euclidean((tlblX, 0), (trbrX, trbrY))
			dimL = dL / pixelsPerMetric
			dimT = dT / pixelsPerMetric

			cv2.putText(img, "{:.2f}in".format(dimL),
				(15, int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
				0.65, (0, 0, 255), 2)
			cv2.putText(img, "{:.2f}in".format(dimT),
				(int(trbrX + 10), 30), cv2.FONT_HERSHEY_SIMPLEX,
				0.65, (0, 0, 255), 2)

	# return whatever is neccessary at the end of the project but for now, return the image
	return img


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-n", "--new", type=int, default=-1)
ap.add_argument("-w", "--width", type=float, required=True,
	help="width of the left-most object in the image (in inches)")
ap.add_argument("-f", "--file", type=str, required=True,
	help="input picture file with jpg format")		
args = vars(ap.parse_args())

# call object detection with file name and refrenece width
image = display_meas(args["file"], args["width"])

# show the image
cv2.imshow("Image", image)
cv2.waitKey(0)