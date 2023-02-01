import numpy as np
import cv2

def order_points(pts):
	# initializing the list of coordinates to be ordered
	rect = np.zeros((4, 2), dtype = "float32")

	s = pts.sum(axis = 1)
	# top-left point will have the smallest sum
	rect[0] = pts[np.argmin(s)]
	# bottom-right point will have the largest sum
	rect[2] = pts[np.argmax(s)]

	'''computing the difference between the points, the
	top-right point will have the smallest difference,
	whereas the bottom-left will have the largest difference'''
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# returns ordered coordinates
	return rect


def perspective_transform(image, pts):
	# unpack the ordered coordinates individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	'''compute the width of the new image, which will be the
	maximum distance between bottom-right and bottom-left
	x-coordiates or the top-right and top-left x-coordinates'''
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	'''compute the height of the new image, which will be the
	maximum distance between the top-left and bottom-left y-coordinates'''
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	'''construct the set of destination points to obtain an overhead shot'''
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# compute the perspective transform matrix
	transform_matrix = cv2.getPerspectiveTransform(rect, dst)
	# Apply the transform matrix
	warped = cv2.warpPerspective(image, transform_matrix, (maxWidth, maxHeight))

	# return the warped image
	return warped