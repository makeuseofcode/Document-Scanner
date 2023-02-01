import cv2
import imutils
from skimage.filters import threshold_local
from transform import perspective_transform

# Loading and displaying the original image
original_img = cv2.imread('sample.jpg')
copy = original_img.copy()
cv2.waitKey(1)

ratio = original_img.shape[0] / 500.0
img_resize = imutils.resize(original_img, height=500)
cv2.imshow('Resized image', img_resize)
cv2.waitKey(0)

gray_image = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayed Image', gray_image)
cv2.waitKey(0)

blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
edged_img = cv2.Canny(blurred_image, 75, 200)
cv2.imshow('Image edges', edged_img)
cv2.waitKey(0)

cnts, _ = cv2.findContours(edged_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        doc = approx
        break

p = []
for d in doc:
    tuple_point = tuple(d[0])
    cv2.circle(img_resize, tuple_point, 3, (0, 0, 255), 4)
    p.append(tuple_point)
cv2.imshow('Circled corner points', img_resize)
cv2.waitKey(0)

warped_image = perspective_transform(copy, doc.reshape(4, 2) * ratio)
warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Warped Image", imutils.resize(warped_image, height=650))
cv2.waitKey(0)

T = threshold_local(warped_image, 11, offset=10, method="gaussian")
warped = (warped_image > T).astype("uint8") * 255
cv2.imwrite('./'+'scan'+'.png',warped)

cv2.imshow("Final Scanned image", imutils.resize(warped, height=650))
cv2.waitKey(0)
cv2.destroyAllWindows()