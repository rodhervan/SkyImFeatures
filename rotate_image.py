# import the necessary packages
import argparse
import imutils
import cv2
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="rotate1.png",
	help="path to the input image")
args = vars(ap.parse_args())


# load the image and show it
image = cv2.imread(args["image"])
cv2.imshow("Original", image)
# grab the dimensions of the image and calculate the center of the
# image
(h, w) = image.shape[:2]
(cX, cY) = (w // 2, h // 2)
# rotate our image by 45 degrees around the center of the image
M = cv2.getRotationMatrix2D((cX, cY), -3.5, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotated by 45 Degrees", rotated)
# rotate our image by -90 degrees around the image
# M = cv2.getRotationMatrix2D((cX, cY), -90, 1.0)
# rotated = cv2.warpAffine(image, M, (w, h))
# cv2.imshow("Rotated by -90 Degrees", rotated)

cv2.imwrite("rotated1.png", rotated) 

cv2.waitKey(0)
cv2.destroyAllWindows()