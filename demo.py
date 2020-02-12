import numpy as np
import cv2
'''
image = cv2.imread('Untitled.png', 1)
cv2.imshow('Image', image)
cv2.waitKey(10000)
cv2.imwrite('lena.jpeg', image)
cv2.destroyAllWindows()
'''

'''
#Video capture via webcam
capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read(0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('image', gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
'''

#Draw circle wherever double clicked
# def draw_circle(event, x, y, flag, param):
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         cv2.circle(image, (x, y), 100, (255, 0, 0), -1)
#
# image = cv2.imread('lena.jpeg', 1)
# cv2.namedWindow('image')
# cv2.setMouseCallback('image', draw_circle)
#
# while True:
#     cv2.imshow('image', image)
#     if cv2.waitKey(20) & 0xFF == 27:
#         break
# cv2.destroyAllWindows()

# Manipulate a certain pixel
# image = cv2.imread('lena.jpeg', 1)
# image[100:150, 100:150] = [0, 0, 0]
# cv2.imshow('image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#Copy and past certain area to other place in image
# image = cv2.imread('lena.jpeg', 1)
# a = image[0:100, 0:100]
# print(image[100, 100])
# image[100:200, 100:200] = a
# cv2.imshow('image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Display particular color from image
# image = cv2.imread('lena.jpeg', 1)
# cv2.imshow('image', image)
# new_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# cv2.imshow('new_image', new_image)
# l_blue = np.array([70, 0, 0])
# u_blue = np.array([150, 252, 255])
# mask = cv2.inRange(new_image, l_blue, u_blue)
# cv2.imshow('mask', mask)
# res = cv2.bitwise_and(image, image, mask=mask)
# cv2.imshow('res', res)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Track object in video
#
# cap = cv2.VideoCapture(0)
# while(1):
#     _, frame = cap.read()
#     new_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     cv2.imshow('new_image', new_image)
#     l_blue = np.array([110, 50, 50])
#     u_blue = np.array([150, 252, 255])
#     mask = cv2.inRange(new_image, l_blue, u_blue)
#     cv2.imshow('mask', mask)
#     res = cv2.bitwise_and(frame, frame, mask=mask)
#     cv2.imshow('res', res)
#     k = cv2.waitKey(5) & 0xFF
#     if k == 27:
#         break

# Thresholding a image (Normal, adaptive(better in vaariable lightening condition), adaptive_gaussian
#
# image = cv2.imread('lena.jpeg', 0)
# cv2.imshow('image', image)
#
# ret, thresh1 = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY)
# thresh2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 20)
# thresh3 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 20)
# cv2.imshow('image', image)
# cv2.imshow('thresh', thresh1)
# cv2.imshow('thresh2', thresh2)
# cv2.imshow('thresh3', thresh3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Translation

image = cv2.imread('lena.jpeg', 0)
cv2.imshow('image', image)

row, col = image.shape
M = np.float32([[1, 0, 100],[0, 1, 100]])
translate = cv2.warpAffine(image, M, (col, row))
cv2.imshow('translate', translate)
resize = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
cv2.imshow('resize', resize)
cv2.waitKey(0)
cv2.destroyAllWindows()