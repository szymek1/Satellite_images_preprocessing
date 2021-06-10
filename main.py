import numpy as np
import cv2

image = cv2.imread('mean_shift_result.png')
original_img = image.copy()
image_copy2 = image.copy()

data_of_clusters = np.loadtxt("clusters.txt", dtype=str)
data_of_clusters = data_of_clusters.astype(np.float64)
#print(data_of_clusters[0]) = [151 131 120] <-- first color I want to detect
cv2.imshow("original",original_img)

blurred_img = cv2.GaussianBlur(image, (27, 27), 0)
#hsv = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)

threshold = np.array([70,30,20])#threshold added to Upper boundary
#Lower = data_of_clusters[0]
#Upper = data_of_clusters[0] #+ threshold

#NEW SOLUTION!!!: each mask given for each data_of_clusters[i] by using cv2.bitwise_and
#should be merged with each other and create one final mask of all clusters
#the try threshold/subtraction etc...
for i in range(21):
    Lower = data_of_clusters[i]
    Upper = data_of_clusters[i] + threshold
    mask = cv2.inRange(blurred_img, Lower, Upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    area = 0
    for c in cnts:
        area += cv2.contourArea(c)
        cv2.drawContours(original_img, [c], 0, (0, 0, 0), 2)

    print(area)
    #cv2.imshow('mask'+str(i), mask)

    #cv2.imshow('original'+str(i), original_img)

cv2.imshow("image_after_mask",cv2.absdiff(image_copy2,original_img))
#subtraction_product = cv2.subtract(image_copy2,original_img)
#final = cv2.bitwise_not(original_img,subtraction_product)
#cv2.imshow("result",final)
#inal_result = cv2.subtract(original_img,image_copy2)
#v2.imshow("result of subtraction",final_result)

#creating a mask
#mask = cv2.inRange(blurred_img, Lower, Upper)
#mask = cv2.erode(mask, None, iterations=2)
#mask = cv2.dilate(mask, None, iterations=2)
#cv2.imshow("mask",mask)
#image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#lower = np.array(data_of_clusters[0], dtype="uint8")
#upper = np.array(data_of_clusters[0], dtype="uint8")
#mask = cv2.inRange(image, lower, upper)
#
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
#
#cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#
#area = 0
#for c in cnts:
#    area += cv2.contourArea(c)
#    cv2.drawContours(original_img,[c], 0, (0,0,0), 2)

#print(area)
#cv2.imshow('mask', mask)
#cv2.imshow('original', original_img)
#cv2.imshow('opening', opening)
cv2.waitKey()
