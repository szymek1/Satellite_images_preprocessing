import numpy as np
import cv2

#def from_min_to_max(img_g,minimum,maximum):
#    ret, mask = cv2.threshold(gray, minimum, maximum, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#    kernel = np.ones((9, 9), np.uint8)
#    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#    return mask
#
#def from_min_to_avg(img_g,minimum,avg):
#    ret, mask = cv2.threshold(gray, minimum, avg, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#    kernel = np.ones((9, 9), np.uint8)
#    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#    return mask
#
#def from_avg_to_max(img_g,avg,maximum):
#    ret, mask = cv2.threshold(gray, avg, maximum, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#    kernel = np.ones((9, 9), np.uint8)
#    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#    return mask


image = cv2.imread('mean_shift_result.png')
original_img = image.copy()
image_copy2 = image.copy()

data_of_clusters = np.loadtxt("clusters.txt", dtype=str)
data_of_clusters = data_of_clusters.astype(np.float64)
#print(data_of_clusters[0]) = [151 131 120] <-- first color I want to detect

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#the idea is to use funcs above to apply threshold and use every element of data_of_clusters elements
#right now it only uses min and max but maybe also avg can help
for i in range(21):
    actual_max = np.amax(data_of_clusters[i])
    actual_min = np.amin(data_of_clusters[i])
    #actual_avg =
    ret, mask = cv2.threshold(gray, actual_min, actual_max, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)





# threshold input image using otsu thresholding as mask and refine with morphology
#ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#kernel = np.ones((9,9), np.uint8)
#mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# put mask into alpha channel of result
result = image.copy()
result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
result[:, :, 3] = mask

# save resulting masked image
cv2.imwrite('retina_masked.png', result)
cv2.imshow("result",result)
cv2.waitKey()