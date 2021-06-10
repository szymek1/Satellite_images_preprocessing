import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import cv2


img = cv2.imread('img15.png')
img = cv2.medianBlur(img, 3)


flat_image = img.reshape((-1,3)) #flattening the imgage so it is no longer 3 dimensional
flat_image = np.float32(flat_image)


bandwidth = estimate_bandwidth(flat_image, quantile=0.06, n_samples=3000)
ms = MeanShift(bandwidth, max_iter=800, bin_seeding=True)
ms.fit(flat_image)
labeled=ms.labels_ #labels each individual class into segements
segments = np.unique(labeled)
cluster_centers = ms.cluster_centers_
cluster_centers = np.uint8(cluster_centers)

textfile = open("clusters.txt", "w+")#here clusters centers will be saved

print('Number of segments: ', segments.shape[0])


total = np.zeros((segments.shape[0], 3), dtype=float)
count = np.zeros(total.shape, dtype=float)
for i, label in enumerate(labeled):
    total[label] = total[label] + flat_image[i]
    count[label] += 1
avg = total/count
avg = np.uint8(avg)

#turning cluster_centers into list of tuples- [(),().....]
#color_boundaries = []
#for i in range(cluster_centers.shape[0]):
#    color_boundaries.append((cluster_centers[i][0],cluster_centers[i][1],cluster_centers[i][2]))

#saving cluster centers into the file
np.savetxt("clusters.txt", cluster_centers)
textfile.close()

# cast the labeled image into the corresponding average color
res = avg[labeled]
result = res.reshape((img.shape))


# show the result
cv2.imshow('result',result)
cv2.imwrite("mean_shift_result.png", result)
cv2.waitKey(0)
cv2.destroyAllWindows()


