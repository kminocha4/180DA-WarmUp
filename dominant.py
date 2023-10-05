
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist
def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar

cap = cv.VideoCapture(0)
while(1):
    # Take each frame
    _, frame = cap.read()
    # Convert BGR to HSV
    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    img = img.reshape((img.shape[0] * img.shape[1],3)) #represent as row*column,channel number
    clt = KMeans(n_clusters=3) #cluster number
    clt.fit(img)

    hist = find_histogram(clt)
    bar = plot_colors2(hist, clt.cluster_centers_)

    plt.axis("off")
    # define range of blue color in HSV
    # lower_green = np.array([40,40,40])
    # upper_green = np.array([70,255,255])
    # # Threshold the HSV image to get only green colors
    # mask = cv.inRange(hsv, lower_green, upper_green)
    # x,y,w,h = cv.boundingRect(mask)
    # rect = cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    # # Bitwise-AND mask and original image
    # res = cv.bitwise_and(frame,frame, mask= mask)
    cv.imshow('frame',frame)
    cv.imshow('bar',bar)
    k = cv.waitKey(2) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()

# img = cv2.imread("pic/img7.jpeg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# img = img.reshape((img.shape[0] * img.shape[1],3)) #represent as row*column,channel number
# clt = KMeans(n_clusters=3) #cluster number
# clt.fit(img)

# hist = find_histogram(clt)
# bar = plot_colors2(hist, clt.cluster_centers_)

# plt.axis("off")
# plt.imshow(bar)
# plt.show()