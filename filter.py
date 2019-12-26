import cv2
import numpy as np
import glob
import os
import time

path = 'C:\\Users\\Andrew\\workspace\\ColorShift\\TestPhotos'

color = 'Blue'

folder = glob.glob('{}\\{}Filtered\\*'.format(path, color))
for f in folder:
    os.remove(f)

lower_red = np.array([165, 100, 120])
upper_red = np.array([190, 255, 255])

lower_blue = np.array([90, 45, 120])
upper_blue = np.array([150, 255, 255])

kernel = np.ones((500, 500), np.float32)/250000

files =  glob.glob('{}\\{}\\*.jpg'.format(path, color))


for image in files:
    start = time.time()
    img = cv2.imread(image)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    smooth = cv2.filter2D(mask, -1, kernel)
    _, thresh = cv2.threshold(smooth, 25, 225, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh.copy(), 1, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)

        M = cv2.moments(c)

        cx = int(M['m10'] / M['m00'])

        cy = int(M['m01'] / M['m00'])

    cv2.circle(img, (cx, cy), 10, 10, thickness=10)
    cv2.circle(img, (1512, cy), 10, 10, thickness=10)
    cv2.circle(img, (1512, 4032), 10, 10, thickness=10)

    cv2.line(img, (1512, cy), (1512, 4032), (0, 0, 255), 10)
    cv2.line(img, (1512, cy), (cx, cy), (255, 255, 255), 10)
    cv2.line(img, (cx, cy), (1512, 4032), (0, 255, 0), 10)

    cv2.putText(img, "Theta = {}".format((180/np.pi) * np.arctan(abs(cx - 1512) / abs(3024 - cy))), (250, 250), cv2.FONT_HERSHEY_PLAIN, 10, (0,0,0), thickness=25)

    cv2.drawContours(img, contours, -1, (255, 0, 0), 10)

    if cx <= 1512:
        print("Turn Left")

    if cx >= 1512:
        print("Turn Right")

    print('Image: {}\n'
          'Time: {}s\n'
          .format(os.path.basename(image), time.time() - start))

    # res = cv2.bitwise_and(img, img, mask=thresh)

    outfile = '{}\\{}Filtered\\{}'.format(path, color, os.path.basename(image))
    cv2.imwrite(outfile, img)
