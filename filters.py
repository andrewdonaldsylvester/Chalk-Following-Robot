import cv2
import numpy as np

lower_red = np.array([165, 100, 120]) # defining lower and upper bounds for red and blue filters
upper_red = np.array([190, 255, 255])

lower_blue = np.array([90, 45, 120])
upper_blue = np.array([150, 255, 255])

kernel = np.ones((500, 500), np.float32)/250000 # defining kernel used to smooth frame

def red_filter(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # converting the frame into hsv format for filtering

    red_mask = cv2.inRange(hsv, lower_red, upper_red)  # filtering the frame for red
    red_smooth = cv2.filter2D(red_mask, -1,
                              kernel)  # smoothing the mask with a large mean kernel to thicken chalk lines
    _, red_thresh = cv2.threshold(red_smooth, 25, 225,
                                  cv2.THRESH_BINARY)  # thresholding the image to remove extra noise
    red_res = cv2.bitwise_and(frame, frame, mask=red_thresh)  # drawing the chalk line onto a black background

    red_contours, _ = cv2.findContours(red_thresh.copy(), 1, cv2.CHAIN_APPROX_NONE)  # outlining the chalk line

    if len(red_contours) > 0:
        red_c = max(red_contours, key=cv2.contourArea)  # finding the largest outline in case of stray noise

        red_moments = cv2.moments(red_c)  # calculating moments in order to find center point of the chalk line

        red_cx = int(red_moments['m10'] / red_moments['m00'])  # x value of the center point

        red_cy = int(red_moments['m01'] / red_moments['m00'])  # y value of the center point

        cv2.circle(red_res, (red_cx, red_cy), 10, 10, thickness=10)  # drawing the center of the chalk line
        cv2.circle(red_res, (1512, red_cy), 10, 10, thickness=10)
        cv2.circle(red_res, (1512, 4032), 10, 10, thickness=10)

        cv2.line(red_res, (1512, red_cy), (1512, 4032), (0, 0, 255), 10)  # drawing a line of where the robot is facing
        cv2.line(red_res, (red_cx, red_cy), (1512, 4032), (0, 255, 0),
                 10)  # drawing a line from the robot to the center of the chalk

        cv2.putText(red_res,  # calculating and writing the angle between the robot and the chalk center
                    "Theta = {}".format((180 / np.pi) * np.arctan(abs(red_cx - 1512) / abs(3024 - red_cy))),
                    (250, 250),
                    cv2.FONT_HERSHEY_PLAIN, 10, (255, 255, 255), thickness=25)

        cv2.drawContours(red_res, red_contours, -1, (255, 0, 0), 10)  # drawing the outline of the chalk line

    return red_res


def blue_filter(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # converting the frame into hsv format for filtering

    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)  #
    blue_smooth = cv2.filter2D(blue_mask, -1, kernel)
    _, blue_thresh = cv2.threshold(blue_smooth, 25, 225, cv2.THRESH_BINARY)
    blue_res = cv2.bitwise_and(frame, frame, mask=blue_thresh)

    blue_contours, _ = cv2.findContours(blue_thresh.copy(), 1, cv2.CHAIN_APPROX_NONE)

    if len(blue_contours) > 0:
        blue_c = max(blue_contours, key=cv2.contourArea)

        blue_moments = cv2.moments(blue_c)

        blue_cx = int(blue_moments['m10'] / blue_moments['m00'])

        blue_cy = int(blue_moments['m01'] / blue_moments['m00'])

        cv2.circle(blue_res, (blue_cx, blue_cy), 10, 10, thickness=10)
        cv2.circle(blue_res, (1512, blue_cy), 10, 10, thickness=10)
        cv2.circle(blue_res, (1512, 4032), 10, 10, thickness=10)

        cv2.line(blue_res, (1512, blue_cy), (1512, 4032), (0, 0, 255), 10)
        cv2.line(blue_res, (1512, blue_cy), (blue_cx, blue_cy), (255, 255, 255), 10)
        cv2.line(blue_res, (blue_cx, blue_cy), (1512, 4032), (0, 255, 0), 10)

        cv2.putText(blue_res,
                    "Theta = {}".format((180 / np.pi) * np.arctan(abs(blue_cx - 1512) / abs(3024 - blue_cy))),
                    (250, 250),
                    cv2.FONT_HERSHEY_PLAIN, 10, (255, 255, 255), thickness=25)

        cv2.drawContours(blue_res, blue_contours, -1, (255, 0, 0), 10)

    return blue_res
