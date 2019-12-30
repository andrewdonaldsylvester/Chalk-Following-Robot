from flask import Flask, render_template, Response
import cv2
import numpy as np

lower_red = np.array([165, 100, 120]) # defining lower and upper bounds for red and blue filters
upper_red = np.array([190, 255, 255])

lower_blue = np.array([90, 45, 120])
upper_blue = np.array([150, 255, 255])

kernel = np.ones((500, 500), np.float32)/250000 # defining kernel used to smooth frame

app = Flask(__name__) # creating a flask app

@app.route('/')
def index():
    """
    Asynchronous function to render the html template
    """
    return render_template('index.html')

def get_frame():
    """
    Returns filtered frame from camera input
    """
    camera_port = 0 # keep at 0 unless multiple cameras are connected
    camera = cv2.VideoCapture(camera_port) # this makes a web cam object

    while True:
        _, frame = camera.read() # this grabs the frame from the camera
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # converting the frame into hsv format for filtering

        red_mask = cv2.inRange(hsv, lower_red, upper_red) # filtering the frame for red
        red_smooth = cv2.filter2D(red_mask, -1, kernel) # smoothing the mask with a large mean kernel to thicken chalk lines
        _, red_thresh = cv2.threshold(red_smooth, 25, 225, cv2.THRESH_BINARY) # thresholding the image to remove extra noise
        red_res = cv2.bitwise_and(frame, frame, mask=red_thresh) # drawing the chalk line onto a black background

        red_contours, _ = cv2.findContours(red_thresh.copy(), 1, cv2.CHAIN_APPROX_NONE) # outlining the chalk line

        if len(red_contours) > 0:
            red_c = max(red_contours, key=cv2.contourArea) # finding the largest outline in case of stray noise

            red_moments = cv2.moments(red_c) # calculating moments in order to find center point of the chalk line

            red_cx = int(red_moments['m10'] / red_moments['m00']) # x value of the center point

            red_cy = int(red_moments['m01'] / red_moments['m00']) # y value of the center point

            cv2.circle(red_res, (red_cx, red_cy), 10, 10, thickness=10) # drawing the center of the chalk line
            cv2.circle(red_res, (1512, red_cy), 10, 10, thickness=10)
            cv2.circle(red_res, (1512, 4032), 10, 10, thickness=10)

            cv2.line(red_res, (1512, red_cy), (1512, 4032), (0, 0, 255), 10) # drawing a line of where the robot is facing
            cv2.line(red_res, (red_cx, red_cy), (1512, 4032), (0, 255, 0), 10) # drawing a line from the robot to the center of the chalk

            cv2.putText(red_res, # calculating and writing the angle between the robot and the chalk center
                        "Theta = {}".format((180 / np.pi) * np.arctan(abs(red_cx - 1512) / abs(3024 - red_cy))),
                        (250, 250),
                        cv2.FONT_HERSHEY_PLAIN, 10, (255, 255, 255), thickness=25)

            cv2.drawContours(red_res, red_contours, -1, (255, 0, 0), 10) # drawing the outline of the chalk line

        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue) #
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


        red_encode = cv2.imencode('.jpg', red_res)[1] # converting the image back into a jpg
        red_string_data = red_encode.tostring() # converting the image to a string
        blue_encode = cv2.imencode('.jpg', blue_res)[1]
        blue_string_data = blue_encode.tostring()
        yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+blue_string_data+b'\r\n') # returning the processed image and then starting the loop again
        
@app.route('/image')
def image():
    """
    A function to send the processed image to the flask server
    """
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='localhost', debug=True, threaded=True) # running the app
    
