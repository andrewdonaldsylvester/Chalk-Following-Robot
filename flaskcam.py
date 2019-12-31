from flask import Flask, render_template, Response
import cv2
import numpy as np
from filters import red_filter, blue_filter
from picameracapture import get_frame

app = Flask(__name__) # creating a flask app

@app.route('/')
def index():
    """
    Asynchronous function to render the html template
    """
    return render_template('index.html')

def filter_frame(frame):
    """
    Returns filtered frame
    """

    while True:
        red_filtered = red_filter(frame)

        red_encode = cv2.imencode('.jpg', red_filtered)[1] # converting the image back into a jpg

        red_string_data = red_encode.tostring() # converting the image to a string

        blue_filtered = blue_filter(frame)

        blue_encode = cv2.imencode('.jpg', blue_filtered)[1]

        blue_string_data = blue_encode.tostring()

        yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+red_string_data+b'\r\n') # returning the processed image and then starting the loop again

@app.route('/image')
def image():
    """
    A function to send the processed image to the flask server
    """
    return Response(filter_frame(get_frame()), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='localhost', debug=True, threaded=True) # running the app
