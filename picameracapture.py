from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

camera = PiCamera()
rawCapture = PiRGBArray(camera, size=(640, 480))

time.sleep(0.1)

def get_frame():
    camera.capture(rawCapture, format="bgr")
    image = rawCapture.array

    return image
