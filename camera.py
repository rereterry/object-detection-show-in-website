from threading import Thread, Lock
import cv2
import numpy as np
import time
from queue import Queue

class CameraStream(object):
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture('IP or webcam or video')
        # self.stream = cv2.VideoCapture(0)
        # self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'));
        # self.stream.set(3,1920) #set frame width
        # self.stream.set(4,1080) #set frame height
        # self.stream.set(cv2.CAP_PROP_FPS, 16) #adjusting fps to 5
        # self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 4);
        # print("FPS的數值 ", self.stream.get(cv2.CAP_PROP_FPS), "緩衝的數量 ", self.stream.get(cv2.CAP_PROP_BUFFERSIZE))
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()
        self.count = 1


    def start(self):
        if self.started:
            print("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            if self.stream.isOpened():
                (grabbed, frame) = self.stream.read()
                self.read_lock.acquire()
                self.grabbed, self.frame = grabbed, frame
                self.read_lock.release()
            else:
                print("the IP cam has be closed")
        self.stream.release()
    

    # 讀取我們接收到的frame
    def read(self):
        frame = None
        self.read_lock.acquire()
        if self.frame is None:
            print ('you read a None, restart the VideoCapture')
            # self.stream.release()
            self.stream = cv2.VideoCapture('IP or webcam or video')
        else:
            
            frame = self.frame.copy()
        self.read_lock.release()
        return frame


    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stream.release()




