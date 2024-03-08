import cv2
import numpy as np


class KalmanFilter:
    def __init__(self, init_x, init_y):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.predict(init_x, init_y)

    def predict(self, x, y):
        measured_value = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measured_value)
        predicted = self.kf.predict()
        x, y = int(predicted[0]), int(predicted[1])
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        return x, y