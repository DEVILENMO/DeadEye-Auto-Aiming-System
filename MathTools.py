import cv2
import numpy as np


class KalmanFilter:
    def __init__(self, init_x, init_y, process_noise_cov=1e-4, measurement_noise_cov=1e-1):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise_cov
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise_cov
        self.kf.statePost = np.array([[init_x], [init_y], [0], [0]], np.float32)

    def predict(self, x, y):
        measured_value = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measured_value)
        predicted = self.kf.predict()
        x, y = max(0, int(predicted[0])), max(0, int(predicted[1]))
        return x, y
