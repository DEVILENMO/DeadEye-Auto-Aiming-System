import cv2
import numpy as np


class KalmanFilter:
    """逐点 2D 卡尔曼滤波（旧接口，保留兼容）；新代码请使用 BoxKalmanFilter。"""

    def __init__(self, init_x, init_y, process_noise_cov=1e-4, measurement_noise_cov=1e-1):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise_cov
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise_cov
        self.kf.statePost = np.array([[init_x], [init_y], [0], [0]], np.float32)
        self.kf.statePre = np.array([[init_x], [init_y], [0], [0]], np.float32)

    def predict(self):
        predicted = self.kf.predict()
        return predicted[0, 0], predicted[1, 0]

    def correct(self, x, y):
        measured_value = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measured_value)
        corrected = self.kf.statePost
        return corrected[0, 0], corrected[1, 0]


class BoxKalmanFilter:
    """
    检测框卡尔曼滤波：等速模型，状态 [cx, cy, w, h, vx, vy, vw, vh]，
    观测 [cx, cy, w, h]（中心 + 尺寸，避免逐角点滤波在运动下扭曲长宽比）。

    过程噪声大 -> 对机动目标响应快；测量噪声大 -> 平滑强。
    """

    def __init__(self, cx, cy, w, h, *, process_noise=1e-2, measurement_noise=5.0):
        self.x = np.zeros(8, dtype=np.float64)
        self.x[:4] = (cx, cy, w, h)

        self.F = np.eye(8)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = self.F[3, 7] = 1.0
        self.H = np.zeros((4, 8))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0
        self.Q = np.eye(8) * process_noise
        self.Q[4:, 4:] *= 10.0  # 速度项更不确定
        self.R = np.eye(4) * measurement_noise

        self.P = np.eye(8) * 10.0
        self.P[4:, 4:] *= 100.0

    def predict(self) -> np.ndarray:
        """推进一步并返回先验框 (cx, cy, w, h)。"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4].copy()

    def update(self, cx: float, cy: float, w: float, h: float) -> np.ndarray:
        """用观测量修正并返回后验框 (cx, cy, w, h)。"""
        z = np.array([cx, cy, w, h], dtype=np.float64)
        y = z - self.H @ self.x
        s = self.H @ self.P @ self.H.T + self.R
        k = self.P @ self.H.T @ np.linalg.inv(s)
        self.x = self.x + k @ y
        self.P = (np.eye(8) - k @ self.H) @ self.P
        return self.x[:4].copy()

    @property
    def box(self) -> tuple[float, float, float, float]:
        """当前估计的框 (x1, y1, x2, y2)。"""
        cx, cy, w, h = self.x[:4]
        return cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5
