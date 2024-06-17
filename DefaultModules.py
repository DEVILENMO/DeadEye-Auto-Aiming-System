# coding: utf-8
# cython: language_level=3
import time
from ctypes import windll

from pynput import mouse

from BaseModules import *
from ScreenShotHelper import *
from TensorRTEngine import *


class SimpleScreenShotCamera(BaseCamera):
    def __init__(self, view_range):
        super(SimpleScreenShotCamera).__init__()
        # resolution
        user32 = windll.user32
        user32.SetProcessDPIAware()
        self.ori_resolution_x, self.ori_resolution_y = (
            win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN),
            win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN))
        print('Screen resolution:', self.ori_resolution_x, 'x', self.ori_resolution_y)
        self.rel_resolution_x, self.rel_resolution_y = dxcam.output_res()[0]
        print('Scaled screen resolution:', self.rel_resolution_x, 'x', self.rel_resolution_y)
        self.screen_shot_camera = ScreenShotHelper(view_range[0], view_range[1], ScreenShotHelper.CameraType.DXCAM)

    def get_image(self):
        image = self.screen_shot_camera.capture_screen_shot()
        if image is not None and self.screen_shot_camera.image_color_mode == ScreenShotHelper.ImageColorMode.BGR:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image



class YoloDetector(DetectModule):
    def __init__(self, model: str):
        super(YoloDetector).__init__()
        self.model = None
        self.model_type = None
        self.load_model(model)

    def on_exit(self):
        if self.model_type == 'trt':
            self.model.on_exit()

    def load_model(self, model: str):
        if '.pt' in model:
            self.model = YOLO(model)
            self.model_type = 'pt'
        else:
            print('Loading TensorRT engine...')
            self.model = Yolov8TensorRTEngine(model)
            self.model_type = 'trt'

    def target_detect(self, img) -> list:
        if self.model is None:
            raise RuntimeError("Model is not loaded. Please call load_model() first.")

        results = []
        if self.model_type == 'pt':
            # h, w = img.shape[:2]
            result = self.model(img, verbose=False, half=True, iou=0.8, conf=0.75)
            detections = result[0]

            # 提取检测结果中的边界框、类别标签和置信度
            boxes = detections.boxes.xyxy.cpu().numpy()  # 边界框坐标
            labels = detections.boxes.cls.cpu().numpy().astype(int)  # 类别标签
            # scores = detections.boxes.conf.cpu().numpy()  # 置信度

            # 在图像上绘制边界框
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box
                class_name = detections.names[label]  # 获取类别名称
                result = [class_name, (int(x1), int(y1)), (int(x2), int(y2))]
                results.append(result)
                # 绘制边界框
                # cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # 绘制类别标签
                # cv2.putText(img, class_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            results = self.model.inference(img, conf=0.75, end2end=True)

        # 显示结果图像
        # cv2.imshow("Detection Results", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return results


class SimpleMouseController(MouseControlModule):
    def __init__(self):
        # mouse controller
        super(SimpleMouseController).__init__()
        self.mc = mouse.Controller()

    def click_left_button(self):
        # ToDo: You are required to implement a mouse click function yourself
        # Warning: You are not allowed to use this project for any unethical or legal behavior under any circumstances
        print('You are required to implement a mouse click function yourself')

    def move_mouse(self, x: int, y: int):
        # ToDo: You are required to implement a mouse move function yourself
        # Warning: You are not allowed to use this project for any unethical or legal behavior under any circumstances
        print('You are required to implement a mouse move function yourself')


DEFAULT_P = 0.5
DEFEULT_I = 0.2
DEFAULT_D = 0.05
DEFAULT_R = 0.5


class DeadEyeAutoAimingModule(AutoAimModule):
    def __init__(self, view_range):
        super(DeadEyeAutoAimingModule).__init__()
        self.view_range_start = self.calculate_view_range_start_pos(view_range)

        self.tracking_target_id = None

        self.mouse_controller = SimpleMouseController()

        # Auto aim settings
        self.auto_aim_range_x = 1.5
        self.auto_aim_range_y = 1.5
        self.aim_sensitive = 1

        # Auto shoot settings
        self.last_auto_shoot_time = time.time()

        # PID
        self.max_movement = 320  # max output movement on single direction, to avoid large movement
        # PID核心参数
        k_adjust = 1
        self.k_p = DEFAULT_P * k_adjust  # 比例系数 主要移动
        self.k_i = DEFEULT_I * k_adjust  # 积分系数 补充移动
        self.k_d = DEFAULT_D * k_adjust  # 微分系数 抑制
        # 回弹强度
        self.rebound_strength = DEFAULT_R  # 作用于积分系数，用于消除瞄准回弹效果
        # PID计算变量
        # 前一次时间，用于计算dt
        self.previous_time = None
        self.x_integral_value = 0
        self.y_integral_value = 0
        self.previous_distance_x = 0
        self.previous_distance_y = 0

    def auto_shoot(self, target_list: list[Target]) -> None:
        """
        DEMO, showing how to implement auto shoot.
        示例程序，展示如何编写自动扳机。
        """
        # 自动扳机
        t = time.time()
        if t - self.last_auto_shoot_time < 0.25:
            return

        mouseX, mouseY = pyautogui.position()
        # print('鼠标位置：', mouseX, ',', mouseY)

        for target in target_list:
            tag, left_top, right_bottom = target.label, target.left_top, target.right_bottom
            width = right_bottom[0] - left_top[0]
            height = right_bottom[1] - left_top[1]
            if left_top[0] + self.view_range_start[0] + 0.25 * width <= mouseX <= right_bottom[0] + \
                    self.view_range_start[
                        0] - 0.25 * width:
                if left_top[1] + self.view_range_start[1] <= mouseY <= right_bottom[1] + self.view_range_start[
                    1] - 0.75 * height:
                    # windll.user32.BlockInput(1)
                    self.shoot()
                    break
            if left_top[0] + self.view_range_start[0] + 0.15 * width <= mouseX <= right_bottom[0] + \
                    self.view_range_start[
                        0] - 0.15 * width:
                if left_top[1] + self.view_range_start[1] <= mouseY <= right_bottom[1] + self.view_range_start[
                    1] - 0.5 * height:
                    # windll.user32.BlockInput(1)
                    self.shoot()
                    self.last_auto_shoot_time = t
                    break
        # print('瞄准计算用时：', time.time() - t)
        return

    def set_tracking_target_id(self, to_track_target_id: [int | None]):
        self.tracking_target_id = to_track_target_id
        self.clear_pid_history()

    def auto_aim(self, target_list: list[Target]) -> tuple[int, int]:
        """
        DEMO, showing how to calculate mouse movement with PID algorithm
        示例程序，作为参考展示了如何用PID来计算鼠标移动参数。
        """
        mouseX, mouseY = pyautogui.position()

        if not len(target_list):
            self.set_tracking_target_id(None)
            return 0, 0
        rel_mouse_x = mouseX - self.view_range_start[0]
        rel_mouse_y = mouseY - self.view_range_start[1]

        aim_target = None
        if self.tracking_target_id is None:
            aim_target = self.find_closest_target(target_list, rel_mouse_x, rel_mouse_y)
            self.set_tracking_target_id(aim_target.index)
        else:
            for target in target_list:
                if target.index == self.tracking_target_id:
                    aim_target = target
                    break
            if aim_target is None:
                aim_target = self.find_closest_target(target_list, rel_mouse_x, rel_mouse_y)
                self.set_tracking_target_id(aim_target.index)

        width = aim_target.right_bottom[0] - aim_target.left_top[0]
        height = aim_target.right_bottom[1] - aim_target.left_top[1]
        distance_x = abs((aim_target.left_top[0] + aim_target.right_bottom[0]) / 2 - rel_mouse_x)
        distance_y = abs((aim_target.left_top[1] + aim_target.right_bottom[1]) / 2 - rel_mouse_y)
        if distance_x > width * self.auto_aim_range_x or distance_y > height * self.auto_aim_range_y:
            return 0, 0
        # print('最近目标：', nearest_target)
        # 移动到最近的目标
        position_fixed = (round((aim_target.left_top[0] + aim_target.right_bottom[0]) * 0.5),
                          round((aim_target.left_top[1] + aim_target.right_bottom[1]) * 0.5))
        x_r, y_r = self.calculate_mouse_movement_by_pid(position_fixed, (rel_mouse_x, rel_mouse_y))  # 计算鼠标移动
        x_r = int(x_r * self.aim_sensitive)
        y_r = int(y_r * self.aim_sensitive)

        # Mouse control or something else...
        #  Todo: 请自己实现 mouse_controller 中对鼠标的控制逻辑。 Implement mouse control logic in mouse controller.
        #  本程序仅提供参考,禁止用于包括但不限于游戏作弊、视觉瞄准机器人等可能涉嫌违法的用途中。
        #  用户在自行实现相关算法后,需要自行承担相应的法律责任。
        #  This program is for reference only and is prohibited from being used for purposes that may involve illegal
        #  activities, including but not limited to game cheating and visual aiming robots. Users shall bear
        #  corresponding legal responsibilities after implementing relevant algorithms on their own.
        self.mouse_controller.move_mouse(x_r, y_r)
        return x_r, y_r

    @staticmethod
    def find_closest_target(target_list: list[Target], mouse_pos_x: int, mouse_pos_y: int) -> Target:
        # 寻找最近目标的时候，最终距离减去目标长度宽度，这样可以避免小目标出现在大目标附近时，实际距离更远的小目标成为最近的目标
        nearest_target = min(target_list, key=lambda k: abs((k.left_top[0] + k.right_bottom[0]) / 2 - mouse_pos_x) +
                                                        abs((k.left_top[1] + k.right_bottom[1]) / 2 - mouse_pos_y) -
                                                        (k.right_bottom[0] - k.left_top[0] + k.right_bottom[1] -
                                                         k.left_top[1]))
        return nearest_target

    def shoot(self):
        #  Todo: 请自己实现 mouse_controller 中对鼠标的控制逻辑。 Implement mouse control logic in mouse controller.
        #  本程序仅提供参考,禁止用于包括但不限于游戏作弊、视觉瞄准机器人等可能涉嫌违法的用途中。
        #  用户在自行实现相关算法后,需要自行承担相应的法律责任。
        #  This program is for reference only and is prohibited from being used for purposes that may involve illegal
        #  activities, including but not limited to game cheating and visual aiming robots. Users shall bear
        #  corresponding legal responsibilities after implementing relevant algorithms on their own.
        self.mouse_controller.click_left_button()

    def set_pid_parameters(self, p=None, i=None, d=None, rebond_strength=None):
        if p is not None:
            self.k_p = p
        if i is not None:
            self.k_i = i
        if d is not None:
            self.k_d = d
        if rebond_strength is not None:
            self.rebound_strength = rebond_strength

    def clear_pid_history(self):
        self.previous_time = None
        self.x_integral_value = 0
        self.y_integral_value = 0
        self.previous_distance_x = 0
        self.previous_distance_y = 0

    def calculate_mouse_movement_by_pid(self, target_position, mouse_position):
        # PID控制算法
        if self.previous_time is not None and time.time() - self.previous_time > 0.5:
            # 消除残像
            self.clear_pid_history()

        # 绝对偏差
        current_time = time.time()
        distance_x = target_position[0] - mouse_position[0]
        if distance_x * self.previous_distance_x < 0:
            self.x_integral_value = self.x_integral_value * self.rebound_strength  # 降低积分量
        distance_y = target_position[1] - mouse_position[1]
        if distance_y * self.previous_distance_y < 0:
            self.y_integral_value = self.y_integral_value * self.rebound_strength  # 降低积分量

        # P/比例
        x_p = self.k_p * distance_x
        y_p = self.k_p * distance_y

        if self.previous_time is not None:
            # I/积分
            d_time = current_time - self.previous_time
            self.x_integral_value = self.x_integral_value + distance_x
            x_i = self.k_i * self.x_integral_value
            self.y_integral_value = self.y_integral_value + distance_y
            y_i = self.k_i * self.y_integral_value

            # D/微分
            if d_time != 0:
                derivative_x = (distance_x - self.previous_distance_x) / d_time
                x_d = self.k_d * derivative_x
                derivative_y = (distance_y - self.previous_distance_y) / d_time
                y_d = self.k_d * derivative_y
            else:
                x_d = 0
                y_d = 0
        else:
            x_i = 0
            y_i = 0
            x_d = 0
            y_d = 0

        # 结果
        x_r = x_p + x_i + x_d
        # print(x_p, x_i, x_d)
        y_r = y_p + y_i + y_d
        # print(y_p, y_i, y_d)

        # 极大值约束
        if abs(x_r) > self.max_movement:
            x_r = x_r / abs(x_r) * self.max_movement
        if abs(y_r) > self.max_movement:
            y_r = y_r / abs(y_r) * self.max_movement

        # 更新旧信息
        self.previous_time = current_time
        self.previous_distance_x = distance_x
        self.previous_distance_y = distance_y
        return x_r, y_r
