import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

def nothing(x):
    pass

def show_pixel_coords(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked pixel: x={x}, y={y}")

class HSVCalibrator(Node):
    def __init__(self):
        print("STEP A: init starting")
        super().__init__('hsv_calibrator')
        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.callback, 10)

        cv2.namedWindow("Trackbars")
        print("STEP B: window created")
        cv2.createTrackbar("H_min", "Trackbars", 0, 179, nothing)
        cv2.createTrackbar("H_max", "Trackbars", 179, 179, nothing)
        cv2.createTrackbar("S_min", "Trackbars", 0, 255, nothing)
        cv2.createTrackbar("S_max", "Trackbars", 255, 255, nothing)
        cv2.createTrackbar("V_min", "Trackbars", 0, 255, nothing)
        cv2.createTrackbar("V_max", "Trackbars", 255, 255, nothing)
        print("STEP C: trackbars created")

        self.x_min = 701
        self.y_min = 196
        self.x_max = 1003
        self.y_max = 422


    def callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        #frame = frame[self.y_min:self.y_max, self.x_min:self.x_max]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h_min = cv2.getTrackbarPos("H_min", "Trackbars")
        h_max = cv2.getTrackbarPos("H_max", "Trackbars")
        s_min = cv2.getTrackbarPos("S_min", "Trackbars")
        s_max = cv2.getTrackbarPos("S_max", "Trackbars")
        v_min = cv2.getTrackbarPos("V_min", "Trackbars")
        v_max = cv2.getTrackbarPos("V_max", "Trackbars")

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower, upper)
        

        kernel = np.ones((5, 5), np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:
                continue

            rect = cv2.minAreaRect(contour)
            (cx, cy), (w, h), angle = rect
            if w < h:
                angle += 90.0

            print(f"Block at pixel ({cx:.1f}, {cy:.1f}),  width ({w:.1f} mm, {h:.1f} mm), (angle={angle:.1f} deg")
    
        result = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)
        cv2.imshow("Result", result)
        cv2.setMouseCallback("Frame", show_pixel_coords)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            print(f"lower = {lower.tolist()}, upper = {upper.tolist()}")

        

def main():
    rclpy.init()
    node = HSVCalibrator()
    rclpy.spin(node)

if __name__ == '__main__':
    main()