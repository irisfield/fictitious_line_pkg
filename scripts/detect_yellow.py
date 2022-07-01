#!/usr/bin/env python3

# yellow line detection node

import cv2
import rospy
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from dynamic_reconfigure.server import Server
from fictitious_line_pkg.cfg import DetectYellowConfig

# global variables
yellow_detected = Bool()

################### callback ###################

def dynamic_reconfigure_callback(config, level):
    global RC
    RC = config
    return config

def image_callback(camera_image):
    global height, width
    try:
        cv_image = CvBridge().imgmsg_to_cv2(camera_image, "bgr8")
    except CvBridgeError:
        print(CvBridgeError)

    hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

    lower_bounds = (RC.hue_l, RC.sat_l, RC.val_l) # lower bounds of h, s, v for the target color
    upper_bounds = (RC.hue_h, RC.sat_h, RC.val_h) # upper bounds of h, s, v for the target color
    image_mask = cv2.inRange(hsv_image, lower_bounds, upper_bounds)


    # find contours in the binary (black and white) image
    contours, _ = cv2.findContours (image_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # initialize the variables for computing the centroid and finding the largest contour
    max_area = 0
    max_contour = []

    if len(contours) != 0:
        # find the largest contour by its area
        max_contour = max(contours, key = cv2.contourArea)
        max_area = cv2.contourArea(max_contour)
    else:
        yellow_detected.data = False
        yellow_detected_pub.publish(yellow_detected)
        return

    try:
        # draw the obtained contour lines (or the set of coordinates forming a line) on the original image
        # https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
        cv2.drawContours(cv_image, max_contour, -1, (0, 0, 255), 5)
    except UnboundLocalError:
        print("max contour not found")

    if max_area > 100:
        yellow_detected.data = True
    else:
        yellow_detected.data = False

    yellow_detected_pub.publish(yellow_detected)

    # cv2.imshow("Yellow Line Detected", image_mask)
    cv2.waitKey(3)

################### main ###################

if __name__ == "__main__":
    rospy.init_node("detect_yellow", anonymous=True)

    rospy.Subscriber("/camera/image_raw", Image, image_callback)

    yellow_detected_pub = rospy.Publisher("yellow_detected", Bool, queue_size=1)

    dynamic_reconfigure_server = Server(DetectYellowConfig, dynamic_reconfigure_callback)

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
