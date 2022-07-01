#!/usr/bin/env python3
# algorithm for the ACTor vehicle to follow a white line with yellow blob detection.

import cv2
import rospy
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32, Empty
from cv_bridge import CvBridge, CvBridgeError
from dynamic_reconfigure.server import Server
from fictitious_line_pkg.cfg import FollowLineConfig     # ../cfg/FollowLine.cfg

# global variables
vel_msg = Twist()
start_stop_msg = Bool()
steer_error_msg = Float32()

drive = False
end_test = False
y_detected = False

n_laps = 0
total_frames = 0
sum_steer_error = 0

# for the vehicle, the message must start at 0.0
vel_msg.linear.x = 0.0
vel_msg.angular.z = 0.0

################### callback ###################

def yellow_callback(msg):
    global y_detected, n_laps, end_test, sum_steer_error, total_frames
    if not y_detected and msg.data:
        n_laps += 1
        steer_error_msg.data = sum_steer_error / total_frames
        steer_error_pub.publish(steer_error_msg)
        total_frames = 0
        sum_steer_error = 0
        y_detected = True
        if n_laps == 2:
            end_test = True # see image_callback()
    elif y_detected and not msg.data:
        y_detected = False
    return

def dynamic_reconfigure_callback(config, level):
    global RC
    RC = config
    if RC.enable_drive:
        start_stop_msg.data = True
        start_stop_pub.publish(start_stop_msg)
    return config

def image_callback(ros_image):
    global end_test

    if end_test:
        stop_vehicle()
        return

    try:
        # convert ros_image into an opencv-compatible image
        cv_image = CvBridge().imgmsg_to_cv2(ros_image, "bgr8")
    except CvBridgeError:
        print(CvBridgeError)

    # get the properties of the image
    (width, height, _) = cv_image.shape

    # crop the camera image
    third_width = int(width / 3)
    quarter_height = int(height / 4)
    cv_image = cv_image[third_width:, quarter_height:(quarter_height * 3)]

    ################### mask ###################
    # convert to grayscale
    # gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # apply filters
    # mask = cv2.medianBlur(gray_image, 15)

    # mask = cv2.Canny(mask, RC.thresh, 255, 3)

    # element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # mask = cv2.dilate(mask, element)

    # binary mask
    # input image, threshol_value, max value in image, threshold type
    # _, mask = cv2.threshold(gray_image, RC.thresh, 255, cv2.THRESH_BINARY)

    #----------------- HSV method ----------------------------
    bgr_image = find_white_balance(cv_image)
    cv2.imshow("White Balance", bgr_image)
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    lower_bounds = (RC.hue_l, RC.sat_l, RC.val_l)  # lower bounds of H, S, V for the target color
    upper_bounds = (RC.hue_h, RC.sat_h, RC.val_h)  # upper bounds of H, S, V for the target color

    mask = cv2.inRange(hsv_image, lower_bounds, upper_bounds)
    ############################################

    # blur_kernel = 1 # must be odd, 1, 3, 5, 7 ...
    # bw_img = cv2.medianBlur(mask, blur_kernel)

    #find contours in the binary (BW) image
    contours, _ = cv2.findContours (mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # initialize the variables for computing the centroid and finding the largest contour
    cx = 0
    cy = 0
    max_contour = []

    if len(contours) != 0:
        # find the largest contour by its area
        max_contour = max(contours, key = cv2.contourArea)
        M = cv2.moments(max_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
    else:
        pass
        # print(f"empty contours: {contours}")

    try: # draw the obtained contour lines (or the set of coordinates forming a line) on the original image
        # https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
        cv2.drawContours(cv_image, max_contour, -1, (0, 0, 255), 10)
    except UnboundLocalError:
        print("max contour not found")

    # draw a circle at centroid (https://www.geeksforgeeks.org/python-opencv-cv2-circle-method)
    cv2.circle(cv_image, (cx, cy), 8, (180, 0, 0), -1)  # -1 fill the circle

    # offset the x position of the robot to follow the lane
    # cx -= 170

    # drive_to_follow_line(mask, cx, cy, width, height)
    drive_to_follow_line(mask, cx, cy, width, height)

    cv2.imshow("CV Image", cv_image)
    cv2.imshow("Binary Image", mask)
    cv2.waitKey(3)

################### algorithms ###################

def drive_to_follow_line(image, cx, cy, width, height):
    global sum_steer_error, total_frames

    if RC.enable_drive:
        vel_msg.linear.x = RC.speed

        # get the center position of the car"s camera view
        camera_center_x = (width / 2) # corresponds to x
        camera_center_y = (height / 2) # corresponds to y


        # compute the difference between vertical center of the centroid and car"s camera view
        steer_error = abs(cx - camera_center_x)

        # In simulation:
        #       <3 - deviates a little inward when turning
        #       3 - follows the line exactly
        #       3> - deviates a little outward when turning
        correction = RC.offset_turn * camera_center_y

        # compute the angular velocity based on the speed of the robot
        angular_vel = float(steer_error / correction)

        if cx > camera_center_x:
            # angular.z is negative
            vel_msg.angular.z = -abs(angular_vel)
        elif cx < camera_center_x:
            # angular.z is positive
            vel_msg.angular.z = abs(angular_vel)
        else:
            vel_msg.angular.z = 0

        total_frames += 1
        sum_steer_error += steer_error
        # NOTE: publishing in drive_controller() instead
        # cmd_vel_pub.publish(vel_msg)
    else:
        stop_vehicle()

def drive_to_follow_line_2(cv_image, cx, cy, width, height):
    global sum_steer_error, total_frames

    mid = height / 2

    if RC.enable_drive:
        vel_msg.linear.x = RC.speed
        steer_err = mid-cx;
        vel_msg.angular.z = 0.01*steer_err
        sum_steer_error += abs(steer_err)
        total_frames += 1
    else: # drive disabled
        stop_vehicle()
    return

################### helper functions ###################

def stop_vehicle():
    vel_msg.linear.x = 0.0
    vel_msg.angular.z = 0.0
    cmd_vel_pub.publish(vel_msg)
    return

def find_white_balance(cv_image):
    result = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
    average_a = np.average(result[:,:,1])
    average_b = np.average(result[:,:,2])
    result[:,:,1] = result[:,:,1] - ((average_a - 128) * (result[:,:,0] / 255.0) * 1.1)
    result[:,:,2] = result[:,:,2] - ((average_b - 128) * (result[:,:,0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


def drive_controller():
    rate = rospy.Rate(20)
    enable_empty_msg = Empty() # required for the vehicle

    # this variable controls whether or not to wait to seconds before moving the vehicle
    use_stop_time = True
    time_start = rospy.Time.now()
    time_stop = 2 # in seconds


    while(not rospy.is_shutdown()):
        if use_stop_time:
            time_elapsed = (rospy.Time.now() - time_start).to_sec()
            if (time_elapsed < time_stop):
                vel_msg.linear.x = 0.0

        enable_drive_pub.publish(enable_empty_msg)
        cmd_vel_pub.publish(vel_msg)
        rate.sleep()
    return

################### main ###################

if __name__ == "__main__":
    rospy.init_node("follow_line", anonymous=True)
    rospy.Subscriber("/camera/image_raw", Image, image_callback)
    rospy.Subscriber("yellow_detected", Bool, yellow_callback)

    start_stop_pub = rospy.Publisher("start_test", Bool, queue_size=1)
    steer_error_pub = rospy.Publisher("steer_err", Float32, queue_size=1)
    cmd_vel_pub = rospy.Publisher("/vehicle/cmd_vel", Twist, queue_size=1)
    enable_drive_pub = rospy.Publisher("/vehicle/enable", Empty, queue_size=1)

    srv = Server(FollowLineConfig, dynamic_reconfigure_callback)

    drive_controller()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
