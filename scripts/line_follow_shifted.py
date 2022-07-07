#!/usr/bin/env python3
# https://www.youtube.com/watch?v=AbqErp4ZGgU
# https://medium.com/@mrhwick/simple-lane-detection-with-opencv-bfeb6ae54ec0
# https://towardsdatascience.com/finding-driving-lane-line-live-with-opencv-f17c266f15db


import cv2
import math
import rospy
import numpy as np
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from dynamic_reconfigure.server import Server
from fictitious_line_pkg.cfg import LineFollowConfig

# global variables
yaw_rate = Float32()

################### callback ###################

def dynamic_reconfigure_callback(config, level):
    global RC
    RC = config
    return config

def image_callback(camera_image):

    try:
        # convert camera_image into an opencv-compatible image
        cv_image = CvBridge().imgmsg_to_cv2(camera_image, "bgr8")
    except CvBridgeError:
        print(CvBridgeError)
        
    width = cv_image.shape[0]
    height = cv_image.shape[1]

    # resize the image
    cv_image = cv2.resize(cv_image, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)

    filtered_image = apply_filters(cv_image)
    filtered_image_with_roi = get_region_of_interest(filtered_image)

    ###################################################################################################
    lines = cv2.HoughLinesP(filtered_image_with_roi, rho=6, theta=(np.pi / 180),
                            threshold=15, lines=np.array([]), minLineLength=20, maxLineGap=30)

    right_line_x = []
    right_line_y = []

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1)
            if (math.fabs(slope) < 0.5):
                continue
            if slope > 0:
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])
    else:
        # print("\n\n LINES IS EMPTY \n\n")
        pass

    # just below the horizon
    min_y = filtered_image_with_roi.shape[0] * (3 / 5)
    print(f"Minimum y coordinate:", min_y)
    # the bottom of the image
    max_y = filtered_image_with_roi.shape[0]
    print(f"Maximum y coordinate:",max_y)
    # middle_line_y_start = min_y
    # middle_line_y_end = max_y

    poly_right = 0

    if len(right_line_x) != 0 and len(right_line_y) != 0:
        poly_right = np.poly1d(np.polyfit(right_line_y, right_line_x, deg=1))
        right_line_x_start = int(poly_right(max_y))
        right_line_x_end = int(poly_right(min_y))
    else:
        right_line_x_start = int(width/3)
        right_line_x_end = int(width/4)

    right_lines= [[ [right_line_x_start, max_y, right_line_x_end, min_y] ]]

    line_image = np.copy(cv_image) * 0

    for line in right_lines:
        for x1, y1, x2, y2 in line:
            print(x1, y1, x2, y2)
            cv2.line(line_image, (x1, y1), (x2, int(y2)), (255, 0, 0), 10)
            cv2.line(line_image, (x1-300, y1), (x2-300, int(y2)), (0, 0, 255), 10)

    #lines_edges = cv2.addWeighted(cv_image, 0.8, line_image, 1, 0)
    middle_line_edge = cv2.addWeighted(cv_image, 0.8, line_image, 1, 0)

    # convert the image to grayscale
    hsv = cv2.cvtColor(middle_line_edge, cv2.COLOR_BGR2HSV)
    thresh1 = cv2.inRange(hsv,np.array((0, 150, 150)), np.array((20, 255, 255)))
    thresh2 = cv2.inRange(hsv,np.array((150, 150, 150)), np.array((180, 255, 255)))
    thresh =  cv2.bitwise_or(thresh1, thresh2)

    # find the contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # initialize the variables for computing the centroid and finding the largest contour
    cx = 0
    cy = 0
    max_contour = []

    if len(contours) != 0:
        # find the largest contour by its area
        max_contour = max(contours, key = cv2.contourArea)

        # https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
        M = cv2.moments(max_contour)

        if M["m00"] != 0:
            # compute the x and y coordinates of the centroid
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
    else:
        # rospy.loginfo(f"empty contours: {contours}")
        pass

    try:
        # draw the obtained contour lines (or the set of coordinates forming a line) on the original image
        cv2.drawContours(middle_line_edge, max_contour, -1, (0, 255, 0), 10)
    except UnboundLocalError:
        rospy.loginfo("max contour not found")

    # draw a circle at centroid (https://www.geeksforgeeks.org/python-opencv-cv2-circle-method)
    cv2.circle(middle_line_edge, (cx, cy), 8, (180, 0, 0), -1)  # -1 fill the circle

    ###################################################################################################

    # get the dimension of the image
    height, width = cv_image.shape[0], cv_image.shape[1]

    # offset the x position of the vehicle to follow the lane
    # cx -= 170
    pub_yaw_rate(cv_image, cx, cy, height, width)

    cv_image1 = get_region_of_interest(cv_image)

    #cv2.imshow("CV Image", cv_image)
    cv2.imshow("Filtered Image with ROI", filtered_image_with_roi)
    #cv2.imshow("Image with ROI", cv_image1)
    #cv2.imshow("Hough Lines", lines_edges)
    cv2.imshow("Middle Hough Lines", middle_line_edge)
    cv2.waitKey(3)
    #rate.sleep()

################### filters ###################

def apply_white_balance(cv_image):

    # convert image to the LAB color space
    lab_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)

    average_a = np.average(lab_image[:,:,1])
    average_b = np.average(lab_image[:,:,2])

    lab_image[:,:,1] = lab_image[:,:,1] - ((average_a - 128) * (lab_image[:,:,0] / 255.0) * 1.1)
    lab_image[:,:,2] = lab_image[:,:,2] - ((average_b - 128) * (lab_image[:,:,0] / 255.0) * 1.1)

    return cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

def apply_filters(cv_image):

    balanced_image = apply_white_balance(cv_image)

    # one more time
    balanced_image = apply_white_balance(balanced_image)

    # convert image to the HLS color space
    hls_image = cv2.cvtColor(balanced_image, cv2.COLOR_BGR2HLS)

    # lower and upper bounds for the color white
    lower_bounds = np.uint8([0, RC.light_l, 0])
    upper_bounds = np.uint8([255, 255, 255])
    white_detection_mask = cv2.inRange(hls_image, lower_bounds, upper_bounds)

    # lower and upper bounds for the color yellow
    # lower_bounds = np.uint8([10, 0, 100])
    # upper_bounds = np.uint8([40, 255, 255])
    # yellow_detection_mask = cv2.inRange(hls_image, lower_bounds, upper_bounds)

    # combine the masks
    # white_or_yellow_mask = cv2.bitwise_or(white_detection_mask, yellow_mask)
    balanced_image_with_mask =  cv2.bitwise_and(balanced_image, balanced_image, mask = white_detection_mask)

    # convert image to grayscale
    gray_balanced_image_with_mask = cv2.cvtColor(balanced_image_with_mask, cv2.COLOR_BGR2GRAY)

    # smooth out the image
    kernel = np.ones((5, 5), np.float32) / 25
    smoothed_gray_image = cv2.filter2D(gray_balanced_image_with_mask, -1, kernel)

    # find and return the edges in in smoothed image
    return cv2.Canny(smoothed_gray_image, 200, 255)

def get_region_of_interest(image):

    width = image.shape[1]
    height = image.shape[0]

    width = width / 8
    height = height / 8

    roi = np.array([[

                       [width*4, height*8],
                       [width*4, height*4],
                       [width*5, height*4],
                       [width*6, height*5],
                       [width*7, height*6],
                       [width*8, height*8]

                   ]], dtype = np.int32)

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, roi, 255)

    # return the image with the region of interest
    return cv2.bitwise_and(image, mask)

def perspective_warp(image,
                     destination_size=(1280, 720),
                     source=np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)]),
                     destination=np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])):

    image_size = np.float32([(image.shape[1], image.shape[0])])
    source = source * image_size

    # For destination points, I'm arbitrarily choosing some points to be a nice fit for displaying
    # our warped result again, not exact, but close enough for our purposes
    destination = destination * np.float32(destination_size)

    # given source and destination points, calculate the perspective transform matrix
    perspective_transform_matrix = cv2.getPerspectiveTransform(source, destination)

    # return the warped image
    return cv2.warpPerspective(image, perspective_transform_matrix, destination_size)


################### algorithms ###################

def pub_yaw_rate(cv_image, cx, cy, width, height):

    # compute the coordinates for the center the vehicle's camera view
    camera_center_y = (height / 4)*3
    camera_center_x = (width / 2)

    # compute the difference between the x and y coordinates of the centroid and the vehicle's camera center
    center_error = cx - camera_center_x

    # In simulation:
    #       less than 3.0 - deviates a little inward when turning
    #                 3.0 - follows the line exactly
    #       more than 3.0 - deviates a little outward when turning
    correction = RC.offset_yaw * camera_center_y

    # compute the yaw rate proportion to the difference between centroid and camera center
    angular_z = float(center_error / correction)

    if cx > camera_center_x:
        # angular.z is negative; left turn
        yaw_rate.data = -abs(angular_z)
    elif cx < camera_center_x:
        # angular.z is positive; right turn
        yaw_rate.data = abs(angular_z)
    else:
        # keep going straight
        yaw_rate.data = 0.0

    yaw_rate_pub.publish(yaw_rate)

    return

################### main ###################

if __name__ == "__main__":

    rospy.init_node("follow_line", anonymous=True)

    rospy.Subscriber("/camera/image_raw", Image, image_callback)

    #rate = rospy.Rate(10)
    yaw_rate_pub = rospy.Publisher("yaw_rate", Float32, queue_size=1)

    dynamic_reconfigure_server = Server(LineFollowConfig, dynamic_reconfigure_callback)

    try:
      rospy.spin()
    except rospy.ROSInterruptException:
      pass
