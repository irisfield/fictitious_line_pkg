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

    # get the dimension of the image
    height, width = cv_image.shape[0], cv_image.shape[1]

    # resize the image
    cv_image = cv2.resize(cv_image, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)

    filtered_image = apply_filters(cv_image)
    filtered_roi_image = get_region_of_interest(filtered_image)

    lines = cv2.HoughLinesP(filtered_roi_image,
                            rho=RC.rho,
                            theta=(np.pi / 180),
                            threshold=RC.threshold,
                            lines=np.array([]),
                            minLineLength=RC.min_line_length,
                            maxLineGap=RC.max_line_gap
                           )

    left_dotted_line_x = []
    left_dotted_line_y = []

    # take only the right slope
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1)
            # positive slope is right
            if slope < 0:
                left_dotted_line_x.extend([x1, x2])
                left_dotted_line_y.extend([y1, y2])

    # get the dimensions of the image
    width = cv_image.shape[1]
    height = cv_image.shape[0]

    # starting end ending point of the line
    left_dotted_start_point_y = height * (1)
    left_dotted_end_point_y = height * (3/5)

    poly_left_dotted = 0

    if len(left_dotted_line_x) != 0 and len(left_dotted_line_y) != 0:
        poly_left_dotted = np.poly1d(np.polyfit(left_dotted_line_y, left_dotted_line_x, deg=1))
        left_dotted_start_point_x = int(poly_left_dotted(left_dotted_start_point_y))
        left_dotted_end_point_x = int(poly_left_dotted(left_dotted_end_point_y))
    else:
        left_dotted_start_point_x = int(width)
        left_dotted_end_point_x = int(width/1.2)

    # the coordinates for the line (x, y)
    left_dotted_lines = [[
                    [left_dotted_start_point_x, left_dotted_start_point_y,
                    left_dotted_end_point_x, left_dotted_end_point_y]
                  ]]

    # blank image to draw the lines
    line_image = np.copy(cv_image) * 0

    for line in left_dotted_lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, int(y1)), (x2, int(y2)), (255, 0, 0), 10)
            cv2.line(line_image, (x1 + 240, int(y1)), (x2 + 240 , int(y2)), (0, 0, 255), 10)

    hough_line_image = cv2.addWeighted(cv_image, 0.8, line_image, 1, 0)

    # convert the image to grayscale
    hsv_image = cv2.cvtColor(hough_line_image, cv2.COLOR_BGR2HSV)
    thresh1 = cv2.inRange(hsv_image,np.array((0, 150, 150)), np.array((20, 255, 255)))
    thresh2 = cv2.inRange(hsv_image,np.array((150, 150, 150)), np.array((180, 255, 255)))
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

    try:
        # draw the obtained contour lines (or the set of coordinates forming a line) on the original image
        cv2.drawContours(hough_line_image, max_contour, -1, (0, 255, 0), 10)
    except UnboundLocalError:
        rospy.loginfo("max contour not found")

    # draw a circle at centroid (https://www.geeksforgeeks.org/python-opencv-cv2-circle-method)
    cv2.circle(hough_line_image, (cx, cy), 8, (180, 0, 0), -1)  # -1 fill the circle

    # offset the x position of the robot to follow the lane
    #cx -= 170

    pub_yaw_rate(hough_line_image, cx, cy)

    # concatenate the roi images to show in a single window
    # the shape of the images must have the same length: len(image.shape)
    #filtered_roi_image_with_channel = cv2.cvtColor(filtered_roi_image, cv2.COLOR_GRAY2BGR)
    filtered_roi_image_with_channel = cv2.cvtColor(filtered_roi_image, cv2.COLOR_GRAY2BGR)
    # filtered_roi_image_with_channel = get_region_of_interest_cropped(filtered_roi_image_with_channel)
    concatenated_roi_image = cv2.vconcat([hough_line_image, filtered_roi_image_with_channel])

    # cv2.imshow("Filtered ROI Image", filtered_roi_image)
    # cv2.imshow("Middle Hough Lines", hough_line_image)
    cv2.imshow("ROI Image and Hough Lines", concatenated_roi_image)

    # show the region of interest the algorithm is using
    roi_image_cropped = get_region_of_interest_cropped(cv_image)
    cv2.imshow("ROI", roi_image_cropped)

    cv2.waitKey(3)
    rate.sleep()

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

    # one more time, comment if it is overcast
    # balanced_image = apply_white_balance(balanced_image)

    # convert image to the HLS color space
    hls_image = cv2.cvtColor(balanced_image, cv2.COLOR_BGR2HLS)

    # lower and upper bounds for the color white
    lower_bounds = np.uint8([0, RC.light_low, 0])
    upper_bounds = np.uint8([255, 255, 255])
    white_detection_mask = cv2.inRange(hls_image, lower_bounds, upper_bounds)

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

                       [width * 4, height * 8],
                       [width * 4, height * 4],
                       [width * 2, height * 4],
                       [width, height * 6],
                       [0, height * 8]

                   ]], dtype = np.int32)

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, roi, (255, 255, 255))

    # return the image with the region of interest
    return cv2.bitwise_and(image, mask)

def get_region_of_interest_cropped(cv_image):
    roi_image = get_region_of_interest(cv_image)

    # crop the black edges and return cropped image
    y_nonzero, x_nonzero, _ = np.nonzero(roi_image)
    return roi_image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

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

def pub_yaw_rate(cv_image, cx, cy):

    # get the dimensions of the image
    width = cv_image.shape[1]
    height = cv_image.shape[0]

    # compute the coordinates for the center the vehicle's camera view
    camera_center_y = (height / 2)
    camera_center_x = (width / 2)

    # compute the difference between the x and y coordinates of the centroid and the vehicle's camera center
    center_error = cx - camera_center_x

    # In simulation:
    #       less than 3.0 - deviates a little inward when turning
    #                 3.0 - follows the line exactly
    #       more than 3.0 - deviates a little outward when turning
    correction = 2.0 * camera_center_y

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

    rate = rospy.Rate(25)
    yaw_rate_pub = rospy.Publisher("/yaw_rate", Float32, queue_size=1)

    dynamic_reconfigure_server = Server(LineFollowConfig, dynamic_reconfigure_callback)

    try:
      rospy.spin()
    except rospy.ROSInterruptException:
      pass
