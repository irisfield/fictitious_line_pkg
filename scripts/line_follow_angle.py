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

    #gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    #filtered_image = cv2.medianBlur(gray_image, 15)
    #canny = cv2.Canny(filtered_image, 100, 200)

    # apply filters to the image
    filtered_image = apply_filters(cv_image)
    filtered_image_with_roi = get_region_of_interest(filtered_image)
    ###################################################################################################
    lines = cv2.HoughLinesP(filtered_image_with_roi, rho=6, theta=(np.pi / 180),
                            threshold=15, lines=np.array([]), minLineLength=20, maxLineGap=30)

    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1)
            if (math.fabs(slope) < 0.5):
                continue
            if slope < 0:
                # if the slope is negative, left line
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else:
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])
    else:
        # print("\n\n LINES IS EMPTY \n\n")
        pass

    # just below the horizon
    min_y = filtered_image_with_roi.shape[0] * (3 / 5)
    # the bottom of the image
    max_y = filtered_image_with_roi.shape[0]

    poly_left = 0
    poly_right = 0
    poly_middle = 0

    if len(left_line_x) != 0 and len(left_line_y) != 0:
        poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x, deg=1))
        left_line_x_start = int(poly_left(max_y))
        left_line_x_end = int(poly_left(min_y))
        # print(f"Left line x coordinate:",left_line_x)
        # print(f"Left line y coordinate:",left_line_y)
        # print(f"Poly left:",poly_left)
        # print(f"x coordinate Start point left:",left_line_x_start)
        # print(f"x coordinate End point left:",left_line_x_end)
    else:
        left_line_x_start = int(width/20)
        left_line_x_end = int(width/17)


    if len(right_line_x) != 0 and len(right_line_y) != 0:
        poly_right = np.poly1d(np.polyfit(right_line_y, right_line_x, deg=1))
        right_line_x_start = int(poly_right(max_y))
        right_line_x_end = int(poly_right(min_y))
    else:
        right_line_x_start = int(width/1)
        right_line_x_end = int(width/2)

    if poly_left != 0 and poly_right != 0:
        poly_middle = np.poly1d(np.polyfit(poly_left, poly_right, deg=1))
        middle_line_x_start = int(poly_middle(max_y))
        middle_line_x_end = int(poly_middle(max_y))
    else:
        middle_line_x_start = int(width/2)
        middle_line_x_end = int(width/3)

    side_lines= [[ [left_line_x_start, max_y, left_line_x_end, min_y], [right_line_x_start, max_y, right_line_x_end, min_y] ]]
    middle_line= [[ [middle_line_x_start, max_y, middle_line_x_end, min_y] ]]

    line_image = np.copy(cv_image) * 0
    copy_image = np.copy(cv_image) * 0

    for line in side_lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, int(y2)), (255, 0, 0), 10)

    for line in middle_line:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, int(y2)), (0, 0, 255), 10)
            cv2.line(copy_image, (x1, y1), (x2, int(y2)), (0, 0, 255), 10)
            cx = abs(x2)
            print(f"cx:", cx)

    lines_edges = cv2.addWeighted(cv_image, 0.9, line_image, 1, 0)
    middle_line_edge = cv2.addWeighted(cv_image, 0.9, copy_image, 1, 0)

    ###################################################################################################

    # get the dimension of the image
    height, width = cv_image.shape[0], cv_image.shape[1]

    # offset the x position of the vehicle to follow the lane
    # cx -= 170

    pub_yaw_rate(cx, height, width)

    cv_image1 = get_region_of_interest(cv_image)

    cv2.imshow("CV Image", cv_image)
    cv2.imshow("Filtered Image with ROI", filtered_image_with_roi)
    cv2.imshow("Image with ROI", cv_image1)
    cv2.imshow("Hough Lines", lines_edges)
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

    # helps remove some of the yellow from the sunlight
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
    equ = cv2.equalizeHist(gray_balanced_image_with_mask)
    blur = cv2.medianBlur(equ, 15)

    # smooth out the image
    kernel = np.ones((5, 5), np.float32) / 25
    img_dilation = cv2.dilate(blur, kernel, iterations=1)
    smoothed_gray_image = cv2.filter2D(img_dilation, -1, kernel)

    # find and return the edges in in smoothed image
    return cv2.Canny(smoothed_gray_image, 200, 255)

def get_region_of_interest(image):

    width = image.shape[1]
    height = image.shape[0]

    width = width / 8
    height = height / 8

    roi = np.array([[

                       [0, height*8],
                       [0, height*5],
                       [width*2, (height*4)-30],
                       [width*5 , (height*4)-30],
                       [width*8, height*7],
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

def pub_yaw_rate(cx, width, height):

    # compute the coordinates for the center the vehicle's camera view
    camera_center_y = (height / 2)
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
