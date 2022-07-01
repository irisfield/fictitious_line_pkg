#!/usr/bin/env python3
# Follow a outer line with Yellow Detection
#import numpy as np
import time
import rospy
import cv2
import numpy as np
import math
from std_msgs.msg import Bool
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from dynamic_reconfigure.server import Server
from fictitious_line_pkg.cfg import FollowLineConfig   # packageName.cfg
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty

# global variables
vel_msg = Twist()
start_stop_msg = Bool()
steer_error_msg = Float32()

drive = False

n_laps = 0
total_frames = 0
sum_steer_error = 0
count = 0

font = cv2.FONT_HERSHEY_SIMPLEX
end_test = False
y_detected = False

# for the vehicle, the message must start at 0.0
#vel_msg.linear.x = 0.0
#vel_msg.angular.z = 0.0

##############
# ALGORITHMS #
#############
def dynamic_reconfigure_callback(config, level):
    global RC
    RC = config
    if RC.enable_drive:
        start_stop_msg.data = True
        start_stop_pub.publish(start_stop_msg)
    return config

def yellow_callback(msg):
    global steer_error_msg, n_laps, total_frames, sum_steer_error, end_test, y_detected
    if not y_detected and msg.data:
        n_laps += 1
        steer_error_msg.data = (sum_steer_error / total_frames) if (total_frames > 0) else 0.0
        steer_error_pub.publish(steer_error_msg)
        sum_steer_error = 0
        total_frames = 0
        y_detected = True
        if n_laps == 2:
            end_test = True
    elif y_detected and not msg.data:
        y_detected = False
    return

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def image_callback(ros_image):
    global end_test, cols, rows

    if end_test:
      stop_vehicle()
      return

    try:
        # convert ros_image into an opencv-compatible image
        cv_image = CvBridge().imgmsg_to_cv2(ros_image, "bgr8")
    except CvBridgeError:
        print(CvBridgeError)


    # resize the image
    cv_image = cv2.resize(cv_image, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
    (width, height, _) = cv_image.shape
    #region_of_interest_vertices = [(0, height),(width / 2, height / 2),(width, height),]
    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray_image, 100, 200)
    #cropped_image = region_of_interest(canny, np.array([region_of_interest_vertices],np.int32),)
    lines = cv2.HoughLinesP(canny,rho=6,theta=np.pi / 180,threshold=15,lines=np.array([]),minLineLength=20,maxLineGap=30)
    print(lines)
    
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1) # <-- Calculating the slope.
            print("slope", slope)
        if math.fabs(slope) < 0.5:
            continue
        if slope <= 0: # <-- If the slope is negative, left group.
            left_line_x.extend([x1, x2])
            left_line_y.extend([y1, y2])
        else: # <-- Otherwise, right group.
            right_line_x.extend([x1, x2])
            right_line_y.extend([y1, y2])
    
    min_y = cv_image.shape[0] * (3 / 5) # <-- Just below the horizon
    max_y = cv_image.shape[0] # <-- The bottom of the image
    
    poly_left = np.poly1d(np.polyfit(left_line_y,left_line_x,deg=1))
    print(poly_left)
    left_x_start = int(poly_left(max_y))
    left_x_end = int(poly_left(min_y))
    
    poly_right = np.poly1d(np.polyfit(right_line_y,right_line_x,deg=1))
    right_x_start = int(poly_right(max_y))
    right_x_end = int(poly_right(min_y))

    poly_middle= np.poly1d(np.polyfit(poly_left, poly_right,deg=1))
    middle_x_start =  int(poly_middle(max_y))
    middle_x_end =  int(poly_middle(max_y))

    side_lines= [[[left_x_start, max_y, left_x_end, min_y],[right_x_start, max_y, right_x_end, min_y],]]
    middle_line= [[[middle_x_start, max_y, middle_x_end, min_y],]]

    line_image = np.copy(cv_image)*0
    copy_image = np.copy(cv_image)*0

    for line in side_lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,int(y2)),(255,0,0),10)
            
    for line in middle_line:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,int(y2)),(0,0,255),10)
            cv2.line(copy_image,(x1,y1),(x2,int(y2)),(0,0,255),10)

    lines_edges = cv2.addWeighted(cv_image, 0.8, line_image, 1, 0)
    middle_line_edge = cv2.addWeighted(cv_image, 0.8, copy_image, 1, 0)

    # do not crop the image for the simulator
    # crop the lower 1/3 of the image for faster processing
    # half_width = int(rows / 2)
    # cv_image = cv_image[0:half_width, 0:cols]

    # convert the image to grayscale
    hsv = cv2.cvtColor(middle_line_edge,cv2.COLOR_BGR2HSV)
    thresh1 = cv2.inRange(hsv,np.array((0, 80, 80)), np.array((20, 255, 255))) 
    thresh2 = cv2.inRange(hsv,np.array((150, 80, 80)), np.array((180, 255, 255))) 
    thresh =  cv2.bitwise_or(thresh1, thresh2)

    # find the contours in the binary image
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

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
        print(f"empty contours: {contours}")

    try:
        # draw the obtained contour lines (or the set of coordinates forming a line) on the original image
        # https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
        cv2.drawContours(middle_line_edge, max_contour, -1, (0, 255, 0), 10)
    except UnboundLocalError:
        print("max contour not found")

    # draw a circle at centroid (https://www.geeksforgeeks.org/python-opencv-cv2-circle-method)
    cv2.circle(middle_line_edge, (cx, cy), 8, (180, 0, 0), -1)  # -1 fill the circle


    # offset the x position of the robot to follow the lane
    drive_to_follow_line_2(middle_line_edge, cx, cy, width, height)

    #cv2.imshow("My Image Window", bw_image)
    cv2.imshow("CV RGB image (upper half)", cv_image)
    #cv2.imshow("Binary Image", bw_image)
    cv2.imshow("Canny", canny)
    cv2.imshow("Hough", lines_edges)
    cv2.imshow("middle", middle_line_edge)
    cv2.waitKey(3)
    
# ALGORITHM 2
def drive_to_follow_line_2(cv_image, cx, cy, width, height):
    global sum_steer_error, total_frames, count, y_detected

    if RC.enable_drive:
        if count == 0:
            left_turn(0.6, 0.127)
            rospy.sleep(2)
            print(f"count is: {count}")
        vel_msg.linear.x = RC.speed

        # get the center position of the car"s camera view
        camera_center_y = (width / 2) # corresponds to x
        camera_center_x = (height / 2) # corresponds to y


        # compute the difference between vertical center of the centroid and car"s camera view
        steer_error = abs(cx - camera_center_x)

        # In simulation:
        #       <3 - deviates a little inward when turning
        #       3 - follows the line exactly
        #       3> - deviates a little outward when turning
        correction = 3.0 * camera_center_y

        # compute the angular velocity based on the speed of the robot
        angular_vel = float(steer_error / correction)

        if cx > camera_center_x:
            # angular.z is negative
            vel_msg.angular.z = -abs(angular_vel)
            velocity_pub.publish(vel_msg)
        elif cx < camera_center_x:
            # angular.z is positive
            vel_msg.angular.z = abs(angular_vel)
            velocity_pub.publish(vel_msg)
        else:
            vel_msg.angular.z = 0
            velocity_pub.publish(vel_msg)

        sum_steer_error += steer_error
        total_frames += 1
        count = 1
        # print(f"count here is: {count}")

        if y_detected:
            rospy.loginfo(f"Stopping at yellow")
            vel_msg.linear.x = 0
            vel_msg.angular.z = 0
            velocity_pub.publish(vel_msg)
            rospy.sleep(3)
            if count ==1:
                left_turn(0.6, 0.07)
                for i in range(40):
                    stop_vehicle()
    else:
        stop_vehicle()

    return

####################
# HELPER FUNCTIONS #
####################
def left_turn(veh_speed, veh_yaw):
    # rospy.sleep(3)
    time_dur = 3.8

    # Loop and publish commands to vehicle
    time_start = rospy.Time.now()
    time_elapsed = 0.0
    time_stop = 0.0

    while(time_elapsed <= time_dur):

        # compute dt in seconds
        time_elapsed = (rospy.Time.now() - time_start).to_sec()

        if(time_elapsed <  time_stop ):
            rospy.loginfo(f"Stopping for {time_stop} seconds")
            vel_msg.linear.x = 0
            vel_msg.angular.z = 0
        elif(time_elapsed < time_dur + time_stop ):
            vel_msg.linear.x = veh_speed
            vel_msg.angular.z = veh_yaw
        else:
            vel_msg.linear.x = 0
            vel_msg.angular.z = 0

        velocity_pub.publish(vel_msg)
    return

def stop_vehicle():
      vel_msg.linear.x = 0.0
      vel_msg.angular.z = 0.0
      velocity_pub.publish(vel_msg)
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

########
# MAIN #
########
if __name__ == "__main__":
    rospy.init_node("follow_line_y", anonymous=True)

    imgtopic = rospy.get_param("~imgtopic_name")  # as defined in the launch file
    rospy.Subscriber(imgtopic, Image, image_callback)
    rospy.Subscriber("yellow_detected", Bool, yellow_callback)

    velocity_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    start_stop_pub = rospy.Publisher("start_test", Bool, queue_size=1)
    steer_error_pub = rospy.Publisher("steer_err", Float32, queue_size=1)

    #cmd_vel_pub = rospy.Publisher("/vehicle/cmd_vel", Twist, queue_size=1)
    #enable_drive_pub = rospy.Publisher("/vehicle/enable", Empty, queue_size=1)

    srv = Server(FollowLineConfig, dynamic_reconfigure_callback)

    #drive_controller()

    try:
      rospy.spin()
    except rospy.ROSInterruptException:
      pass
