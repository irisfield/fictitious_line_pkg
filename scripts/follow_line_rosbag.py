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

def apply_filters(cv_image):

    #apply white balance filter to even out the image
    cv_image= find_white_balance(cv_image)

    #convert image to hls
    imgHLS = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HLS)
    lower = np.uint8([  0, 200,   0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(imgHLS, lower, upper)
    # yellow color mask
    lower = np.uint8([ 10,   0, 100])
    upper = np.uint8([ 40, 255, 255])
    yellow_mask = cv2.inRange(imgHLS, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    res =  cv2.bitwise_and(cv_image, cv_image, mask = mask)

    #convert image to grayscale
    gray_image = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    #smooth the image out
    kernel = np.ones((5, 5), np.float32) / 25    
    smoothed = cv2.filter2D(gray_image, -1, kernel)

    #find edges in picture
    filtered = cv2.Canny(smoothed, 200, 255)

    return filtered

def find_white_balance(cv_image):
    result = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
    average_a = np.average(result[:,:,1])
    average_b = np.average(result[:,:,2])
    result[:,:,1] = result[:,:,1] - ((average_a - 128) * (result[:,:,0] / 255.0) * 1.1)
    result[:,:,2] = result[:,:,2] - ((average_b - 128) * (result[:,:,0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

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
    cv_image = cv_image[int(width/4):, :]
    #region_of_interest_vertices = [(0, height),(width / 2, height / 2),(width, height),]
    #gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    #canny = cv2.Canny(gray_image, 100, 200)
    #cropped_image = region_of_interest(canny, np.array([region_of_interest_vertices],np.int32),)
    white_balanced = find_white_balance(cv_image)
    filtered = apply_filters(white_balanced)
    lines = cv2.HoughLinesP(filtered,rho=6,theta=np.pi / 180,threshold=15,lines=np.array([]),minLineLength=20,maxLineGap=30)
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

    lines1= [[[left_x_start, max_y, left_x_end, min_y],[right_x_start, max_y, right_x_end, min_y],[middle_x_start, max_y, middle_x_end, min_y],]]

    line_image = np.copy(cv_image)*0

    for line in lines1:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,int(y2)),(255,0,0),10)

    lines_edges = cv2.addWeighted(cv_image, 0.8, line_image, 1, 0)
    
    #cv2.imshow("My Image Window", bw_image)
    cv2.imshow("CV RGB image (upper half)", cv_image)
    #cv2.imshow("Binary Image", bw_image)
    cv2.imshow("Canny", filtered)
    cv2.imshow("Hough", lines_edges)
    cv2.waitKey(3)

########
# MAIN #
########
if __name__ == "__main__":
    rospy.init_node("follow_line_y", anonymous=True)

    # imgtopic = rospy.get_param("~imgtopic_name")  # as defined in the launch file
    rospy.Subscriber("/camera/image_raw", Image, image_callback)
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
