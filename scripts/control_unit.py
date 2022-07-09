#!/usr/bin/env python3

import cv2
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32, Empty
from dynamic_reconfigure.server import Server
from fictitious_line_pkg.cfg import ControlUnitConfig

# global variables
vel_msg = Twist()

# as required by the drive-by-wire system
vel_msg.linear.x = 0.0
vel_msg.angular.z = 0.0

################### callback ###################

def dynamic_reconfigure_callback(config, level):
    global RC
    RC = config
    return config

def yaw_rate_callback(angular_z):
    global yaw_rate
    yaw_rate = angular_z.data

def yellow_line_callback(yellow_line):
    global vel_msg

    if RC.enable_drive:
        if yellow_line.data:
            # drive to the yellow line
            drive_duration(1.0, 0.0, 3.0)

            # stop at the yelllow line for 3 seconds
            drive_duration(0.0, 0.0, 3.0)

            # drive the curve until it finds the other lane
            drive_duration(1.0, 0.0, 2.0)
            drive_duration(1.0, 0.12, 16.0)
        else:
            # engage the line following algorithm
            vel_msg.linear.x = RC.speed
            vel_msg.angular.z = yaw_rate

            # this is being publish in publish_vel_msg()
            # cmd_vel_pub.publish(vel_msg)
    else:
        stop_vehicle()

    return

################### helper functions ###################

def drive_duration(speed, yaw_rate, duration):
    time_initial = rospy.Time.now()
    time_elapsed = 0.0

    while(time_elapsed <= duration):
        vel_msg.linear.x = speed
        vel_msg.angular.z = yaw_rate
        cmd_vel_pub.publish(vel_msg)

        # compute elapsed time in seconds
        time_elapsed = (rospy.Time.now() - time_initial).to_sec()

    # stop the vehicle after driving the duration
    stop_vehicle()

    return

def publish_vel_msg():
    global vel_msg

    rate = rospy.Rate(25)
    enable_drive_msg = Empty()

    time_start = rospy.Time.now()

    # wait two seconds for the drive-by-wire system to synchronize
    dbw_wait_time = 2.0

    while (not rospy.is_shutdown()):
        time_elapsed = (rospy.Time.now() - time_start).to_sec()

        if (time_elapsed <= dbw_wait_time):
            vel_msg.linear.x = 0.0
            vel_msg.angular.z = 0.0

        # publish empty message to prevent the drive-by-wire system from timing out
        enable_drive_pub.publish(enable_drive_msg)
        # publish the vel_msg here
        cmd_vel_pub.publish(vel_msg)
        rate.sleep()

    return

def stop_vehicle():
    vel_msg.linear.x = 0.0
    vel_msg.angular.z = 0.0
    cmd_vel_pub.publish(vel_msg)
    return

################### main ###################

if __name__ == "__main__":
    rospy.init_node("control_unit", anonymous=True)

    rospy.Subscriber("/yaw_rate", Float32, yaw_rate_callback)
    rospy.Subscriber("/yellow_line_detected", Bool, yellow_line_callback)

    cmd_vel_pub = rospy.Publisher("/vehicle/cmd_vel", Twist, queue_size=1)
    enable_drive_pub = rospy.Publisher("/vehicle/enable", Empty, queue_size=1)

    dynamic_reconfigure_server = Server(ControlUnitConfig, dynamic_reconfigure_callback)

    # publish velocity message
    publish_vel_msg()

    try:
      rospy.spin()
    except rospy.ROSInterruptException:
      pass
