#!/usr/bin/env python3

# speed, distance and time (sdt) reporter node

import rospy
import numpy as np
from std_msgs.msg import Float32, Int32
from dbw_polaris_msgs.msg import SteeringReport

# global variables
speed_ms = 0.0
distance_m = 0.0
average_speed_mph = 0.0


time_of_lap = 0
time_initial = 0
time_elapsed_secs = 0

n_laps = 0
previous_time = 0
start_time = True

time_msg = Int32()
speed_msg = Float32()
distance_msg = Float32()

################### callback ###################

def steering_report_callback(report):
    global speed_ms, time_initial, time_elapsed_secs, start_time, previous_time, distance_m

    speed_ms = report.speed
    speed_mph = speed_ms * 2.237

    # keep track of the total time the vehicle is in motion
    if (speed_ms > 0.0) and start_time:
        # get the initial time in seconds
        time_initial = report.header.stamp.secs
        start_time = False
    elif (speed_ms > 0.0) and not start_time:
        # the vehicle is in motion, keep adding the time that has elapsed
        time_final = report.header.stamp.secs
        time_elapsed_secs += time_final - time_initial
        time_initial = time_final
    else:
        # the vehicle stopped, pause the time
        start_time = True

    # compute distance_m using the distance_m formula: speed * change_in_time
    if (previous_time == 0):
        previous_time = time_elapsed_secs
    elif ((time_elapsed_secs - previous_time) == 1):
        distance_m += speed_ms * (time_elapsed_secs - previous_time)
        # display the instantaneous speed and the time every second
        rospy.loginfo(f"Instantaneous Speed: {speed_ms:0.1f} m/s -> {speed_mph:0.1f} mph | Distance: {distance_m:0.1f} m | Time: {time_elapsed_secs} s")
        previous_time = time_elapsed_secs

    time_msg.data = time_elapsed_secs
    speed_msg.data = speed_ms
    distance_msg.data = distance_m

    report_time_pub.publish(time_msg)
    report_speed_pub.publish(speed_msg)
    report_distance_m_pub.publish(distance_msg)

################### main ###################

if __name__ == "__main__":
    rospy.init_node("sdt_report", anonymous=True)

    rospy.Subscriber("/vehicle/steering_report", SteeringReport, steering_report_callback)

    report_time_pub = rospy.Publisher("/sdt_report/time_secs", Int32, queue_size=1)
    report_speed_pub = rospy.Publisher("/sdt_report/speed_ms", Float32, queue_size=1)
    report_distance_m_pub = rospy.Publisher("/sdt_report/distance_m", Float32, queue_size=1)


    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
