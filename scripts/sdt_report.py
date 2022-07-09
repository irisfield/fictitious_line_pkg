#!/usr/bin/env python3

# speed, distance and time (sdt) reporter node

import rospy
import numpy as np
from std_msgs.msg import Float32, Bool
from dbw_polaris_msgs.msg import SteeringReport

# global variables
distance_m = 0.0
previous_time = 0.0
time_elapsed_secs = 0.0

lap_time = 0.0
lap_distance = 0.0
lap_time_initial = 0.0
lap_distance_initial = 0.0

num_laps = 0

start_time = True
start_time_lap = True

list_of_speeds = []

speed_msg = Float32()

################### callback ###################

def steering_report_callback(report):
    global start_time, time_initial, previous_time
    global speed_ms, speed_mph, distance_m, time_elapsed_secs

    speed_ms = report.speed
    speed_mph = speed_ms * 2.237

    # keep track of the total time the vehicle is in motion
    if (speed_ms > 0.0) and start_time:
        # get the initial time in seconds
        time_initial = rospy.Time.now()
        start_time = False
    elif (speed_ms > 0.0) and not start_time:
        # the vehicle is in motion, keep adding the time that has elapsed
        time_final = rospy.Time.now()
        time_elapsed_secs += (time_final - time_initial).to_sec()
        time_initial = time_final
    else:
        # the vehicle stopped, pause the time
        start_time = True

    # compute distance using the distance formula: speed * change_in_time
    if (previous_time == 0.0):
        previous_time = time_elapsed_secs
    elif (int(time_elapsed_secs - previous_time) == 1):
        distance_m += speed_ms * (time_elapsed_secs - previous_time)
        previous_time = time_elapsed_secs

        # display the instantaneous speed, distance and time (SDT) report every second
        rospy.loginfo(f"SDT: {speed_ms:0.1f} m/s -> {speed_mph:0.1f} mph | {distance_m:0.1f} m | {time_elapsed_secs:0.0f} s")

    speed_msg.data = speed_ms
    report_speed_pub.publish(speed_msg)

def yellow_line_callback(yellow_line):
    global lap_time_initial, lap_distance_initial
    global num_laps, lap_time, lap_distance, list_of_speeds

    # display the lap report every lap
    if (speed_ms > 0.0) and (lap_time == 0.0) and (lap_distance == 0.0):
        lap_time_initial = time_elapsed_secs
        lap_distance_initial = distance_m
    elif yellow_line.data:
        num_laps += 1
        average_speed_mph = round((sum(list_of_speeds) / len(list_of_speeds)), 1)
        rospy.loginfo(f"LAP {num_laps} | Average Speed: {average_speed_mph:0.1f} mph | Distance: {lap_distance:0.1f} m | Time: {lap_time:0.0f} s")

        # reset time, distance, and average speed
        lap_time = 0.0
        lap_distance = 0.0
        list_of_speeds = []
    else:
        lap_time_final = time_elapsed_secs
        lap_time += lap_time_final - lap_time_initial
        lap_time_initial = lap_time_final

        lap_distance_final = distance_m
        lap_distance += lap_distance_final - lap_distance_initial
        lap_distance_initial = lap_distance_final

        if (speed_mph > 0.0):
            list_of_speeds.append(speed_mph)

################### main ###################

if __name__ == "__main__":
    rospy.init_node("sdt_report", anonymous=True)

    rospy.Subscriber("/yellow_line_detected", Bool, yellow_line_callback)
    rospy.Subscriber("/vehicle/steering_report", SteeringReport, steering_report_callback)

    report_speed_pub = rospy.Publisher("/sdt_report/speed_ms", Float32, queue_size=1)

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
