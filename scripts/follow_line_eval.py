#!/usr/bin/env python3
# evaluation node for line follow with yellow blob detection

import rospy
from std_msgs.msg import Bool, Float32

# global variables
y_detected = False
test_started = False

t0 = 0
n_laps = 0
avg_steer_error = 0
perimeter = 86.32 # length of course in meters

################### callback ###################

# envoked when drive enabled checkbox is checked
def start_test_callback(m):
  global test_started, t0
  if not test_started and m.data:
    t0 = rospy.Time.now().to_sec()
  elif test_started and not m.data:
    test_started = False
  return

def yellow_callback(msg):
  global y_detected, t0, n_laps, perimeter
  if not y_detected and msg.data:
    n_laps = n_laps + 1
    t1 = rospy.Time.now().to_sec()
    dt = t1 - t0
    m_s = perimeter / dt  # meter per second
    km_h = m_s * 3.6      # kilometer per hour
    mi_h = km_h / 1.609   # miles per hour
    print(f"**Lap #{n_laps}, t taken: {dt:.2f} secs, Avg. speed: {m_s:.2f} m/s, {km_h:.2f} km/h, {mi_h: .2f} miles/h")
    y_detected = True
    t0 = t1
  elif y_detected and not msg.data:
    y_detected = False
  return

def steer_err_callback(msg):
  print(f"    Avg. steer err = {msg.data:.2f}")
  return

# def line_touch_detected_cb(msg):
#   return

################### main ###################
if __name__ == "__main__":
  rospy.init_node("follow_line_eval", anonymous=True)

  rospy.Subscriber("start_test", Bool, start_test_callback)
  rospy.Subscriber("yellow_detected", Bool, yellow_callback)
  rospy.Subscriber("steer_err", Float32, steer_err_callback)
  # rospy.Subscriber("line_touch_detected", Bool, line_touch_detected_cb)
  # rospy.Subscriber("end_of_test", Float32, end_of_test_callback)

  try:
    rospy.spin()
  except rospy.ROSInterruptException:
    pass
