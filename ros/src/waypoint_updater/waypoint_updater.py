#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

import math
import tf

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
ACC_FACTOR = 0.5

class WaypointUpdater(object):
    def __init__(self):

        # Initialize the node with the Master Process
        rospy.init_node('waypoint_updater')

        # Subscribers
        self.base_waypoints_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        # rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb)

        # Publishers
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1, latch = True)

        # TODO: Add other member variables you need below
        self.car_position = None
        self.car_yaw = None
        self.car_curr_vel = None
        self.cruise_speed = None
        self.closestWaypoint = None
        self.waypoints = []
        self.final_waypoints = []
        self.tl_state = None

        # rospy.spin()
        
        self.loop()

    # Main loop
    def loop(self):
        rate = rospy.Rate(5)
        # setting

        self.cruise_speed = 40.0
        self.decel_limit = abs(rospy.get_param('~/twist_controller/decel_limit', -5))
        self.accel_limit = rospy.get_param('~/twist_controller/accel_limit', 1)

        while not rospy.is_shutdown():
            if (self.car_position != None and self.waypoints != None and self.car_curr_vel != None):
                self.closestWaypoint = self.NextWaypoint(self.car_position, self.car_yaw, self.waypoints)
                self.generate_final_waypoints(self.closestWaypoint, self.waypoints)

                # output
                self.publish()
            else:
                rospy.logwarn("Data not received")
            rate.sleep()

    def pose_cb(self, msg):
        car_pose = msg.pose
        self.car_position = car_pose.position
        car_orientation = car_pose.orientation
        quaternion = (car_orientation.x, car_orientation.y, car_orientation.z, car_orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        self.car_yaw = euler[2]

    def waypoints_cb(self, msg):
        for waypoint in msg.waypoints:
            self.waypoints.append(waypoint)
        self.base_waypoints_sub.unregister()

    def traffic_cb(self, msg):
        if msg.data == 0:
            self.tl_state = "RED"
            # self.tl_idx = msg.waypoint
        elif msg.data == 1:
            self.tl_state = "YELLOW"
            # self.tl_idx = msg.waypoint
        elif msg.data == 2:
            self.tl_state = "GREEN"
            # self.tl_idx = msg.waypoint
        elif msg.data == 4:
            self.tl_state = "NO"
            # self.tl_idx = msg.waypoint

    def current_velocity_cb(self, msg):
        curr_lin = [msg.twist.linear.x, msg.twist.linear.y]
        self.car_curr_vel = math.sqrt(curr_lin[0]**2 + curr_lin[1]**2)

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        self.obstacle_waypoint = msg.data

    def stop_waypoints(self, closestWaypoint, waypoints):
        init_vel = self.car_curr_vel
        end = closestWaypoint + LOOKAHEAD_WPS
        if end > len(waypoints) - 1:
            end = len(waypoints) - 1
        for idx in range(closestWaypoint, end):
            velocity = 0.0
            self.set_waypoint_velocity(waypoints, idx, velocity)
            self.final_waypoints.append(waypoints[idx])

    def go_waypoints(self, closestWaypoint, waypoints):
        init_vel = self.car_curr_vel
        end = closestWaypoint + LOOKAHEAD_WPS
        if end > len(waypoints) - 1:
           end = len(waypoints) - 1
        a = ACC_FACTOR * self.accel_limit
        for idx in range(closestWaypoint, end):
            dist = self.distance(waypoints, closestWaypoint, idx+1)
            velocity = math.sqrt(init_vel**2 + 2 * a * dist)
            if velocity > self.cruise_speed:
               velocity = self.cruise_speed
            self.set_waypoint_velocity(waypoints, idx, velocity)
            self.final_waypoints.append(waypoints[idx])

    def generate_final_waypoints(self, closestWaypoint, waypoints):
        self.final_waypoints = []
        if self.tl_state == "RED":
            self.stop_waypoints(closestWaypoint, waypoints)
        else:
            self.go_waypoints(closestWaypoint, waypoints)
        # self.go_waypoints(closestWaypoint, waypoints)

    def publish(self):
        final_waypoints_msg = Lane()
        final_waypoints_msg.header.frame_id = '/World'
        final_waypoints_msg.header.stamp = rospy.Time(0)
        final_waypoints_msg.waypoints = list(self.final_waypoints)
        self.final_waypoints_pub.publish(final_waypoints_msg)

    def closest_waypoint(self, position, waypoints):
        closestLen = float("inf")
        closestWaypoint = 0
        dist = 0.0
        for idx in range(0, len(waypoints)):
            x = position.x
            y = position.y
            map_x = waypoints[idx].pose.pose.position.x
            map_y = waypoints[idx].pose.pose.position.y
            dist = self.distance_any(x, y, map_x, map_y)
            if (dist < closestLen):
                closestLen = dist
                closestWaypoint = idx
        return closestWaypoint

    def NextWaypoint(self, position, yaw, waypoints):
        closestWaypoint = self.closest_waypoint(position, waypoints)
        map_x = waypoints[closestWaypoint].pose.pose.position.x
        map_y = waypoints[closestWaypoint].pose.pose.position.y
        heading = math.atan2((map_y - position.y), (map_x - position.x))
        angle = abs(yaw - heading)
        if (angle > math.pi/4):
            closestWaypoint += 1
            if (closestWaypoint > len(waypoints)-1):
                closestWaypoint -= 1
        return closestWaypoint

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def distance_any(self, x1, y1, x2, y2):
        return math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
