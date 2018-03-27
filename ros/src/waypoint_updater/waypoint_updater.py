#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint, CustomTrafficLight
from std_msgs.msg import Int32

import math

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
        self.base_waypoints_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_call_back)
        # rospy.Subscriber('/base_waypoints', Lane, self.waypoints_call_back)
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_call_back)
        rospy.Subscriber('/traffic_waypoint', CustomTrafficLight, self.traffic_call_back)
        rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_call_back)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_call_back)

        # Publishers
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1, latch = True)

        self.car_position = None
        self.car_curr_vel = None
        self.cruise_speed = None
        self.closestWaypoint = None
        self.waypoints = []
        self.final_waypoints = []
        self.tl_state = None
        self.velocity = None
        self.dist_to_stop = None

        self.loop()

    # Main loop
    def loop(self):
        rate = rospy.Rate(5)

        self.cruise_speed = rospy.get_param('~/waypoint_loader/velocity', 40.0)*0.278
        self.decel_limit = abs(rospy.get_param('~/twist_controller/decel_limit', -5))
        self.accel_limit = rospy.get_param('~/twist_controller/accel_limit', 1)

        while not rospy.is_shutdown():
            if (self.car_position != None and self.waypoints != None and self.car_curr_vel != None):
                self.closestWaypoint = self.closest_waypoint(self.car_position, self.waypoints)
                self.generate_final_waypoints(self.closestWaypoint, self.waypoints)
                self.publish()
            rate.sleep()

    def pose_call_back(self, msg):
        car_pose = msg.pose
        self.car_position = car_pose.position

    def waypoints_call_back(self, msg):
        for waypoint in msg.waypoints:
            self.waypoints.append(waypoint)
        self.base_waypoints_sub.unregister()

    def traffic_call_back(self, msg):
        if msg.state == 0:
            self.tl_state = "RED"
        elif msg.state == 1:
            self.tl_state = "YELLOW"
        elif msg.state == 2:
            self.tl_state = "GREEN"
        elif msg.state == 4:
            self.tl_state = "NO"
        self.dist_to_stop = float(msg.dist) / 1000

    def current_velocity_call_back(self, msg):
        curr_lin = [msg.twist.linear.x, msg.twist.linear.y]
        self.car_curr_vel = math.sqrt(curr_lin[0]**2 + curr_lin[1]**2)

    def obstacle_call_back(self, msg):
        self.obstacle_waypoint = msg.data

    def deceleration_waypoints(self, closestWaypoint, waypoints, velocity):
        end = closestWaypoint + LOOKAHEAD_WPS
        if end > len(waypoints) - 1:
            end = len(waypoints) - 1
        for idx in range(closestWaypoint, end):
            self.set_waypoint_velocity(waypoints, idx, velocity)
            self.final_waypoints.append(waypoints[idx])

    def acceleration_waypoints(self, closestWaypoint, waypoints):
        init_vel = self.car_curr_vel
        end = closestWaypoint + LOOKAHEAD_WPS
        if end > len(waypoints) - 1:
           end = len(waypoints) - 1
        a = ACC_FACTOR * self.accel_limit
        for idx in range(closestWaypoint, end):
            dist = self.distance_1(waypoints, closestWaypoint, idx+1)
            velocity = math.sqrt(init_vel**2 + 2 * a * dist)
            if velocity > self.cruise_speed:
               velocity = self.cruise_speed
            self.set_waypoint_velocity(waypoints, idx, velocity)
            self.final_waypoints.append(waypoints[idx])

    def generate_final_waypoints(self, closestWaypoint, waypoints):
        self.final_waypoints = []
        if self.tl_state == "RED":
            a = ACC_FACTOR * self.decel_limit
            self.velocity = 0.3*self.dist_to_stop-1
            if self.velocity > self.cruise_speed:
                self.velocity = self.cruise_speed
            elif self.velocity < 0:
                self.velocity = 0
            self.deceleration_waypoints(closestWaypoint, waypoints, self.velocity)
        else:
            self.acceleration_waypoints(closestWaypoint, waypoints)

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
            dist = self.distance_2(x, y, map_x, map_y)
            if (dist < closestLen):
                closestLen = dist
                closestWaypoint = idx
        return closestWaypoint

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance_1(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def distance_2(self, x1, y1, x2, y2):
        return math.sqrt((x1-x2)**2 + (y1-y2)**2)


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
