#!/usr/bin/env python
import rospy
import numpy as np
from keras.models import load_model
from geometry_msgs.msg import PoseStamped, Pose, TwistStamped
from styx_msgs.msg import TrafficLightArray, TrafficLight, CustomTrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import cv2
import yaml
import math
from math import pow, sqrt
from keras import backend as K

STATE_COUNT_THRESHOLD = 7
SMOOTH = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + SMOOTH) / (K.sum(y_true_f) + K.sum(y_pred_f) + SMOOTH)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.detector_model = None
        self.lights = []
        self.distance_to_tl_threshold = 40
        self.state = TrafficLight.UNKNOWN
        self.state_count = 0
        self.has_image = False
        self.dist_pre = 0
        self.last_state = TrafficLight.UNKNOWN
        self.car_curr_vel = None
        self.car_dist = 0

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.light_classifier = TLClassifier()
        model = load_model(self.config['tl']['tl_classification_model'])
        resize_width = self.config['tl']['classifier_resize_width']
        resize_height = self.config['tl']['classifier_resize_height']
        self.light_classifier.setup_classifier(model, resize_width, resize_height)
        self.detector_model = load_model(self.config['tl']['tl_detection_model'], custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef })
        self.detector_model._make_predict_function()
        self.resize_width = self.config['tl']['detector_resize_width']
        self.resize_height = self.config['tl']['detector_resize_height']
        self.resize_height_ratio = self.config['camera_info']['image_height']/float(self.resize_height)
        self.resize_width_ratio = self.config['camera_info']['image_width']/float(self.resize_width)
        self.middle_col = self.resize_width/2
        self.is_carla = self.config['tl']['is_carla']
        self.projection_threshold = self.config['tl']['projection_threshold']
        self.projection_min = self.config['tl']['projection_min']
        self.color_mode = self.config['tl']['color_mode']

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_call_back)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_call_back)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_call_back)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_call_back)
        rospy.Subscriber('/image_color', Image, self.image_call_back)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', CustomTrafficLight, queue_size=1)
        self.bridge = CvBridge()
        detector_rate = rospy.Rate(self.config['tl']['detector_rate'])
        while not rospy.is_shutdown():
            self.find_traffic_lights()
            detector_rate.sleep()

    def pose_call_back(self, msg):
        self.pose = msg

    def waypoints_call_back(self, waypoints):
        self.waypoints = waypoints.waypoints

    def traffic_call_back(self, msg):
        self.lights = msg.lights

    def image_call_back(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg

    def current_velocity_call_back(self, msg):
        curr_lin = [msg.twist.linear.x, msg.twist.linear.y]
        self.car_curr_vel = math.sqrt(curr_lin[0]**2 + curr_lin[1]**2)

    def find_traffic_lights(self):
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if state == 0 or state == 2:
            tl_result = CustomTrafficLight()
            tl_result.state = state
            tl_result.dist = int(self.car_dist * 1000)
            self.upcoming_red_light_pub.publish(tl_result)
            self.last_state = state
            self.state_count = 0
        else:
            if self.last_state == 0 and self.state_count <= STATE_COUNT_THRESHOLD and self.car_curr_vel < 0.5:
                tl_result = CustomTrafficLight()
                tl_result.state = self.last_state
                tl_result.dist = int(self.car_dist * 1000)
                self.upcoming_red_light_pub.publish(tl_result)
                self.state_count += 1
            else:
                tl_result = CustomTrafficLight()
                tl_result.state = state
                tl_result.dist = int(self.car_dist * 1000)
                self.upcoming_red_light_pub.publish(tl_result)
                self.state_count += 1

    def distance_to_point(self, pose, wp_pose):
        x_squared = pow((pose.position.x - wp_pose.position.x), 2)
        y_squared = pow((pose.position.y - wp_pose.position.y), 2)
        dist = sqrt(x_squared + y_squared)
        return dist

    def get_closest_waypoint(self, pose, waypoints):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        min_dist = float("inf")
        closest_wp_idx = -1

        if not waypoints:
            rospy.logwarn("No waypoints.")
        else:

            for idx, wp in enumerate(waypoints):
                dist = self.distance_to_point(pose, wp.pose.pose)
                if(dist < min_dist):
                    min_dist = dist
                    closest_wp_idx = idx
        return closest_wp_idx

    def _extract_image(self, pred_image_mask, image):
        if (np.max(pred_image_mask) < self.projection_min):
            return None

        row_projection = np.sum(pred_image_mask, axis = 1)
        row_index =  np.argmax(row_projection)

        if (np.max(row_projection) < self.projection_threshold):
            return None

        zero_row_indexes = np.argwhere(row_projection <= self.projection_threshold)
        top_part = zero_row_indexes[zero_row_indexes < row_index]
        top = np.max(top_part) if top_part.size > 0 else 0
        bottom_part = zero_row_indexes[zero_row_indexes > row_index]
        bottom = np.min(bottom_part) if bottom_part.size > 0 else self.resize_height

        roi = pred_image_mask[top:bottom,:]
        column_projection = np.sum(roi, axis = 0)

        if (np.max(column_projection) < self.projection_min):
            return None

        non_zero_column_index = np.argwhere(column_projection > self.projection_min)

        index_of_column_index = np.argmin(np.abs(non_zero_column_index - self.middle_col))
        column_index = non_zero_column_index[index_of_column_index][0]

        zero_colum_indexes = np.argwhere(column_projection == 0)
        left_side = zero_colum_indexes[zero_colum_indexes < column_index]
        left = np.max(left_side) if left_side.size > 0 else 0
        right_side = zero_colum_indexes[zero_colum_indexes > column_index]
        right = np.min(right_side) if right_side.size > 0 else self.resize_width
        return image[int(top*self.resize_height_ratio):int(bottom*self.resize_height_ratio), int(left*self.resize_width_ratio):int(right*self.resize_width_ratio)]

    def detect_traffic_light(self, cv_image):
        resize_image = cv2.cvtColor(cv2.resize(cv_image, (self.resize_width, self.resize_height)), cv2.COLOR_RGB2GRAY)
        resize_image = resize_image[..., np.newaxis]
        if self.is_carla:
            mean = int(np.mean(resize_image)) # mean for data centering
            std = int(np.std(resize_image)) # std for data normalization

            resize_image -= mean
            resize_image /= std

        image_mask = self.detector_model.predict(resize_image[None, :, :, :], batch_size=1)[0]
        image_mask = (image_mask[:,:,0]*255).astype(np.uint8)

        # cv2.imwrite("Image_mask_"+ str(self.ctr_2) + "_.jpg", cv2.addWeighted(cv2.bitwise_not(image_mask), 0.3,
        #                                                                       resize_image, 0.7, 1))
        #
        #
        # self.ctr_2 = self.ctr_2 + 1

        return self._extract_image(image_mask, cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        if self.pose is not None and self.has_image:
            tl_id = self.get_closest_waypoint(self.pose.pose, self.lights)
            if (tl_id >= 0):
                stop_line = self.config['stop_line_positions'][tl_id]
                stop_line_pose = Pose()
                stop_line_pose.position.x = stop_line[0]
                stop_line_pose.position.y = stop_line[1]
                wp_id = self.get_closest_waypoint(stop_line_pose, self.waypoints)
                if (wp_id == -1):
                    return -1, TrafficLight.UNKNOWN
                self.car_dist = self.distance_to_point(self.pose.pose, stop_line_pose)
                state = TrafficLight.UNKNOWN
                if (self.car_dist < self.distance_to_tl_threshold and self.car_dist <= self.dist_pre):
                    cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, self.color_mode)
                    tl_image = self.detect_traffic_light(cv_image)
                    if tl_image is not None:
                        state = self.light_classifier.get_classification(tl_image)
                        if state == 0:
                            rospy.loginfo("TrafficLight is RED")
                        elif state == 1:
                            rospy.loginfo("TrafficLight is YELLOW")
                        elif state == 2:
                            rospy.loginfo("TrafficLight is GREEN")
                        elif state == 4:
                            rospy.loginfo("TrafficLight is NO")
                        else:
                            rospy.loginfo("TrafficLight is UN")
                        # cv2.imwrite("Image_"+ str(self.ctr) + "_" + state_txt + "_" + str(self.car_dist) +"_.jpg", cv2.cvtColor(
                        #     tl_image,
                        # cv2.COLOR_BGR2RGB))
                        # self.ctr = self.ctr + 1
                else:
                    state = TrafficLight.UNKNOWN
                self.dist_pre = self.car_dist
                return wp_id, state
            else:
                return -1, TrafficLight.UNKNOWN
        else:
            return -1, TrafficLight.UNKNOWN
        self.waypoints = None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
