#!/usr/bin/env python2
import rospy
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Quaternion, Pose, Point, Vector3
from std_msgs.msg import Header, ColorRGBA
import math, numpy as np
import tf2_ros, tf
from scipy import stats




class LineExtractionPaper():
    """
    exploring what is being said and done in
        "Feature-Based Laser Scan Matching For Accurate and High Speed Mobile Robot Localization"
    """


    def __init__(self):

        # defining constants
        self.ANGLE_INCREMENT = 0.017501922324299812 # degrees

        self.SIGMA_R = 0.03 # meter
        self.LAMBDA = 10 # degrees
        self.D_MAX_CONSTANT = math.sin(math.radians(self.ANGLE_INCREMENT))/math.sin(math.radians(self.LAMBDA-self.ANGLE_INCREMENT))


        self.Z_OFFSET = 0.1

        # marker constants
        self.base_marker_type = Marker.LINE_STRIP
        self.base_marker_lifetime = rospy.Duration(20)
        self.base_marker_header_frame_id = 'base_scan'
        self.base_marker_action = 0
        self.base_marker_scale_x = 0.01
        self.base_marker_pose_orientation = Quaternion(0.0, 0.0, 0.0, 1.0)

        self.marker_id = 0
        self.point_id = 0

        # feature collection
        self.corners = [] # features list

        # initializing rospy publishers and subscribers
        rospy.init_node('clusters', anonymous=True)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.line_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
        rospy.Subscriber('scan', LaserScan, self.callback)



        rospy.spin()

    def polar_to_cartesian(self, distance, angle):
        """ This function converts polar coordinates to cartesian coordinates

        Args:
            distance (float): The distance that the Robot is away from the point that is being viewed
            angle    (float): The angle that the Robot is away from the point that is being viewed

        Returns:
            pointX (float): the X coordinate
            pointY (float): the Y coordinate

        """
        pointX = distance * math.cos(angle)
        pointY = distance * math.sin(angle)
        return pointX, pointY

    def distance(self, a, b):
        """ This is a distance function that returns the distance between the 2 given points squared

        Args:
            a (Point): a point with x and y coordinates
            b (Point): a point with x and y coordinates

        Returns:
            dist (float): the distance between the 2 given points squared
        """
        dist = (b.x - a.x) ** 2 + (b.y - a.y) ** 2
        return dist

    def show_point_in_rviz(self, point, point_color=ColorRGBA(0.0, 1.0, 0.0, 0.8)):
        """ This function takes a point to then place a Marker at that position
        With an optional argument to set the color

        Args:
            point        (Point): Point to be displayed
            (point_color (ColorRGBA)): Optional color argument to change the color of the point
        """
        marker = Marker(
                    header=Header(
                    #frame_id=self.base_marker_header_frame_id),
                    frame_id='odom'), # odom is the fixed frame, it sounds like with real data that will not be available?
                    id=self.point_id,
                    type=Marker.SPHERE,
                    pose=Pose(point, Quaternion(0, 0, 0, 1)),
                    scale=Vector3(0.1, 0.1, 0.1),
                    color=point_color,
                    lifetime=rospy.Duration(1))
        self.line_pub.publish(marker)
        self.point_id += 1

    def callback(self, data):
        """ Essentially the main function of the program, this will call any functions
        required to get to the final answer: what set of points is a door
        Args:
            data (LaserScan): the data that the LaserScan returns
        """


        points = self.preprocessing(data)

        breakpoints = self.breakpoint_detection(points)

        for point in breakpoints:
            if point[3] == False and point[2] == False:
                self.show_point_in_rviz(point[0], ColorRGBA(0.0, 1.0, 0.0, 0.8))
            elif point[3] == False and point[2] == True:
                self.show_point_in_rviz(point[0], ColorRGBA(1.0, 0.0, 0.0, 0.8))
            elif point[3] == True and point[2] == False:
                self.show_point_in_rviz(point[0], ColorRGBA(0.0, 0.0, 1.0, 0.8))
            elif point[3] == True and point[2] == True:
                self.show_point_in_rviz(point[0], ColorRGBA(0.0, 1.0, 1.0, 0.8))
        #self.line_extraction(breakpoints)




    def preprocessing(self, data):
        """ This function preedits the scan points to attempt to remove noise (TODO)
        and then attempts to determine where a rupture occurs (a rupture is where the
        environment does not offer information to the laserscanner, like when a wall is too far away)
        the rupture points will be removed and flags will be set to indicate where
        points have been removed (this happens in rupture_detection, which is called in here)
        Args:
            data (LaserScan): the data that the LaserScan returns
        Returns:
            points (List): the points from the LaserScan in cartesian coordinates, relative
                           to the robot, where the ruptured points have been removed, and
                           if the point is next to a ruptured point.
                           [Point (cartesian Point), List (polar Point), ruptured]
        """

        scans = data.ranges

        # TODO preprocessing/compensation needed, this way the points are supposed to try and be more accurate



        points = []

        for i, scan in enumerate(scans):

            # setting default values
            point_to_add = Point()
            rupture = True

            # if the scan returns a value that is not infinity, the default values are not used
            if scan != float('inf'):
                pointX, pointY = self.polar_to_cartesian(scan, self.ANGLE_INCREMENT * (i + 1))
                point_to_add = Point(pointX, pointY, self.Z_OFFSET)
                rupture = False
            points.append([point_to_add, [scan, self.ANGLE_INCREMENT * (i + 1)], rupture])

        points = self.rupture_detection(points)

        return points


    def rupture_detection(self, points):
        """ This function removes any ruptured points and sets flags accordingly
        (TODO add example if that makes this clearer)
        Args:
            points (List): the points from the LaserScan in cartesian coordinates, relative
                           to the robot, and if the point is not to be counted due to a rupture
                           [Point (cartesian Point), List (polar Point), ruptured]
        Returns:
            rupture_flags (List): the points from the LaserScan in cartesian coordinates, relative
                                  to the robot, where the ruptured points have been removed, and
                                  if the point is next to a ruptured point.
                                  [Point (cartesian Point), List (polar Point), rupture]
        """

        rupture_flags = []
        add_rupture = False

        # [Point (cartesian Point), List (polar Point), ruptured]
        for point in points:

            # if there was no rupture on the current point
            if not point[2]:
                # add the current point as a point that will be passed on
                rupture_flags.append([point[0], point[1], False])

                # If there was a rupture on the point before this one
                if add_rupture:
                    # we set the rupture flag of the current point to true,
                    # this way we surround the points that have been left out
                    rupture_flags[-1][2] = True
                    add_rupture = False
            # if there is a rupture on the current point, we do not want to add it
            else:

                # We ensure there is at least one scan point before we set a rupture Flag
                if len(rupture_flags) > 0:
                    # we set the previous points rupture flag to true
                    rupture_flags[-1][2] = True
                    # we want to set the rupture flag of the next valid point to True as well
                    add_rupture = True


        return rupture_flags

    def breakpoint_detection(self, points):
        """
        This iterates through all the ruptureless points and will now attempt to find non connected objects.
        An example could be if a pillar is in front of a wall, but not connected that the program can detect
        and flag that.

        Args:
            points (List): the points from the LaserScan in cartesian coordinates, relative
                           to the robot, where the ruptured points have been removed, and
                           if the point is next to a ruptured point.
                           [Point (cartesian Point), List (polar Point), rupture]
        Returns:
            breakpoints (List): the input list with the additional flag of a breakpoint
                                [Point (cartesian Point), List (polar Point), rupture, breakpoint]
        """

        breakpoints = []

        # [Point (cartesian Point), List (polar Point), ruptured]
        last_point = points[0]
        breakpoints.append([last_point[0], last_point[1], last_point[2], False])
        for current_point in points[1:]:

            distance_max = last_point[1][0] * self.D_MAX_CONSTANT + (3 * self.SIGMA_R)
            # r_{n - 1} * (sin(delta phi) / sin(lambda - delta phi)) + 3 sigma_r

            # The parameters used in the text and thusly here are: sigma_r = 0.03 m, provided by the laser scan manufacturer, and lambda = 10

            breakpoints.append([current_point[0], current_point[1], last_point[2], False])
            if self.distance(last_point[0], current_point[0]) > distance_max:
                breakpoints[-2][3] = True
                breakpoints[-1][3] = True

        return breakpoints

    def line_extraction(self, breakpoints):
        """
        Args:
            breakpoints (List): the input list with the additional flag of a breakpoint
                                [Point (cartesian Point), List (polar Point), rupture, breakpoint]

        """

        list_of_lines = [] # a line is a list consisting of: p = polar discance of the line, a = polar angle, covariance matrix of (p, a)^T, xa = one end of the line, ya = same end only y coordinate, xb = other end x coordinate, yb = other end only y coordinate
        n_iterator = 0
        while n_iterator < len(breakpoints):
            n_start_of_region = n_iterator
            n_iterator = n_start_of_region + 1 # we will not look for the last point of the region
            while breakpoints[n_iterator][3] == False and breakpoints[n_iterator][2] == False:
                n_iterator = n_iterator + 1
                if n_iterator == len(breakpoints):
                    break
            #if (n_iterator - n_start_of_region + 1) > N_min: # N_min is minimum number of support points
                # L* <- (I T , n i , n e ) /* Extract lines from the current region */
                # L <- S union S* /* Add the lines to the main list */
