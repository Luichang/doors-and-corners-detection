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
        self.ANGLE_INCREMENT = 0 # 0.017501922324299812 # degrees
        self.ANGLES = []

        self.SIGMA_R = 0.03 # meters
        self.LAMBDA = 10 # degrees
        self.D_MAX_CONSTANT = math.sin(math.radians(self.ANGLE_INCREMENT))/math.sin(math.radians(self.LAMBDA-self.ANGLE_INCREMENT))


        self.Z_OFFSET = 0.1

        # marker constants
        self.base_marker_type = Marker.LINE_STRIP
        self.base_marker_lifetime = rospy.Duration(0.3)
        self.base_marker_header_frame_id = 'cloud'
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

    def distance_line_to_point(self, p1, p2, p3):
        """ This function finds the shortest distance between a line defined by two points and another point
        The distance of the line calculated will always be perpendicular to the line the first two points span

        Args:
            p1 (Point): one point of the line. The distance to this line is what we want to calculate
            p2 (Point): the other point of the line. The distance to this line is what we want to calculate
            p3 (Point): this is the point whose distance to the line we are interested in

        Returns:
            dist (float): the distance from the point to the line
        """
        numerator = abs((p2.y - p1.y) * p3.x - (p2.x - p1.x) * p3.y + p2.x * p1.y - p2.y * p1.x)
        denominator = math.sqrt((p2.y - p1.y) ** 2 + (p2.x - p1.x) ** 2)
        dist = numerator / denominator
        return dist

    def angle_between_points(self, point1, point2):
        """
        Calculates the angle between 2 points.

        Args:
            point1 (Point): first point
            point2 (Point): second point

        Returns:
            deg (float): Angle in degrees between the two points
        """
        x1 = point1.x
        y1 = point1.y
        x2 = point2.x
        y2 = point2.y
        dX = x2 - x1
        dY = y2 - y1
        rads = math.atan2(-dY, dX)
        deg = math.degrees(rads)
        return deg

    def angle_between_lines(self, line1, line2):
        """
        Calculates the angle between 2 lines by first determining the angle of the individual
        lines and then subtracting them from each other

        Args:
            line1 (List): List of start and end Point of the line
            line2 (List): List of start and end Point of the line

        Returns:
            angle_of_lines (float): angle between the two lines in degrees
        """
        angle_line1 = self.angle_between_points(line1[0], line1[1])
        angle_line2 = self.angle_between_points(line2[0], line2[1])
        angle_of_lines = abs(angle_line1 - angle_line2)
        return angle_of_lines

    def create_wall(self, start_point, end_point):
        """
        This function exsists solely to create a uniform wall object. It is important to note,
        going from the start_point to the end_point, if one looks to the left side of that, there
        is to be empty, traversable space.

        Args:
            start_point (Point): the start point of the wall
            end_point (Point): the end point of the wall

        Returns:
            wall (List): list containing the first and last point of a wall, indicating the
                         straight line segment created by the wall
        """
        return [start_point, end_point]

    def create_corner(self, corner_list, first_wall, second_wall, inner_corner=True):
        """
        This function exists to create a uniform corner type. The input is the 2 touching walls
        The output is the corner connecting the two walls and the boolean of if the corner is an
        inside corner (meaning the angle we can see from our current position is < 180, usually 90)
        or an outside corner (meaning the angle we can see from our current position is > 180, usually 270)

        Args:
            corner_list (List): the list that will contain all existing corners
            first_wall (List): the list of the points describing the first wall
            second_wall (List): the list of the points describing the second wall
            (inner_corner (bool)): optional boolean to determine, if the corner point is an inner point
        """
        if first_wall[0] in second_wall:
            corner_list.append([first_wall[0], inner_corner])
        elif first_wall[1] in second_wall:
            corner_list.append([first_wall[1], inner_corner])

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
                    frame_id='cloud'), # odom is the fixed frame, it sounds like with real data that will not be available?
                    id=self.point_id,
                    type=Marker.SPHERE,
                    pose=Pose(point, Quaternion(0, 0, 0, 1)),
                    scale=Vector3(0.1, 0.1, 0.1),
                    color=point_color,
                    lifetime=self.base_marker_lifetime)
        self.line_pub.publish(marker)
        self.point_id += 1

    def show_line_in_rviz(self, start_point, end_point, line_color=ColorRGBA(1, 0, 0, 0.7)):
        """ This function takes two points to then place a Marker line in the frame
        With an optional argument to set the color

        Args:
            start_point       (Point): Start Point from the line to be displayed
            end_point         (Point): End Point from the line to be displayed
            (point_color      (ColorRGBA)): Optional color argument to change the color of the point
        """
        marker = Marker()
        marker.type = self.base_marker_type
        marker.id = self.marker_id
        marker.lifetime = self.base_marker_lifetime
        marker.header.frame_id = self.base_marker_header_frame_id
        marker.action = self.base_marker_action
        marker.scale.x = self.base_marker_scale_x
        marker.pose.orientation = self.base_marker_pose_orientation

        marker.points.append(start_point)
        marker.points.append(end_point)
        marker.colors.append(line_color)
        marker.colors.append(ColorRGBA(0, 0, 1, 0.7))

        self.line_pub.publish(marker)

        self.marker_id += 1

    def print_wall(self, wall):
        """
        Function to show the lines of a wall or door

        Args:
            wall (List): list of points defining the boundry of the wall
        """
        line_color = ColorRGBA(1, 0, 0, 0.7)
        if self.distance(wall[0], wall[1]) > 0.5 and self.distance(wall[0], wall[1]) < 1:
            line_color = ColorRGBA(0, 1, 0, 0.7)
        self.show_line_in_rviz(wall[0], wall[1], line_color)

    def print_corner(self, corner):
        """
        Function to show the current corner

        Args:
            corner (List): list containing a point, representing the corner and a boolean
                           representing if the corner is an inside corner (when True)
        """
        corner_color = ColorRGBA(1, 0, 0, 0.7)
        if corner[1]:
            corner_color = ColorRGBA(0, 1, 0, 0.7)
        self.show_point_in_rviz(corner[0], corner_color)

    def callback(self, data):
        """ Essentially the main function of the program, this will call any functions
        required to get to the final answer: what set of points is a door
        Args:
            data (LaserScan): the data that the LaserScan returns
        """

        if not self.ANGLES:
            self.ANGLE_INCREMENT = data.angle_increment
            self.ANGLES = [x for x in np.arange(data.angle_min, data.angle_max, self.ANGLE_INCREMENT)]
            self.D_MAX_CONSTANT = math.sin(math.radians(self.ANGLE_INCREMENT))/math.sin(math.radians(self.LAMBDA-self.ANGLE_INCREMENT))

        points = self.preprocessing(data)

        breakpoints = self.breakpoint_detection(points)

        # for point in breakpoints:
        #     #  breakpoint            rupture
        #     if point[3] == False and point[2] == False:
        #         self.show_point_in_rviz(point[0], ColorRGBA(0.0, 1.0, 0.0, 0.8))
        #     elif point[3] == False and point[2] == True:
        #         self.show_point_in_rviz(point[0], ColorRGBA(1.0, 0.0, 0.0, 0.8))
        #     elif point[3] == True and point[2] == False:
        #         self.show_point_in_rviz(point[0], ColorRGBA(0.0, 0.0, 1.0, 0.8))
        #     elif point[3] == True and point[2] == True:
        #         self.show_point_in_rviz(point[0], ColorRGBA(0.0, 1.0, 1.0, 0.8))
        self.line_extraction(breakpoints)




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

        for i, scan in zip(self.ANGLES, scans):

            # setting default values
            point_to_add = Point()
            rupture = True

            # if the scan returns a value that is not infinity, the default values are not used
            if scan != 0.0:
                pointX, pointY = self.polar_to_cartesian(scan, i) #self.ANGLE_INCREMENT * (i + 1))
                point_to_add = Point(pointX, pointY, self.Z_OFFSET)
                rupture = False
            points.append([point_to_add, [scan, i], rupture]) #self.ANGLE_INCREMENT * (i + 1)], rupture])

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
            last_point = current_point

        return breakpoints

    def line_extraction(self, breakpoints):
        """
        Args:
            breakpoints (List): the input list with the additional flag of a breakpoint
                                [Point (cartesian Point), List (polar Point), rupture, breakpoint]

        """

        list_of_lines = [] # a line is a list consisting of:
        # p = polar discance of the line, a = polar angle, covariance matrix of (p, a)^T,
        # xa = one end of the line, ya = same end only y coordinate,
        # xb = other end x coordinate, yb = other end only y coordinate
        list_of_points_for_lines = []
        n_iterator = 0
        while n_iterator < len(breakpoints) - 1:
            n_start_of_region = n_iterator
            n_iterator = n_start_of_region + 1 # we will not look for the last point of the region

            # this loop groups continuous wall segments
            while breakpoints[n_iterator][3] == False and breakpoints[n_iterator][2] == False:
                n_iterator = n_iterator + 1
                if n_iterator >= len(breakpoints) - 1:
                    break

            # at this point we have a continuous line segment. This segment can span over multiple walls, connected by corners.
            # the goal now is to split the line segments at the corners to receive not only walls but hopefully also doors.

            N_min = 3
            # for refference, n_i = n_start_of_region and n_e = n_iterator
            if (n_iterator - n_start_of_region + 1) > N_min: # N_min is minimum number of support points

                # L* <- Phi(I^T , n_i , n_e ) /* Extract lines from the current region */ Phi is a Kernel for line extraction
                self.iterative_end_point_fit(list_of_points_for_lines, breakpoints, n_start_of_region, n_iterator)

                list_of_corners = []
                if list_of_points_for_lines:
                    for line_index in reversed(range(len(list_of_points_for_lines))):
                        if line_index > 0:
                            if list_of_points_for_lines[line_index][0] in list_of_points_for_lines[line_index - 1] or list_of_points_for_lines[line_index][1] in list_of_points_for_lines[line_index - 1]:
                                angle_of_lines = self.angle_between_lines(list_of_points_for_lines[line_index], list_of_points_for_lines[line_index - 1])
                                if angle_of_lines < 2:
                                    # if we get in here the corner that has been detected is not an actual corner and should be removed
                                    list_of_points_for_lines[line_index - 1][0] = list_of_points_for_lines[line_index][0]
                                    list_of_points_for_lines.pop(line_index)
                                    continue
                        self.print_wall(list_of_points_for_lines[line_index])
                        for second_line_index in range(line_index, len(list_of_points_for_lines)):
                            if list_of_points_for_lines[line_index][1] == list_of_points_for_lines[second_line_index][0]:
                                self.create_corner(list_of_corners, list_of_points_for_lines[line_index], list_of_points_for_lines[second_line_index])
                                break

                            # you could think that the symmetrical case would allso occure, but at least the way the scans are made and walls created it
                            # does not seem to come up
                            # if list_of_points_for_lines[line_index][0] == list_of_points_for_lines[second_line_index][1]:
                            #     self.create_corner(list_of_corners, list_of_points_for_lines[line_index], list_of_points_for_lines[second_line_index], False)
                            #     break



                if list_of_corners:
                    for corner in list_of_corners:
                        self.print_corner(corner)
                # temp list of points that make up the whole line?

                # L <- Omega^S union Omega^S_* /* Add the lines to the main list */







    def iterative_end_point_fit(self, list_of_points_for_lines, breakpoints, start_of_region, end_of_region):
        """
        Rekursive Function
        checks the given range of start_of_region and end_of_region if the continuous wall segment contains corners
        and splits at these corners and checks both sides of the corner for more corners. Once it does not find any
        more corners it adds the Points of the begining and end of the wall to list_of_points_for_lines

        Args:
            list_of_points_for_lines (List): List of lists, inner list contains first and last Point of a line segment
                                             without corners
            breakpoints (List):              List of all points with the flags as set by breakpoint_detection
            start_of_region (int):           index of point indicating the start of a wall segment to be checked for corners
            end_of_region (int):             index of point indicating the end of a wall segment to be checked for corners

        """
        minimum_distance_to_be_a_corner = 0.03 # meter, the value set is a guess and may need adjusting
        N_min = 3 # this probably should be turned into a variable part of self
        if (end_of_region - start_of_region + 1) <= N_min:
            return None
        max_distance = 0
        farthest_point = -1
        for potential_corner in range(start_of_region + 1, end_of_region):
            distance_to_line = self.distance_line_to_point(breakpoints[start_of_region][0], breakpoints[end_of_region][0], breakpoints[potential_corner][0])
            if distance_to_line > minimum_distance_to_be_a_corner and distance_to_line > max_distance:
                max_distance = distance_to_line
                farthest_point = potential_corner

        if farthest_point == -1:
            list_of_points_for_lines.append([breakpoints[start_of_region][0], breakpoints[end_of_region][0]])
        else:
            self.iterative_end_point_fit(list_of_points_for_lines, breakpoints, start_of_region, farthest_point)
            self.iterative_end_point_fit(list_of_points_for_lines, breakpoints, farthest_point, end_of_region)
