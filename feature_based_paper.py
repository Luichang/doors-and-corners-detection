#!/usr/bin/env python2
import rospy
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Quaternion, Pose, Point, Vector3
from std_msgs.msg import Header, ColorRGBA
import math, numpy as np
import tf
from scipy import stats




class ClusterPaper():
    """
    exploring what is being said and done in
        "Feature-Based Laser Scan Matching For Accurate and High Speed Mobile Robot Localization"
    """


    def __init__(self):

        # defining constants
        self.ANGLE_INCREMENT = 0.017501922324299812
        self.DISTANCE_THETA = 0.1 ** 2 # this constant is for the clustering process
        self.MINIMUM_CORNER_ANGLE = 0.1 # this constant is for seeing if 2 lines are angled enough to form a corner
        self.MINIMUM_LINE_LENGTH = 0.3 # this constant is for seeing if a line has the minumum length to be by a corner (?)
        self.MINIMUM_POINT_COUNT = 4 # this constant is for seeing if a line has a minumum amount of points
        self.RANGE_DIVIDENT = 3 # this constant is to limit the points we are looking at when it comes to finding corners
        self.Z_OFFSET = 0.1
        self.line_color = ColorRGBA(1, 0, 0, 0.7)

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
        self.line_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
        rospy.Subscriber('scan', LaserScan, self.callback)



        rospy.spin()


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


    def callback(self, data):
        """
        Args:
            data (LaserScan): the data that the LaserScan returns
        """
        clusters = self.clustering(data)

        # don't want to define corners here anymore. Corners will be features and unchanging
        # corners = [] # do I want this to be a variable of the class or just one that is passed around here?

        lines = []
        corners = self.potential_corners(clusters)

        # The following is terrible and to be removed
        # for potential_corner_list in corners:
        #     potential_corner = potential_corner_list[0]
        #     add_point = True
        #     for corner in self.corners:
        #         if self.distance(potential_corner, corner) < 0.1:
        #             add_point = False
        #             break
        #     if add_point:
        #         self.corners.append(potential_corner)
        #

        # TODO the points currently move because they are set relatively to the robot, which is how they are found
        # need to find a way to find the points relative to the robot and set them
        # relative to the fixed frame...
        self.corners.extend(corners)
        for corner in self.corners:
            self.show_point_in_rviz(corner[0].x, corner[0].y)

        # for cluster in clusters:
        #     if len(cluster) > 1:
        #         start_point, end_point, _, _ = self.detect_line(cluster)
        #
        #         self.show_line_in_rviz(start_point, end_point)

    #def create_dissimilarity_matrix(self, existing_cluster, new_cluster):



    # TODO Look into principle component Analyses for corner finding

    def potential_corners(self, clusters):
        """ This function aims to find the corners from the given data points
        At first this function looks to see that the two lines, given by the clusters,
        have a minimum angle 'self.MINIMUM_CORNER_ANGLE' between them. After that it
        is ensured that both wall segments have a minimum distance
        'self.MINIMUM_LINE_LENGTH' to each other.

        Once established that we are dealing with a corner the corner detection starts.
        At any point the algorithm will now be looking at two consecutive wall segments 'i and i + 1'.
        The assumed corner point 'potential_corner_point' is set to be the last scan point
        of the first of the two wall segments. Each point within a range 'r' to this point
        will be added to a list 'points_between_i_j' to be analyzed. The points farthest away but
        still within the range 'r' are set to pi and pj. From here we look for the point that
        is the farthest from the line created by pi and pj. If the point we find is our
        'potential_corner_point' we add this point as a corner point, otherwise we set
        'potential_corner_point' to the point we found and try again.



        Args:
            clusters (list): list of clusters found

        Returns:
            corner_points (list): list containing a potential corner point with the angles and clusters of the points it is combining
                                  the elements of the list are made up of
                                  |            0                           |  |           1              |
                                  [the point that should be the corner point, the angle of the second line,
                                  |           2                   | |         3               |  |         4                  |
                                  the cluster from the second line, the angle of the first line, the cluster of the first line]

        """

        lines = [] # list of lists with angle to x-axis, number of points in line, length of line
        corner_points = []
        for cluster in clusters:
            if len(cluster) > 1:
                start_point, end_point, slope, _ = self.detect_line(cluster)

                #            |   0           |  |     1    |  |            2                      |  |     3   |  |   4   |  |  5  |
                lines.append([math.atan(slope), len(cluster), self.distance(start_point, end_point), start_point, end_point, cluster])

        for i in range(len(lines) - 1):

            # The next set of if is to rule out line segments that are connected but not corners
            if abs(lines[i + 1][0] - lines[i][0]) <= self.MINIMUM_CORNER_ANGLE:
                # do something
                continue
            if (lines[i + 1][1] <= self.MINIMUM_POINT_COUNT or lines[i][1] <= self.MINIMUM_POINT_COUNT
                or lines[i + 1][2] <= self.MINIMUM_LINE_LENGTH or lines[i][2] <= self.MINIMUM_LINE_LENGTH):
                # do something again
                continue
            # At this point we want to think we are sure that we are dealing with a propper corner
            r = min(lines[i + 1][2], lines[i][2]) / self.RANGE_DIVIDENT

            potential_corner_point = lines[i][5][-1][2]
            #potential_corner_point = Point(x, y, self.Z_OFFSET)

            for j in range(20): # TODO 20 is a terrible hardcoded number but seems better that while True
                max_distance_point = Point()
                max_distance = 0
                points_between_i_j = []
                pi = Point() # lines[i][5][-1] # starting at the end of the cluster to go from closest to the point to farthest away
                pj = Point() # lines[i + 1][5][0]
                for point in reversed(lines[i][5]):
                    #_, _, x, y, _ = point

                    if self.distance(potential_corner_point, point[2]) > r:
                        break
                    pi = point[2]
                    points_between_i_j.append(point[2])

                # rospy.loginfo("There are {} elements in lines".format(len(lines)))
                # rospy.loginfo("which at {} has {} elements".format(i + 1, len(lines[i + 1])))
                for point in lines[i + 1][5]:
                    if self.distance(potential_corner_point, point[2]) > r:
                        break
                    pj = point[2]
                    points_between_i_j.append(point[2])

                for point in points_between_i_j:
                    if type(pi) != type(Point()) or type(pj) != type(Point()) or type(point) != type(Point()):
                        rospy.loginfo((pi))
                        rospy.loginfo(type(pi))
                        rospy.loginfo((pj))
                        rospy.loginfo(type(pj))
                        rospy.loginfo((point))
                        rospy.loginfo(type(point))
                    temp_distance = self.distance_line_to_point(pi, pj, point)
                    if temp_distance > max_distance:
                        max_distance = temp_distance
                        max_distance_point = point

                if potential_corner_point == max_distance_point:
                    break

                potential_corner_point = max_distance_point
                # TODO Somehow r needs to be decreased...

            corner_points.append([potential_corner_point, lines[i + 1][0], lines[i + 1][5], lines[i][0], lines[i][5]])

        for line in lines:
            self.show_line_in_rviz(line[3], line[4])

        #for corner in corner_points:
        #    self.show_point_in_rviz(corner[0].x, corner[0].y)

        return corner_points





    def clustering(self, data):
        """ This function is to find all clusters within the dataset
        The bulk of the paper that is being applied in the class is within this function, the clustering

        Args:
            data (LaserScan): the data that the LaserScan returns

        Returns:
            clusters (list): the list of clusters found
        """
        scans = data.ranges
        clusters = [] # this will be an array of arrays or other object that can holf many items
        points = []

        for i, scan in enumerate(scans):
            if scan != float('inf'):
                pointX, pointY = self.polar_to_cartesian(scan, self.ANGLE_INCREMENT * (i + 1))
                points.append([scan, i, Point(pointX, pointY, self.Z_OFFSET)])
        clusters.append([points[0]])

        for point in points[1:]:
            if self.distance(clusters[-1][-1][2], point[2]) < self.DISTANCE_THETA:
                clusters[-1].append(point)
            else:
                clusters.append([point])

        return clusters


    #def detect_features(self, clusters):
        """ This is supposed to find features within the created clusters. These
        Features will end up being things like corners

        """


    def detect_line(self, cluster):
        """ This function is to find the corner points of the cluster provided

        Args:
            cluster (list): A list of points

        Returns:
            start_point (Point): The first corner of the detected line
            end_point (Point): The end point of the detected line
            slope (float): the float vaulue equivalent to m in mx + b
            intercept (float): the float value equivalent to b in mx + b
        """

        # Linear Regression Method
        xs = []
        ys = []

        for c in cluster:
            x = c[2].x
            y = c[2].y
            xs.append(x)
            ys.append(y)
        xs = np.array(xs)
        ys = np.array(ys)

        slope, intercept, _, _, _ = stats.linregress(xs,ys)
        start_point = Point()
        end_point = Point()
        start_point.x = cluster[0][2].x
        start_point.y = intercept + slope * start_point.x

        end_point.x = cluster[-1][2].x
        end_point.y = intercept + slope * end_point.x


        # First and Last point method
        #start_point = cluster[0][2]
        #end_point = cluster[-1][2]
        return start_point, end_point, slope, intercept

    # def addCornerPoint(corner_list, corner):
    #     """ This function adds a corner to the existing list of corners. The intent is to be able to reuse corners at another time
    #
    #     Args:
    #         corner_list (list): The list of existing corners
    #         corner (Point): The point of a new corner that has been determined (temporary or certain?)
    #     """

    def polar_to_cartesian(self, distance, angle):
        """ This function converts polar coordinates to cartesian

        Args:
            distance (float): The distance that the Robot is away from the point that is being viewed
            angle (float): The angle that the Robot is away from the point that is being viewed

        Returns:
            pointX (float): the X coordinate
            pointY (float): the Y coordinate

        """
        pointX = distance * math.cos(angle)
        pointY = distance * math.sin(angle)
        return pointX, pointY

    def show_line_in_rviz(self, start_point, end_point):

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
        marker.colors.append(self.line_color)
        marker.colors.append(self.line_color)

        self.line_pub.publish(marker)

        self.marker_id += 1

    def show_point_in_rviz(self, pointX, pointY):
        """ This function takes an X and a Y to then place a Marker in the frame that is passed along

        Args:
            pointX (float): X coordinate
            pointY (float): Y coordinate
            (frame) (str): what frame the coordinates should be displayed on
        """
        marker = Marker(
                    header=Header(
                    frame_id=self.base_marker_header_frame_id),
                    #frame_id='odom'),
                    id=self.point_id,
                    type=Marker.SPHERE,
                    pose=Pose(Point(pointX, pointY, self.Z_OFFSET), Quaternion(0, 0, 0, 1)),
                    scale=Vector3(0.1, 0.1, 0.1),
                    color=ColorRGBA(0.0, 1.0, 0.0, 0.8),
                    lifetime=rospy.Duration(1))
        self.line_pub.publish(marker)
        self.point_id += 1

if __name__ == '__main__':
    ClusterPaper()
