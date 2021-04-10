# door-and-wall-detection
This repo intends to yield the locations of doors, corners, walls and corridor entrances. 

Needed for this is a working instalation of ROS of which an instalation can be found here http://wiki.ros.org/ROS/Installation

The folder doors_and_corners/ is the base. This is to be placed into the catkin_ws/src folder. As custom messages are used you will also need to run catkin make. The CMakeLists.txt and package.xml files should be already confiured where that is all that is needed to get setup.

Once ROS and the project are setup you can start running the launch file and see what results this repository has come up with. If you would like to use your own data simply place a new rosbag in the rosbags folder and edit the launch file to include your new rosbag.

In the scripts folder is where the magic happens. The file line_extraction_paper.py takes all points and finds the points relevant for doors, corners, walls and corridor entrances. 
