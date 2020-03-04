#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "ros/ros.h"
#include "std_msgs/String.h"
#include <iostream>
#include <geometry_msgs/Twist.h>

enum direc{RD, LD};

int main(int argc, char **argv) {
    ros::init(argc, argv, "pwm_control_node");
    ros::NodeHandle nh;

    ros::Publisher pub = nh.advertise<geometry_msgs::Twist>("twist_msg",1000);
    ros::Rate loop_rate(10);

    int count = 0;
    enum direc dir = RD;
    geometry_msgs::Twist msg;
    msg.linear.x = 1400;
    msg.angular.z = 1400;
    
    while(ros::ok()) {
	    if((msg.angular.z < 1800) && (dir == RD)) {
		msg.angular.z+=10;
		msg.linear.x+=10;
	    }
    	    else
		dir = LD;
	    if((msg.angular.z > 1400) && (dir == LD)) {
		msg.angular.z-=10;
		msg.linear.x-=10;
	    }
	    else
		dir = RD;
		    
	    ROS_INFO("Throttle = %f\nSteer = %f\n", \
			    msg.linear.x, \
			    msg.angular.z);
        pub.publish(msg);
		ros::spinOnce();
		loop_rate.sleep();
    }
}
