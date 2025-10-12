//
// Created by yuyu on 24-11-30.
//

//#include "/home/yuyu/workspace/rm/rm_ws/src/radar_vision/include/hero_radar.h"

// main.cpp
#include "hero2target.h"
#include <ros/ros.h>
#include <thread>


int main(int argc, char **argv) {
    ros::init(argc, argv, "radar_vision");

    ros::NodeHandle nh;
    radar_vision::Radar_vision tf_transformer(nh);
    ros::spin();
    return 0;
}