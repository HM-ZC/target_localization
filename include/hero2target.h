//
// Created by yuyu on 25-2-14.
//

#ifndef HERO2TAREGT_H
#define HERO2TAREGT_H

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Float64.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/convert.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf/transform_datatypes.h>
#include "geometry_msgs/Point.h"
//#include <rm_msgs/TrackData.h>
//#include <rm_msgs/TargetDetectionArray.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <std_msgs/Float32.h>
#include <cmath>
#include <tf2_msgs/TFMessage.h>
#include <dynamic_reconfigure/server.h>
#include <radar_vision/targetConfig.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include "radar_vision/TrackData.h"

namespace radar_vision{
    class Radar_vision {
    public:
        Radar_vision(ros::NodeHandle& nh);
        virtual ~Radar_vision() = default;
    private:
        void initialize();
        void loadStaticTransform();
        void tfCallback(const tf2_msgs::TFMessage::ConstPtr& msg);
        void count_y_Callback(const std_msgs::Float32::ConstPtr& msg);
        void count_x_Callback(const std_msgs::Float32::ConstPtr& msg);
        void count_z_Callback(const std_msgs::Float32::ConstPtr& msg);
        void dynamicReconfigCallback(radar_vision::targetConfig &config, uint32_t level);

        void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg);
        geometry_msgs::Point calculateAverageTargetFromPointCloud(
            const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
            const geometry_msgs::Point& center,
            double radius);

        std::shared_ptr<dynamic_reconfigure::Server<radar_vision::targetConfig>> server_ptr_;
        bool dynamic_reconfig_initialized_ = false;
        ros::NodeHandle nh;

        tf2_ros::Buffer tf_buffer_;
        tf2_ros::TransformListener tf_listener;
        tf2_ros::TransformBroadcaster tf_broadcaster;
        ros::Subscriber sub_;
        ros::Subscriber sub_count_x_;
        ros::Subscriber sub_count_y_;
        ros::Subscriber sub_count_z_;
        ros::Subscriber sub_pointcloud_;
        ros::Publisher pub_;
        ros::Publisher dis_pub_;
        ros::Publisher z_dis_pub_;
        ros::NodeHandle nh_;
        std::uint8_t targetid = 8;
        double target_x_{};
        double target_y_{};
        double target_z_{};

        int det_x;
        int det_y;
        int det_z;
        double current_value_ = 0;
        double step_x=0.07;
        double step_y=0.07;
        double step_z=0.07;
        double delta{};
        double pending_y_offset_{};
        double pending_x_offset_{};
        double pending_z_offset_{};
        geometry_msgs::PoseStamped map2target;
        geometry_msgs::PoseStamped odom2target;

        double search_radius_ = 2.0;
        bool use_pointcloud_refinement_ = true;
        double min_points_threshold_ = 10;  // 最小点数阈值
        std::string pointcloud_topic_ = "/corrected_current_pcd";
        bool pointcloud_received_ = false;

        geometry_msgs::Point current_target_position_;
        std::mutex target_mutex_;
    };
}

//class TfTransformer {
//public:
//    TfTransformer(ros::NodeHandle& nh);
//    void tfCallback(const tf2_msgs::TFMessage::ConstPtr& msg);
//    void dynamicReconfigCallback(radar_vision::targetConfig &config, uint32_t level);
//
//    geometry_msgs::TransformStamped calculateOdom2Target(const geometry_msgs::TransformStamped& map2odom,
//                                                         const geometry_msgs::TransformStamped& map2target);
//
//
//private:
//    void loadStaticTransform();
//    dynamic_reconfigure::Server<radar_vision::targetConfig> dyn_server_;
//    dynamic_reconfigure::Server<radar_vision::targetConfig>::CallbackType
//            callback_;
//    std::unique_ptr<dynamic_reconfigure::Server<Config>> image_converter_cfg_srv_;
//    dynamic_reconfigure::Server<Config>::CallbackType image_converter_cfg_cb_;
//    bool dynamic_reconfig_initialized_ = false;
//    ros::NodeHandle nh;
//
//    tf2::Quaternion q;
//    tf2::Quaternion q_map2odom, q_map2target, q_odom2target;
//
//    tf2_ros::Buffer tf_buffer_;
//    tf2_ros::TransformListener tf_listener;
//    tf2_ros::StaticTransformBroadcaster static_tf_broadcaster;
//    tf2_ros::TransformBroadcaster tf_broadcaster;
//    ros::Subscriber sub_;
//    ros::Publisher pub_;
//    ros::Publisher dis_pub_;
//    double roll, pitch, yaw;
//    std::uint8_t targetid=8;
//    bool iftracking = false;
//    double distance;
//    double target_x_;
//    double target_y_;
//    double target_z_;
//
//    geometry_msgs::PoseStamped map2target;
//    geometry_msgs::PoseStamped odom2target;
//};

#endif //HERO2TAREGT_H

