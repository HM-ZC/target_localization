#include "hero2target.h"
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <boost/bind.hpp>

namespace radar_vision {

    Radar_vision::Radar_vision(ros::NodeHandle& nh)
            : nh_("~"),
              tf_listener(tf_buffer_)

    {
        server_ptr_ = std::make_shared<dynamic_reconfigure::Server<radar_vision::targetConfig>>(nh_);

        dynamic_reconfigure::Server<radar_vision::targetConfig>::CallbackType cb;
        cb = boost::bind(&Radar_vision::dynamicReconfigCallback, this, _1, _2);
        server_ptr_->setCallback(cb);

        ROS_WARN("[Radar] Dynamic server initialized at namespace: %s",
                 nh_.getNamespace().c_str());
        initialize();
    }

    void Radar_vision::initialize()
    {
        loadStaticTransform();
        sub_ = nh.subscribe("/tf", 10, &Radar_vision::tfCallback, this);
        sub_count_y_ = nh.subscribe("/rm_manual/target_y/total_target_y", 10, &Radar_vision::count_y_Callback, this);
        sub_count_x_ = nh.subscribe("/rm_manual/target_x/total_target_x", 10, &Radar_vision::count_x_Callback, this);
        sub_count_z_ = nh.subscribe("/rm_manual/target_z/total_target_z", 10, &Radar_vision::count_z_Callback, this);

        sub_pointcloud_ = nh.subscribe(pointcloud_topic_, 1, &Radar_vision::pointCloudCallback, this);

        pub_ = nh.advertise<radar_vision::TrackData>("/odom2target", 10);
        dis_pub_ = nh.advertise<std_msgs::Float32>("/dis_baselink2target", 10);
        z_dis_pub_ = nh.advertise<std_msgs::Float32>("/z_dis_baselink2target", 10);

        ROS_INFO("Dynamic reconfigure server initialized!");
        ROS_INFO("PointCloud subscriber initialized for topic: %s", pointcloud_topic_.c_str());
    }

    void Radar_vision::dynamicReconfigCallback(radar_vision::targetConfig &config, uint32_t level) {
        if (!dynamic_reconfig_initialized_) {
            config.target_x = target_x_;
            config.target_y = target_y_;
            config.target_z = target_z_;
            dynamic_reconfig_initialized_ = true;
            return;
        }
        target_x_ = config.target_x;
        target_y_ = config.target_y;
        target_z_ = config.target_z;

        map2target.pose.position.x = target_x_;
        map2target.pose.position.y = target_y_;
        map2target.pose.position.z = target_z_;

        std::lock_guard<std::mutex> lock(target_mutex_);
        current_target_position_.x = target_x_;
        current_target_position_.y = target_y_;
        current_target_position_.z = target_z_;

        ROS_INFO("Set Complete: target position [%f, %f, %f]", target_x_, target_y_, target_z_);
    }

    void Radar_vision::loadStaticTransform() {
        nh.param("static_transform/translation/x", map2target.pose.position.x, -21.5);
        nh.param("static_transform/translation/y", map2target.pose.position.y, -0.7);
        nh.param("static_transform/translation/z", map2target.pose.position.z, 1.043);

        nh.param("static_transform/target_x", target_x_, map2target.pose.position.x);
        nh.param("static_transform/target_y", target_y_, map2target.pose.position.y);
        nh.param("static_transform/target_z", target_z_, map2target.pose.position.z);

        nh.param("pointcloud_refinement/search_radius", search_radius_, 2.0);
        nh.param("pointcloud_refinement/use_refinement", use_pointcloud_refinement_, true);
        nh.param("pointcloud_refinement/min_points_threshold", min_points_threshold_, 10.0);
        nh.param("pointcloud_refinement/pointcloud_topic", pointcloud_topic_, std::string("/current_pcd"));

        nh.param("step_x", step_x, 0.07);
        nh.param("step_y", step_y, 0.07);
        nh.param("step_z", step_z, 0.07);

        int temp_target_id;
        nh.param("target_id", temp_target_id, 8);
        targetid = static_cast<std::uint8_t>(temp_target_id);

        map2target.pose.position.x = target_x_;
        map2target.pose.position.y = target_y_;
        map2target.pose.position.z = target_z_;

        current_target_position_.x = target_x_;
        current_target_position_.y = target_y_;
        current_target_position_.z = target_z_;

        nh.setParam("target_x", target_x_);
        nh.setParam("target_y", target_y_);
        nh.setParam("target_z", target_z_);

        ROS_INFO("Target configuration loaded:");
        ROS_INFO("  Position: [%.3f, %.3f, %.3f]", target_x_, target_y_, target_z_);
        ROS_INFO("  Steps: [%.3f, %.3f, %.3f]", step_x, step_y, step_z);
        ROS_INFO("  Target ID: %d", targetid);
        ROS_INFO("PointCloud refinement parameters:");
        ROS_INFO("  Topic: %s", pointcloud_topic_.c_str());
        ROS_INFO("  Radius: %.2f, Enabled: %s, Min points: %.0f",
                 search_radius_, use_pointcloud_refinement_ ? "true" : "false", min_points_threshold_);
    }

    void Radar_vision::pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
            if (!use_pointcloud_refinement_) {
                return;
            }

            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::fromROSMsg(*msg, *cloud);

            if (cloud->empty()) {
                ROS_WARN("Received empty point cloud!");
                return;
            }

            geometry_msgs::Point search_center;
            {
                std::lock_guard<std::mutex> lock(target_mutex_);
                search_center = current_target_position_;
                search_center.x += pending_x_offset_;
                search_center.y += pending_y_offset_;
                search_center.z += pending_z_offset_;
            }

            geometry_msgs::Point refined_target = calculateAverageTargetFromPointCloud(cloud, search_center, search_radius_);

            if (!std::isnan(refined_target.x) && !std::isnan(refined_target.y) && !std::isnan(refined_target.z)) {
                std::lock_guard<std::mutex> lock(target_mutex_);
                current_target_position_ = refined_target;

                pending_x_offset_ = 0.0;
                pending_y_offset_ = 0.0;
                pending_z_offset_ = 0.0;

                pointcloud_received_ = true;

                ROS_INFO_THROTTLE(2.0, "Target refined by pointcloud: [%.3f, %.3f, %.3f] -> [%.3f, %.3f, %.3f]",
                                search_center.x, search_center.y, search_center.z,
                                refined_target.x, refined_target.y, refined_target.z);

                ROS_INFO_THROTTLE(5.0, "Manual offsets cleared after successful pointcloud refinement");
            }
    }

    geometry_msgs::Point Radar_vision::calculateAverageTargetFromPointCloud(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
        const geometry_msgs::Point& center,
        double radius) {

        geometry_msgs::Point result;
        result.x = result.y = result.z = std::numeric_limits<double>::quiet_NaN();

        std::vector<pcl::PointXYZ> nearby_points;

        for (const auto& point : cloud->points) {
            double distance = sqrt(
                pow(point.x - center.x, 2) +
                pow(point.y - center.y, 2) +
                pow(point.z - center.z, 2)
            );

            if (distance <= radius) {
                nearby_points.push_back(point);
            }
        }

        if (nearby_points.size() < min_points_threshold_) {
            ROS_WARN_THROTTLE(5.0, "Not enough points (%zu) within radius %.2f around target [%.2f, %.2f, %.2f]",
                            nearby_points.size(), radius, center.x, center.y, center.z);
            return result;
        }

        double sum_x = 0, sum_y = 0, sum_z = 0;
        for (const auto& point : nearby_points) {
            sum_x += point.x;
            sum_y += point.y;
            sum_z += point.z;
        }

        result.x = sum_x / nearby_points.size();
        result.y = sum_y / nearby_points.size();
        result.z = sum_z / nearby_points.size();

        ROS_DEBUG("Found %zu points within radius %.2f, average position: [%.3f, %.3f, %.3f]",
                 nearby_points.size(), radius, result.x, result.y, result.z);

        return result;
    }

    void Radar_vision::count_y_Callback(const std_msgs::Float32::ConstPtr& msg) {
        if (fabs(msg->data) < 0.001f) {
            ROS_DEBUG("Ignoring near-zero input: %.3f", msg->data);
            return;
        }

        float new_offset = msg->data * step_y;

        if (fabs(new_offset - pending_y_offset_) > 0.001f) {
            pending_y_offset_ = new_offset;
            ROS_INFO("Counter updated: Δy=%.3f (input=%.1f * step=%.3f)",
                     pending_y_offset_, msg->data, step_y);
        } else {
            ROS_DEBUG("Ignoring duplicate input: %.1f", msg->data);
        }
    }

    void Radar_vision::count_x_Callback(const std_msgs::Float32::ConstPtr& msg) {
        if (fabs(msg->data) < 0.001f) {
            ROS_DEBUG("Ignoring near-zero input: %.3f", msg->data);
            return;
        }

        float new_offset = msg->data * step_x;

        if (fabs(new_offset - pending_x_offset_) > 0.001f) {
            pending_x_offset_ = new_offset;
            ROS_INFO("Counter updated: Δx=%.3f (input=%.1f * step=%.3f)",
                     pending_x_offset_, msg->data, step_x);
        } else {
            ROS_DEBUG("Ignoring duplicate input: %.1f", msg->data);
        }
    }

    void Radar_vision::count_z_Callback(const std_msgs::Float32::ConstPtr& msg) {
        if (fabs(msg->data) < 0.001f) {
            ROS_DEBUG("Ignoring near-zero input: %.3f", msg->data);
            return;
        }

        float new_offset = msg->data * step_z;

        if (fabs(new_offset - pending_z_offset_) > 0.001f) {
            pending_z_offset_ = new_offset;
            ROS_INFO("Counter updated: Δz=%.3f (input=%.1f * step=%.3f)",
                     pending_z_offset_, msg->data, step_z);
        } else {
            ROS_DEBUG("Ignoring duplicate input: %.1f", msg->data);
        }
    }

    void Radar_vision::tfCallback(const tf2_msgs::TFMessage::ConstPtr& msg) {
        for (const auto& transform : msg->transforms) {
            if (transform.child_frame_id == "odom" && transform.header.frame_id == "map") {
                ros::Time timestamp = transform.header.stamp;
                try {
                    geometry_msgs::TransformStamped odom_transform =
                            tf_buffer_.lookupTransform("odom", "map", timestamp, ros::Duration(1));

                    geometry_msgs::PoseStamped target_to_transform;
                    {
                        std::lock_guard<std::mutex> lock(target_mutex_);
                        if (use_pointcloud_refinement_ && pointcloud_received_) {
                            target_to_transform.pose.position = current_target_position_;
                        } else {
                            target_to_transform.pose.position.x = map2target.pose.position.x + pending_x_offset_;
                            target_to_transform.pose.position.y = map2target.pose.position.y + pending_y_offset_;
                            target_to_transform.pose.position.z = map2target.pose.position.z + pending_z_offset_;
                        }
                    }
                    target_to_transform.pose.orientation.w = 1.0;

                    tf2::doTransform(target_to_transform.pose, odom2target.pose, odom_transform);

                    geometry_msgs::TransformStamped odom_target_tf;
                    odom_target_tf.header.stamp = timestamp;
                    odom_target_tf.header.frame_id = "odom";
                    odom_target_tf.child_frame_id = "target";
                    odom_target_tf.transform.translation.x = odom2target.pose.position.x;
                    odom_target_tf.transform.translation.y = odom2target.pose.position.y;
                    odom_target_tf.transform.translation.z = odom2target.pose.position.z;
                    odom_target_tf.transform.rotation = tf2::toMsg(tf2::Quaternion(0, 0, 0, 1));
                    tf_broadcaster.sendTransform(odom_target_tf);

                    geometry_msgs::TransformStamped base_link_tf =
                            tf_buffer_.lookupTransform("base_link", "target", timestamp, ros::Duration(1));

                tf2::Vector3 xy_position(
                        base_link_tf.transform.translation.x,
                        base_link_tf.transform.translation.y,
                        0.0
                );
                double xy_distance = xy_position.length();
                tf2::Vector3 z_position(
                        0.0,
                        0.0,
                        base_link_tf.transform.translation.z
                );
                double z_distance = z_position.length();
                std_msgs::Float32 xyDistanceMsg;
                xyDistanceMsg.data = xy_distance;
                dis_pub_.publish(xyDistanceMsg);
                std_msgs::Float32 zDistanceMsg;
                zDistanceMsg.data = z_distance;
                z_dis_pub_.publish(zDistanceMsg);

                    radar_vision::TrackData trackDataMsg;
                    trackDataMsg.header.stamp = timestamp;
                    trackDataMsg.id = targetid;
                    trackDataMsg.tracking = true;
                    trackDataMsg.armors_num = 1;
                    trackDataMsg.header.frame_id = "odom";
                    trackDataMsg.position.x = odom2target.pose.position.x;
                    trackDataMsg.position.y = odom2target.pose.position.y;
                    trackDataMsg.position.z = odom2target.pose.position.z;
                    pub_.publish(trackDataMsg);
                }
                catch (tf2::TransformException& ex) {
                    ROS_WARN("%s", ex.what());
                }
            }
        }
    }

} // namespace radar_vision