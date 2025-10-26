#include "hero2target.h"
#include "dbscan.hpp"
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <boost/bind.hpp>
#include <nanoflann.hpp>

using gtsam::Pose3;
using gtsam::Point3;
using gtsam::Rot3;
using gtsam::noiseModel::Diagonal;
using gtsam::Symbol;

namespace {  // 仅此编译单元可见
template <typename T>
struct PCLPointCloudAdaptor {
    const pcl::PointCloud<pcl::PointXYZ>& cloud;
    explicit PCLPointCloudAdaptor(const pcl::PointCloud<pcl::PointXYZ>& c) : cloud(c) {}

    inline size_t kdtree_get_point_count() const { return cloud.size(); }

    inline T kdtree_get_pt(const size_t idx, const size_t dim) const {
        const auto& p = cloud.points[idx];
        if (dim == 0) return static_cast<T>(p.x);
        if (dim == 1) return static_cast<T>(p.y);
        return static_cast<T>(p.z);
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};
} // anonymous namespace

namespace radar_vision {

    Radar_vision::Radar_vision(ros::NodeHandle& nh)
            : nh_("~")

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
        sub_count_y_ = nh.subscribe("/rm_manual/target_y/total_target_y", 10, &Radar_vision::count_y_Callback, this);
        sub_count_x_ = nh.subscribe("/rm_manual/target_x/total_target_x", 10, &Radar_vision::count_x_Callback, this);
        sub_count_z_ = nh.subscribe("/rm_manual/target_z/total_target_z", 10, &Radar_vision::count_z_Callback, this);

        sub_pointcloud_ = nh.subscribe(pointcloud_topic_, 1, &Radar_vision::pointCloudCallback, this);

        pub_ = nh.advertise<radar_vision::TrackData>("/odom2target", 10);
        dis_pub_ = nh.advertise<std_msgs::Float32>("/dis_baselink2target", 10);
        z_dis_pub_ = nh.advertise<std_msgs::Float32>("/z_dis_baselink2target", 10);

        target_marker_pub_ = nh.advertise<visualization_msgs::Marker>("/radar_vision/target_marker", 1);
        best_cluster_pub_  = nh.advertise<sensor_msgs::PointCloud2>("/radar_vision/best_cluster", 1);


        ROS_INFO("Dynamic reconfigure server initialized!");
        ROS_INFO("PointCloud subscriber initialized for topic: %s", pointcloud_topic_.c_str());

        initializeGtsam();
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

        nh.param("gtsam/use_optimization", use_gtsam_optimization_, true);
        nh.param("gtsam/meas_sigma_xy",    meas_sigma_xy_,    0.10);
        nh.param("gtsam/meas_sigma_z",     meas_sigma_z_,     0.10);
        nh.param("gtsam/smooth_sigma_xy",  smooth_sigma_xy_,  0.05);
        nh.param("gtsam/smooth_sigma_z",   smooth_sigma_z_,   0.05);
        nh.param("gtsam/prior_sigma_xy",   prior_sigma_xy_,   0.20);
        nh.param("gtsam/prior_sigma_z",    prior_sigma_z_,    0.20);
        nh.param("gtsam/orient_sigma_loose", orient_sigma_loose_, 100.0);

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

        nh.param("refine/roi_radius", roi_radius_, 2.5);
        nh.param("refine/voxel_leaf", voxel_leaf_, 0.05);
        nh.param("refine/use_sor", use_sor_, true);
        nh.param("refine/sor_mean_k", sor_mean_k_, 20);
        nh.param("refine/sor_stddev_mul", sor_stddev_mul_, 1.0);

        nh.param("clustering/dbscan_eps", dbscan_eps_, 0.25);
        nh.param("clustering/dbscan_min_pts", dbscan_min_pts_, 20);
        nh.param("clustering/min_points", cluster_min_size_, 30);
        nh.param("clustering/max_points", cluster_max_size_, 50000);

        nh.param("nanogicp/threads", ng_threads_, 4);
        nh.param("nanogicp/correspondences", ng_corr_rand_, 15);
        nh.param("nanogicp/max_iter", ng_max_iter_, 60);
        nh.param("nanogicp/ransac_max_iter", ng_ransac_max_iter_, 5);
        nh.param("nanogicp/max_corr_dist", ng_max_corr_dist_, 1.0);
        nh.param("nanogicp/icp_score_thr", ng_icp_score_thr_, 10.0);
        nh.param("nanogicp/trans_eps", ng_trans_eps_, 1e-3);
        nh.param("nanogicp/euclid_fit_eps", ng_euclid_fit_eps_, 1e-3);
        nh.param("nanogicp/ransac_outlier_thr", ng_ransac_outlier_thr_, 1.0);

        // 新增：时序配准参数
        nh.param("temporal_registration/enable", use_temporal_registration_, true);
        nh.param("temporal_registration/max_dt", temporal_max_dt_, 0.3);

        // 配置 NanoGICP 参数
        nano_gicp_.setNumThreads(ng_threads_);
        nano_gicp_.setCorrespondenceRandomness(ng_corr_rand_);
        nano_gicp_.setMaximumIterations(ng_max_iter_);
        nano_gicp_.setRANSACIterations(ng_ransac_max_iter_);
        nano_gicp_.setMaxCorrespondenceDistance(ng_max_corr_dist_);
        nano_gicp_.setTransformationEpsilon(ng_trans_eps_);
        nano_gicp_.setEuclideanFitnessEpsilon(ng_euclid_fit_eps_);
        nano_gicp_.setRANSACOutlierRejectionThreshold(ng_ransac_outlier_thr_);

        ROS_INFO("Temporal registration: %s, max_dt=%.2fs",
                 use_temporal_registration_ ? "ON" : "OFF", temporal_max_dt_);
    }

    void Radar_vision::initializeGtsam() {
        std::lock_guard<std::mutex> lk(gtsam_mutex_);
        gtsam::ISAM2Params params;
        params.relinearizeThreshold = 0.01;
        params.relinearizeSkip = 10;
        isam2_ = gtsam::ISAM2(params);
        graph_.resize(0);
        initial_.clear();
        gtsam_index_ = 0;
        gtsam_inited_ = true;
        gtsam_has_estimate_ = false;
    }

    void Radar_vision::resetGtsam(const geometry_msgs::Point& p0) {
        if (!gtsam_inited_) initializeGtsam();

        // 重新构建 iSAM2
        initializeGtsam();

        const auto key0 = Symbol('x', 0);
        const Pose3 pose0(gtsam::Rot3::Quaternion(1,0,0,0), Point3(p0.x, p0.y, p0.z));

        // Prior: 仅平移有约束，姿态给松噪声
        auto prior_noise = Diagonal::Sigmas((gtsam::Vector(6) <<
            orient_sigma_loose_, orient_sigma_loose_, orient_sigma_loose_,
            prior_sigma_xy_, prior_sigma_xy_, prior_sigma_z_).finished());

        graph_.add(gtsam::PriorFactor<Pose3>(key0, pose0, prior_noise));
        initial_.insert(key0, pose0);

        isam2_.update(graph_, initial_);
        graph_.resize(0);
        initial_.clear();

        auto est = isam2_.calculateEstimate<Pose3>(key0);
        {
            std::lock_guard<std::mutex> lk(gtsam_mutex_);
            gtsam_estimated_point_.x = est.translation().x();
            gtsam_estimated_point_.y = est.translation().y();
            gtsam_estimated_point_.z = est.translation().z();
            gtsam_index_ = 0;
            gtsam_has_estimate_ = true;
        }

        const auto t0 = est.translation();
        ROS_INFO("[GTSAM] Initial optimized target (map): [%.3f, %.3f, %.3f]", t0.x(), t0.y(), t0.z());

    }

    void Radar_vision::addPointMeasurementToGtsam(const geometry_msgs::Point& p, const ros::Time& /*stamp*/) {
        if (!gtsam_inited_) initializeGtsam();

        // 第一次测量 -> reset
        if (!gtsam_has_estimate_) {
            resetGtsam(p);
            return;
        }

        std::lock_guard<std::mutex> lk(gtsam_mutex_);

        const auto prevKey = Symbol('x', gtsam_index_);
        const auto currKey = Symbol('x', gtsam_index_ + 1);

        // 平滑Between: 近似恒定，平移小噪声，姿态松噪声
        auto smooth_noise = Diagonal::Sigmas((gtsam::Vector(6) <<
            orient_sigma_loose_, orient_sigma_loose_, orient_sigma_loose_,
            smooth_sigma_xy_, smooth_sigma_xy_, smooth_sigma_z_).finished());
        graph_.add(gtsam::BetweenFactor<Pose3>(prevKey, currKey, Pose3(gtsam::Rot3::Quaternion(1,0,0,0), Point3(0,0,0)), smooth_noise));

        // 当前帧观测 Prior
        auto meas_noise = Diagonal::Sigmas((gtsam::Vector(6) <<
            orient_sigma_loose_, orient_sigma_loose_, orient_sigma_loose_,
            meas_sigma_xy_, meas_sigma_xy_, meas_sigma_z_).finished());
        const Pose3 measPose(gtsam::Rot3::Quaternion(1,0,0,0), Point3(p.x, p.y, p.z));
        graph_.add(gtsam::PriorFactor<Pose3>(currKey, measPose, meas_noise));

        // 初值：用测量平移作初值
        initial_.insert(currKey, measPose);

        isam2_.update(graph_, initial_);
        graph_.resize(0);
        initial_.clear();

        auto est = isam2_.calculateEstimate<Pose3>(currKey);
        gtsam_estimated_point_.x = est.translation().x();
        gtsam_estimated_point_.y = est.translation().y();
        gtsam_estimated_point_.z = est.translation().z();
        gtsam_index_++;

        const auto t = est.translation();
        ROS_INFO("[GTSAM] Optimized target (map): [%.3f, %.3f, %.3f]", t.x(), t.y(), t.z());
    }

    void Radar_vision::pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
        if (!use_pointcloud_refinement_) return;

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);
        if (cloud->empty()) return;

        geometry_msgs::Point center_map;
        {
            std::lock_guard<std::mutex> lock(target_mutex_);
            center_map = current_target_position_;
            center_map.x += pending_x_offset_;
            center_map.y += pending_y_offset_;
            center_map.z += pending_z_offset_;
        }

        geometry_msgs::Point refined_map =
            refineTargetWithDBSCANAndNanoGICP(cloud, msg->header.frame_id, center_map, msg->header.stamp);

        if (!std::isnan(refined_map.x)) {
            {
            std::lock_guard<std::mutex> lock(target_mutex_);
            current_target_position_ = refined_map;
            pending_x_offset_ = pending_y_offset_ = pending_z_offset_ = 0.0;
            pointcloud_received_ = true;
            }
            if (use_gtsam_optimization_) addPointMeasurementToGtsam(refined_map, msg->header.stamp);
            ROS_INFO_THROTTLE(2.0, "Refined (DBSCAN+NanoGICP): [%.3f, %.3f, %.3f] -> [%.3f, %.3f, %.3f]",
                            center_map.x, center_map.y, center_map.z,
                            refined_map.x, refined_map.y, refined_map.z);
        }
        radar_vision::TrackData track;
        track.header.stamp = msg->header.stamp;
        track.header.frame_id = "odom";
        track.id = targetid;
        track.tracking = true;
        track.armors_num = 1;
        track.position.x = refined_map.x;
        track.position.y = refined_map.y;
        track.position.z = refined_map.z;
        pub_.publish(track);

        // 可选：发布 marker（map 帧）
        if (target_marker_pub_.getNumSubscribers() > 0) {
            visualization_msgs::Marker m;
            m.header.stamp = msg->header.stamp;
            m.header.frame_id = "odom";
            m.ns = "radar_vision";
            m.id = 0;
            m.type = visualization_msgs::Marker::SPHERE;
            m.action = visualization_msgs::Marker::ADD;
            m.pose.position.x = refined_map.x;
            m.pose.position.y = refined_map.y;
            m.pose.position.z = refined_map.z;
            m.pose.orientation.w = 1.0;
            m.scale.x = m.scale.y = m.scale.z = 0.3;
            m.color.r = 1.0; m.color.g = 0.4; m.color.b = 0.1; m.color.a = 0.9;
            m.lifetime = ros::Duration(0.0);
            target_marker_pub_.publish(m);
        }
    }

    geometry_msgs::Point Radar_vision::calculateAverageTargetFromPointCloud(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
        const geometry_msgs::Point& center,
        double radius) {

        geometry_msgs::Point result;
        result.x = result.y = result.z = std::numeric_limits<double>::quiet_NaN();

        if (!cloud || cloud->empty() || radius <= 0.0) {
            ROS_WARN_THROTTLE(5.0, "Invalid cloud or radius in refinement (empty=%s, radius=%.3f)",
                            (!cloud || cloud->empty()) ? "true" : "false", radius);
            return result;
        }

        // 使用 nanoflann 构建 3D KD-tree（对 PCL 点云零拷贝适配）
        using Adaptor = PCLPointCloudAdaptor<double>;
        using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<double, Adaptor>, Adaptor, 3>;

        Adaptor adaptor(*cloud);
        KDTree index(3 /*dim*/, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* leaf max */));
        index.buildIndex();

        // 查询中心与半径（注意 nanoflann 的 radiusSearch 传入的是“平方半径”）
        const double query_pt[3] = { center.x, center.y, center.z };
        const double search_radius_sq = radius * radius;

        std::vector<std::pair<size_t, double>> matches;
        matches.reserve(64);
        nanoflann::SearchParams params;
        params.sorted = false; // 仅需索引，不必排序

        const size_t nMatches = index.radiusSearch(query_pt, search_radius_sq, matches, params);
        const size_t min_pts = static_cast<size_t>(std::max(0.0, min_points_threshold_));

        if (nMatches < min_pts) {
            ROS_WARN_THROTTLE(5.0,
                "Not enough points (%zu) within radius %.2f around target [%.2f, %.2f, %.2f] (min=%zu)",
                nMatches, radius, center.x, center.y, center.z, min_pts);
            return result;
        }

        double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
        for (const auto& m : matches) {
            const auto& p = cloud->points[m.first];
            sum_x += p.x;
            sum_y += p.y;
            sum_z += p.z;
        }
        const double invN = 1.0 / static_cast<double>(nMatches);
        result.x = sum_x * invN;
        result.y = sum_y * invN;
        result.z = sum_z * invN;

        ROS_DEBUG("Found %zu points within radius %.2f, average position: [%.3f, %.3f, %.3f]",
                nMatches, radius, result.x, result.y, result.z);
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

    geometry_msgs::Point Radar_vision::refineTargetWithDBSCANAndNanoGICP(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
        const std::string& /*cloud_frame*/,
        const geometry_msgs::Point& center_map,
        const ros::Time& stamp) {

        geometry_msgs::Point nanp;
        nanp.x = nanp.y = nanp.z = std::numeric_limits<double>::quiet_NaN();
        if (!cloud || cloud->empty()) return nanp;

        // 假定 cloud 与 center_map 在同一坐标系（map），不做 TF 变换
        pcl::PointCloud<pcl::PointXYZ>::Ptr roi(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::CropBox<pcl::PointXYZ> crop;
        crop.setInputCloud(cloud);
        const float r = static_cast<float>(roi_radius_);
        Eigen::Vector4f minb(center_map.x - r, center_map.y - r, center_map.z - r, 1.0f);
        Eigen::Vector4f maxb(center_map.x + r, center_map.y + r, center_map.z + r, 1.0f);
        crop.setMin(minb);
        crop.setMax(maxb);
        crop.filter(*roi);
        if (roi->empty()) return nanp;

        pcl::PointCloud<pcl::PointXYZ>::Ptr clean(new pcl::PointCloud<pcl::PointXYZ>);
        if (use_sor_) {
            pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
            sor.setInputCloud(roi);
            sor.setMeanK(sor_mean_k_);
            sor.setStddevMulThresh(sor_stddev_mul_);
            sor.filter(*clean);
        } else {
            clean = roi;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr roids(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::VoxelGrid<pcl::PointXYZ> vox;
        vox.setInputCloud(clean);
        vox.setLeafSize(voxel_leaf_, voxel_leaf_, voxel_leaf_);
        vox.filter(*roids);
        if (roids->empty()) return nanp;

        ROS_INFO("[DBSCAN] input: roi=%zu, after_sor=%zu, after_voxel=%zu, eps=%.3f, min_pts=%d, size_range=[%d,%d]",
         roi->size(), clean->size(), roids->size(),
         dbscan_eps_, dbscan_min_pts_, cluster_min_size_, cluster_max_size_);

        // DBSCAN
        std::vector<std::vector<int>> clusters_idx;
        if (!simple_dbscan::dbscan<pcl::PointXYZ>(roids, clusters_idx, dbscan_eps_, dbscan_min_pts_)) {
            ROS_WARN_THROTTLE(5.0, "DBSCAN found 0 cluster in ROI");
            return nanp;
        }

        const int max_print = 3;
        ROS_INFO("[DBSCAN] clusters=%zu", clusters_idx.size());
        for (size_t ci = 0; ci < clusters_idx.size() && ci < (size_t)max_print; ++ci) {
            const auto& idxs = clusters_idx[ci];
            Eigen::Vector4f c; pcl::compute3DCentroid(*roids, idxs, c);
            ROS_INFO("[DBSCAN]  - c%zu: size=%zu, centroid=[%.3f, %.3f, %.3f]",
                    ci, idxs.size(), c[0], c[1], c[2]);
        }

        // 选最靠近 center_map 的簇
        pcl::PointCloud<pcl::PointXYZ>::Ptr best_cluster(new pcl::PointCloud<pcl::PointXYZ>);
        Eigen::Vector4f best_centroid(0,0,0,0);
        double best_d2 = std::numeric_limits<double>::infinity();
        const Eigen::Vector3f c0(center_map.x, center_map.y, center_map.z);

        for (const auto& idxs : clusters_idx) {
            if ((int)idxs.size() < cluster_min_size_ || (int)idxs.size() > cluster_max_size_) continue;
            Eigen::Vector4f c; pcl::compute3DCentroid(*roids, idxs, c);
            double d2 = (c.head<3>() - c0).squaredNorm();
            if (d2 < best_d2) {
                best_d2 = d2; best_centroid = c;
                best_cluster->clear(); best_cluster->reserve(idxs.size());
                for (int i : idxs) best_cluster->push_back((*roids)[i]);
            }
        }
        if (best_cluster->empty()) return nanp;

        // 当前簇质心（map 下）
        Eigen::Vector3f refined_c = best_centroid.head<3>();

        // 可选：基于上一帧的时序配准（仍在同一坐标系，无需 TF）
        pcl::PointCloud<pcl::PointXYZ> aligned;
        bool use_aligned_for_vis = false;
        if (use_temporal_registration_) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr prev;
            ros::Time prev_stamp;
            std::string prev_frame; // 不再使用

            {
                std::lock_guard<std::mutex> lk(last_cluster_mutex_);
                prev = last_cluster_;
                prev_stamp = last_cluster_stamp_;
                prev_frame = last_cluster_frame_;
            }

            if (prev && !prev->empty() &&
                std::fabs((stamp - prev_stamp).toSec()) <= temporal_max_dt_) {

                Eigen::Vector4f lastC4; pcl::compute3DCentroid(*prev, lastC4);
                const Eigen::Vector3f lastC = lastC4.head<3>();

                nano_gicp_.setInputSource(best_cluster);
                nano_gicp_.calculateSourceCovariances();
                nano_gicp_.setInputTarget(prev);
                nano_gicp_.calculateTargetCovariances();

                Eigen::Matrix4f guess = Eigen::Matrix4f::Identity();
                guess.block<3,1>(0,3) = lastC - refined_c;

                nano_gicp_.align(aligned, guess);
                const double score = nano_gicp_.getFitnessScore();

                if (nano_gicp_.hasConverged() && score < ng_icp_score_thr_) {
                    const Eigen::Matrix4f T = nano_gicp_.getFinalTransformation();
                    refined_c = T.block<3,3>(0,0) * refined_c + T.block<3,1>(0,3);
                    use_aligned_for_vis = true;
                    ROS_DEBUG("Temporal NanoGICP ok, score=%.6f", score);
                } else {
                    ROS_WARN_THROTTLE(5.0, "Temporal NanoGICP not converged or bad score=%.6f, use centroid", score);
                }
            }
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_vis = best_cluster;
        if (use_aligned_for_vis) {
            cluster_vis.reset(new pcl::PointCloud<pcl::PointXYZ>(aligned));
        }

        if (best_cluster_pub_.getNumSubscribers() > 0) {
            sensor_msgs::PointCloud2 cluster_msg;
            pcl::toROSMsg(*cluster_vis, cluster_msg);
            cluster_msg.header.frame_id = "map";  // 固定在 map
            cluster_msg.header.stamp = stamp;
            best_cluster_pub_.publish(cluster_msg);
        }

        // 直接返回 map 坐标（不再做 TF 变换）
        geometry_msgs::Point out;
        out.x = refined_c.x(); out.y = refined_c.y(); out.z = refined_c.z();
        return out;
    }


} // namespace radar_vision