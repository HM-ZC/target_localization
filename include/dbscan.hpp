#pragma once

#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <vector>

namespace simple_dbscan {

template <typename PointT>
bool dbscan(const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
            std::vector<std::vector<int>>& clusters,
            double eps,
            int min_pts) {
  clusters.clear();
  if (!cloud || cloud->empty() || eps <= 0.0 || min_pts <= 1) return false;

  pcl::KdTreeFLANN<PointT> tree;
  tree.setInputCloud(cloud);

  const int n = static_cast<int>(cloud->size());
  std::vector<int> labels(n, 0);  // 0=unvisited, -1=noise, >0=cluster id
  int cluster_id = 0;

  std::vector<int> nn_indices;
  std::vector<float> nn_dists;

  auto regionQuery = [&](int idx, std::vector<int>& out_indices) {
    out_indices.clear();
    if (tree.radiusSearch(cloud->points[idx], eps, out_indices, nn_dists) < 1) {
      out_indices.clear();
    }
  };

  for (int i = 0; i < n; ++i) {
    if (labels[i] != 0) continue;

    regionQuery(i, nn_indices);
    if (static_cast<int>(nn_indices.size()) < min_pts) {
      labels[i] = -1;  // 噪声
      continue;
    }

    // 创建新簇
    ++cluster_id;
    labels[i] = cluster_id;

    // 扩展簇
    std::vector<int> seeds = nn_indices;  // 种子队列
    for (size_t k = 0; k < seeds.size(); ++k) {
      int j = seeds[k];

      if (labels[j] == -1) {
        // 噪声点作为边界点并入当前簇
        labels[j] = cluster_id;
      }
      if (labels[j] != 0) continue;  // 已分配过簇/访问过

      labels[j] = cluster_id;

      regionQuery(j, nn_indices);
      if (static_cast<int>(nn_indices.size()) >= min_pts) {
        // 将新的邻域点加入 seeds（可能有重复，labels 会防止重复扩展）
        seeds.insert(seeds.end(), nn_indices.begin(), nn_indices.end());
      }
    }
  }

  if (cluster_id == 0) return false;

  clusters.resize(cluster_id);
  for (int idx = 0; idx < n; ++idx) {
    int lbl = labels[idx];
    if (lbl > 0) clusters[lbl - 1].push_back(idx);
  }
  return !clusters.empty();
}

}  // namespace simple_dbscan
