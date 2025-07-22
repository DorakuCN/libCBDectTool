#include "libcbdetect/boards_from_corners.h"
#include "libcbdetect/config.h"
#include "libcbdetect/find_corners.h"
#include "libcbdetect/plot_boards.h"
#include "libcbdetect/plot_corners.h"
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>
#include <cmath>
#define _USE_MATH_DEFINES

// Ceres优化库支持（如果可用）
#ifdef USE_CERES
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#endif

using namespace std::chrono;

// 重投影误差结构体（用于Bundle Adjustment优化）
#ifdef USE_CERES
struct ReprojError {
    ReprojError(cv::Point2f obs, int r, int c)
        : u(obs.x), v(obs.y), row(r), col(c) {}

    template<typename T>
    bool operator()(const T* const O,
                    const T* const R,
                    const T* const C,
                    const T* const camera,
                    const T* const pose,
                    T* residuals) const
    {
        // 1. 计算棋盘点 3D
        T P[3];
        for (int k=0;k<3;++k)
            P[k] = O[k] + T(row)*R[k] + T(col)*C[k];

        // 2. 变换到相机坐标
        T Pc[3];
        ceres::AngleAxisRotatePoint(pose, P, Pc);
        Pc[0] += pose[3]; Pc[1] += pose[4]; Pc[2] += pose[5];

        // 3. 投影
        T xp = Pc[0] / Pc[2];
        T yp = Pc[1] / Pc[2];
        const T& fx = camera[0]; const T& fy = camera[1];
        const T& cx = camera[2]; const T& cy = camera[3];
        T u_pred = fx * xp + cx;
        T v_pred = fy * yp + cy;

        residuals[0] = u_pred - T(u);
        residuals[1] = v_pred - T(v);
        return true;
    }

    static ceres::CostFunction* Create(cv::Point2f obs, int r, int c)
    {
        return (new ceres::AutoDiffCostFunction<ReprojError, 2, 3, 3, 3, 4, 6>(
            new ReprojError(obs, r, c)));
    }
    
    double u, v;
    int row, col;
};

// Bundle Adjustment优化函数
bool optimizeChessboardPose(const std::vector<std::vector<cv::Point2f>>& grid,
                           const cv::Mat& camera_matrix,
                           const cv::Mat& dist_coeffs,
                           cv::Mat& rvec,
                           cv::Mat& tvec,
                           std::vector<cv::Point3f>& object_points,
                           double& final_cost) {
    if (grid.empty() || grid[0].empty()) {
        printf("错误：网格为空，无法进行BA优化\n");
        return false;
    }
    
    int rows = grid.size();
    int cols = grid[0].size();
    
    // 1. 准备观测数据
    std::vector<cv::Point2f> observations;
    std::vector<int> obs_rows, obs_cols;
    
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (grid[r][c].x >= 0 && grid[r][c].y >= 0) {
                observations.push_back(grid[r][c]);
                obs_rows.push_back(r);
                obs_cols.push_back(c);
            }
        }
    }
    
    if (observations.size() < 10) {
        printf("错误：有效观测点太少 (%zu < 10)，无法进行BA优化\n", observations.size());
        return false;
    }
    
    printf("开始BA优化，有效观测点: %zu\n", observations.size());
    
    // 2. 初始化优化参数
    double O[3] = {0, 0, 0};  // 棋盘原点
    double R[3] = {1, 0, 0};  // 行方向
    double C[3] = {0, 1, 0};  // 列方向
    
    // 从相机矩阵提取内参
    double camera[4] = {
        camera_matrix.at<double>(0,0),  // fx
        camera_matrix.at<double>(1,1),  // fy
        camera_matrix.at<double>(0,2),  // cx
        camera_matrix.at<double>(1,2)   // cy
    };
    
    // 从rvec和tvec初始化pose
    double pose[6];
    pose[0] = rvec.at<double>(0);
    pose[1] = rvec.at<double>(1);
    pose[2] = rvec.at<double>(2);
    pose[3] = tvec.at<double>(0);
    pose[4] = tvec.at<double>(1);
    pose[5] = tvec.at<double>(2);
    
    // 3. 构建优化问题
    ceres::Problem problem;
    
    for (size_t i = 0; i < observations.size(); ++i) {
        ceres::CostFunction* cost_function = 
            ReprojError::Create(observations[i], obs_rows[i], obs_cols[i]);
        
        problem.AddResidualBlock(cost_function, 
                                new ceres::HuberLoss(1.0),  // 鲁棒核函数
                                O, R, C, camera, pose);
    }
    
    // 4. 设置参数块为常量（可选）
    // problem.SetParameterBlockConstant(camera);  // 如果不想优化内参
    
    // 5. 求解优化问题
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 100;
    options.function_tolerance = 1e-6;
    options.gradient_tolerance = 1e-10;
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    // 6. 输出优化结果
    printf("BA优化完成:\n");
    printf("  迭代次数: %d\n", summary.num_iterations);
    printf("  最终成本: %.6f\n", summary.final_cost);
    printf("  收敛状态: %s\n", summary.termination_type == ceres::CONVERGENCE ? "收敛" : "未收敛");
    
    final_cost = summary.final_cost;
    
    // 7. 更新结果
    if (summary.termination_type == ceres::CONVERGENCE) {
        rvec.at<double>(0) = pose[0];
        rvec.at<double>(1) = pose[1];
        rvec.at<double>(2) = pose[2];
        tvec.at<double>(0) = pose[3];
        tvec.at<double>(1) = pose[4];
        tvec.at<double>(2) = pose[5];
        
        // 更新3D点
        object_points.clear();
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                cv::Point3f pt;
                pt.x = O[0] + r * R[0] + c * C[0];
                pt.y = O[1] + r * R[1] + c * C[1];
                pt.z = O[2] + r * R[2] + c * C[2];
                object_points.push_back(pt);
            }
        }
        
        printf("BA优化成功，更新了位姿和3D点\n");
        return true;
    } else {
        printf("BA优化失败，保持原始结果\n");
        return false;
    }
}

// 简化的重投影误差计算（不使用Ceres时的备选方案）
double computeReprojError(const std::vector<cv::Point2f>& image_points,
                         const std::vector<cv::Point3f>& object_points,
                         const cv::Mat& camera_matrix,
                         const cv::Mat& dist_coeffs,
                         const cv::Mat& rvec,
                         const cv::Mat& tvec) {
    if (image_points.size() != object_points.size()) {
        return std::numeric_limits<double>::max();
    }
    
    std::vector<cv::Point2f> projected_points;
    cv::projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs, projected_points);
    
    double total_error = 0.0;
    for (size_t i = 0; i < image_points.size(); ++i) {
        double error = cv::norm(image_points[i] - projected_points[i]);
        total_error += error * error;  // 平方误差
    }
    
    return std::sqrt(total_error / image_points.size());  // RMS误差
}
#endif

// 重投影误差计算函数（始终可用）
double computeReprojError(const std::vector<cv::Point2f>& image_points,
                         const std::vector<cv::Point3f>& object_points,
                         const cv::Mat& camera_matrix,
                         const cv::Mat& dist_coeffs,
                         const cv::Mat& rvec,
                         const cv::Mat& tvec) {
    if (image_points.size() != object_points.size()) {
        return std::numeric_limits<double>::max();
    }
    
    std::vector<cv::Point2f> projected_points;
    cv::projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs, projected_points);
    
    double total_error = 0.0;
    for (size_t i = 0; i < image_points.size(); ++i) {
        double error = cv::norm(image_points[i] - projected_points[i]);
        total_error += error * error;  // 平方误差
    }
    
    return std::sqrt(total_error / image_points.size());  // RMS误差
}

// 改进的角点评分函数（FAST + Laplacian of Gaussian）
float cornerScore(const cv::Mat& gray, cv::Point2f p) {
    // 检查边界条件
    int x = (int)p.x, y = (int)p.y;
    if (x < 4 || y < 4 || x >= gray.cols - 4 || y >= gray.rows - 4) {
        return 0.0f;  // 边界点返回0分
    }
    
    // FAST + Laplacian of Gaussian 的加权
    const int kSize = 7;
    cv::KeyPoint kp(p, kSize);
    std::vector<cv::KeyPoint> v{kp};
    cv::FAST(gray, v, 20, true);               // FAST 响应
    float fastResp = v.empty() ? 0.0f : v[0].response;

    // 计算Laplacian of Gaussian响应
    cv::Mat roi = gray(cv::Rect(x-4, y-4, 9, 9));
    cv::Mat gaussian_kernel = cv::getGaussianKernel(9, 1.0, CV_32F);
    cv::Mat log_kernel = gaussian_kernel * gaussian_kernel.t();
    
    // 应用LoG核
    cv::Mat filtered;
    cv::filter2D(roi, filtered, CV_32F, log_kernel);
    float logResp = std::fabs(filtered.at<float>(4, 4));  // 中心点响应

    return 0.6f * fastResp + 0.4f * logResp;       // 经验加权
}

// 批量角点评分函数
std::vector<float> computeCornerScores(const cv::Mat& gray, const std::vector<cv::Point2f>& corners) {
    std::vector<float> scores;
    scores.reserve(corners.size());
    
    for (const auto& corner : corners) {
        float score = cornerScore(gray, corner);
        scores.push_back(score);
    }
    
    return scores;
}

// 极简 1-D DBSCAN (用于行/列聚类)
void dbscan1D(const std::vector<float>& proj,
              float eps, int min_samples,
              std::vector<int>& labels)
{
    const int n = static_cast<int>(proj.size());
    labels.assign(n, -1);
    int current_cluster = 0;

    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a,int b){ return proj[a] < proj[b]; });

    for (int id = 0; id < n; ++id)
    {
        int i = idx[id];
        if (labels[i] != -1) continue;               // 已归簇

        // 向左右扩张
        int l = id, r = id;
        while (l-1 >= 0 && fabs(proj[idx[l-1]] - proj[i]) <= eps) --l;
        while (r+1 <  n && fabs(proj[idx[r+1]] - proj[i]) <= eps) ++r;

        int cluster_size = r - l + 1;
        if (cluster_size < min_samples) continue;   // 噪声点

        // 标记整个区间
        for (int k = l; k <= r; ++k)
            labels[idx[k]] = current_cluster;
        ++current_cluster;
    }

    // 归一化标签（0,1,2,... 连续）
    std::map<int,int> remap;
    for (auto& v : labels)
        if (v >= 0 && remap.find(v) == remap.end())
            remap[v] = static_cast<int>(remap.size());
    for (auto& v : labels) if (v >= 0) v = remap[v];
}

// 计算中位数
float computeMedian(const std::vector<float>& values) {
    if (values.empty()) return 0.0f;
    
    std::vector<float> sorted_values = values;
    std::sort(sorted_values.begin(), sorted_values.end());
    
    int n = sorted_values.size();
    if (n % 2 == 0) {
        return (sorted_values[n/2 - 1] + sorted_values[n/2]) / 2.0f;
    } else {
        return sorted_values[n/2];
    }
}

// 相机内参和畸变系数（示例值，实际应用中应该从标定获得）
cv::Mat K = (cv::Mat_<double>(3,3) << 
    1000.0, 0.0, 320.0,
    0.0, 1000.0, 240.0,
    0.0, 0.0, 1.0);
cv::Mat distCoeffs = (cv::Mat_<double>(5,1) << 0.0, 0.0, 0.0, 0.0, 0.0);

// 基于单应性矩阵的投影函数
cv::Point2f projectRC(const cv::Mat& H, float r, float c) {
    cv::Mat p = H * (cv::Mat_<double>(3, 1) << r, c, 1);
    cv::Point2f img(p.at<double>(0)/p.at<double>(2), p.at<double>(1)/p.at<double>(2));
    
    // 反向畸变到原图坐标
    std::vector<cv::Point2f> tmp{img};
    cv::undistortPoints(tmp, tmp, K, distCoeffs, cv::noArray(), K);
    return tmp[0];
}

// 单应 + 局部搜索预测缺失角
static bool localCornerSearch(const cv::Mat& gray,
                              const cv::Point2f& pred,
                              float search_rad,
                              cv::Point2f& best_pt)
{
    cv::Rect win(pred.x - search_rad,
                 pred.y - search_rad,
                 2*search_rad+1,
                 2*search_rad+1);
    win &= cv::Rect(0,0,gray.cols,gray.rows);
    if (win.width < 5 || win.height < 5) return false;

    std::vector<cv::Point2f> cand;
    cv::goodFeaturesToTrack(gray(win), cand, 20, 0.01, 3);

    float best_dist = 1e9;
    for (auto& p : cand)
    {
        cv::Point2f real = p + cv::Point2f(win.x,win.y);
        float d = cv::norm(real - pred);
        if (d < best_dist)
        {
            best_dist = d;
            best_pt   = real;
        }
    }
    return best_dist < 5.0f;   // 5 px 阈值
}

void predictMissingCorners(const cv::Mat& gray,
                           const std::vector<cv::Point2f>& meas_pts,
                           const std::vector<int>& meas_r,
                           const std::vector<int>& meas_c,
                           int rows, int cols,
                           std::vector<cv::Point2f>& out_pts,
                           std::vector<char>& out_mask,
                           std::vector<int>& out_r,
                           std::vector<int>& out_c)
{
    /*--- 1. 按 (r,c)→(x,y) 求单应 H ---*/
    cv::Mat A(meas_pts.size(), 2, CV_64F);
    cv::Mat B(meas_pts.size(), 2, CV_64F);
    for (size_t i=0;i<meas_pts.size();++i)
    {
        A.at<double>(i,0) = meas_r[i];
        A.at<double>(i,1) = meas_c[i];
        B.at<double>(i,0) = meas_pts[i].x;
        B.at<double>(i,1) = meas_pts[i].y;
    }
    cv::Mat H = cv::findHomography(A, B, cv::RANSAC, 5.0);
    if (H.empty()) return;       // Fallback: 不补全

    /*--- 2. 生成全 88 个预测 ---*/
    cv::Mat rc(3,1,CV_64F);
    rc.at<double>(2)=1.0;
    const float med_spacing =
        cv::norm(meas_pts[0] - meas_pts[1]);        // 粗略 spacing
    const float win_r = 0.5f * med_spacing;

    std::set<std::pair<int,int>> occupied;
    for (size_t i=0;i<meas_r.size();++i)
        occupied.insert({meas_r[i],meas_c[i]});

    out_pts  = meas_pts;
    out_r    = meas_r;
    out_c    = meas_c;
    out_mask = std::vector<char>(meas_pts.size(),1);   // 1=原测得

    for (int r=0;r<rows;++r)
    for (int c=0;c<cols;++c)
    {
        if (occupied.count({r,c})) continue;          // 已有

        rc.at<double>(0)=r; rc.at<double>(1)=c;
        cv::Mat pred_h = H * rc;
        cv::Point2f pred(pred_h.at<double>(0)/pred_h.at<double>(2),
                         pred_h.at<double>(1)/pred_h.at<double>(2));

        cv::Point2f best;
        if (localCornerSearch(gray, pred, win_r, best))
        {
            out_pts .push_back(best);
            out_r   .push_back(r);
            out_c   .push_back(c);
            out_mask.push_back(2);                    // 2=补全
        }
        else
        {
            // 直接使用预测值但标记为"低置信度"，稍后可再剔除
            out_pts .push_back(pred);
            out_r   .push_back(r);
            out_c   .push_back(c);
            out_mask.push_back(0);                    // 0=仅预测
        }
    }
}

// 替换原 completeMissingCorners
void completeMissingCorners(const cv::Mat& gray,
                             std::vector<cv::Point2f>& pts,
                             std::vector<int>& row_lbl,
                             std::vector<int>& col_lbl,
                             std::vector<char>& inlier_mask,
                             int rows, int cols)
{
    /*--- 仅保留当前内点作为测量 ---*/
    std::vector<cv::Point2f> meas;
    std::vector<int>         meas_r, meas_c;
    for (size_t i=0;i<pts.size();++i)
        if (inlier_mask[i] && row_lbl[i]>=0 && col_lbl[i]>=0)
        {
            meas.push_back(pts[i]);
            meas_r.push_back(row_lbl[i]);
            meas_c.push_back(col_lbl[i]);
        }

    /*--- 预测缺失 ---*/
    std::vector<cv::Point2f> all_pts;
    std::vector<char>        all_mask;
    std::vector<int>         all_r, all_c;

    predictMissingCorners(gray, meas, meas_r, meas_c,
                          rows, cols,
                          all_pts, all_mask, all_r, all_c);

    /*--- 写回 ---*/
    pts         = all_pts;
    row_lbl     = all_r;
    col_lbl     = all_c;
    inlier_mask = all_mask;      // 0=预测-低置信,1=原,2=补全
    printf("角点补全完成: 总 %zu 个 (新补全 %zu 个, 低置信度 %zu 个)\n",
           pts.size(),
           (size_t)std::count(all_mask.begin(),all_mask.end(),2),
           (size_t)std::count(all_mask.begin(),all_mask.end(),0));
}

// 图像预处理函数：可选CLAHE、锐化、去噪，支持参数
cv::Mat preprocess_image(const cv::Mat& input, bool use_clahe = true, bool use_sharpen = false, bool use_denoise = false, double clahe_clip = 2.0, int clahe_tile = 8) {
    cv::Mat img = input.clone();
    if (use_clahe) {
        // 对Y通道做CLAHE
        cv::Mat img_yuv;
        cv::cvtColor(img, img_yuv, cv::COLOR_BGR2YCrCb);
        std::vector<cv::Mat> channels;
        cv::split(img_yuv, channels);
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clahe_clip, cv::Size(clahe_tile,clahe_tile));
        clahe->apply(channels[0], channels[0]);
        cv::merge(channels, img_yuv);
        cv::cvtColor(img_yuv, img, cv::COLOR_YCrCb2BGR);
    }
    if (use_sharpen) {
        cv::Mat kernel = (cv::Mat_<float>(3,3) << 0,-1,0,-1,5,-1,0,-1,0);
        cv::filter2D(img, img, img.depth(), kernel);
    }
    if (use_denoise) {
        cv::fastNlMeansDenoisingColored(img, img, 3, 3, 7, 21);
    }
    return img;
}

// PCA主方向估算函数（用于RANSAC后的精确估算）
void estimateMainAxes(const std::vector<cv::Point2f>& pts, cv::Point2f& dir1, cv::Point2f& dir2) {
    if (pts.size() < 2) return;
    
    cv::Mat data(pts.size(), 2, CV_32F);
    for (size_t i = 0; i < pts.size(); ++i) {
        data.at<float>(i, 0) = pts[i].x;
        data.at<float>(i, 1) = pts[i].y;
    }
    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);
    cv::Vec2f axis1 = pca.eigenvectors.row(0);
    cv::Vec2f axis2 = pca.eigenvectors.row(1);
    dir1 = cv::Point2f(axis1);
    dir2 = cv::Point2f(axis2);
}

// ---------- 1) RANSAC 主方向 ----------
void estimateMainAxesRansac(const std::vector<cv::Point2f>& pts,
                            cv::Point2f& dir1, cv::Point2f& dir2,
                            int max_iter = 500, float tol = 3.0f)
{
    if (pts.size() < 2) return;
    
    cv::RNG rng(cv::getTickCount());
    int best_inlier = 0;
    cv::Point2f best_v;

    for (int it = 0; it < max_iter; ++it) {
        int i = rng.uniform(0, (int)pts.size());
        int j = rng.uniform(0, (int)pts.size()-1);
        if (j >= i) ++j;
        cv::Point2f v = pts[j] - pts[i];
        if (cv::norm(v) < 1e-3) continue;
        v *= 1.0f / cv::norm(v);

        // 统计同向、反向点（绝对值投影）
        int inliers = 0;
        for (auto& p : pts) {
            cv::Point2f d = p - pts[i];
            float proj = fabs(d.dot(v));
            float ortho = fabs(d.x * v.y - d.y * v.x);  // 垂距
            if (ortho < tol) ++inliers;
        }
        if (inliers > best_inlier) {
            best_inlier = inliers;
            best_v = v;
        }
    }
    
    // 以最大内点集再跑一次 PCA
    std::vector<cv::Point2f> inlier_pts;
    for (auto& p : pts) {
        if (fabs((p - pts[0]).x * best_v.y - (p - pts[0]).y * best_v.x) < tol)
            inlier_pts.push_back(p);
    }
    estimateMainAxes(inlier_pts, dir1, dir2);
}

// ---------- 2) 改进的局部一致性剔除 ----------
void rejectOutliers(const std::vector<cv::Point2f>& pts,
                    const std::vector<int>& row_lbl,
                    const std::vector<int>& col_lbl,
                    int rows, int cols,
                    std::vector<char>& inlier_mask,
                    float k_sigma = 3.0f)
{
    printf("开始局部一致性剔除，角点数量: %zu, 网格尺寸: %dx%d\n", pts.size(), rows, cols);
    
    // 估计行/列平均向量
    std::vector<float> dx, dy;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols-1; ++c) {
            int idx1 = -1, idx2 = -1;
            for (size_t i = 0; i < pts.size(); ++i)
                if (row_lbl[i]==r && col_lbl[i]==c) idx1 = (int)i;
            for (size_t i = 0; i < pts.size(); ++i)
                if (row_lbl[i]==r && col_lbl[i]==c+1) idx2 = (int)i;
            if (idx1>=0 && idx2>=0) {
                dx.push_back(pts[idx2].x - pts[idx1].x);
                dy.push_back(pts[idx2].y - pts[idx1].y);
            }
        }
    }
    
    // 计算列间距
    for (int c = 0; c < cols; ++c) {
        for (int r = 0; r < rows-1; ++r) {
            int idx1 = -1, idx2 = -1;
            for (size_t i = 0; i < pts.size(); ++i)
                if (row_lbl[i]==r && col_lbl[i]==c) idx1 = (int)i;
            for (size_t i = 0; i < pts.size(); ++i)
                if (row_lbl[i]==r+1 && col_lbl[i]==c) idx2 = (int)i;
            if (idx1>=0 && idx2>=0) {
                dx.push_back(pts[idx2].x - pts[idx1].x);
                dy.push_back(pts[idx2].y - pts[idx1].y);
            }
        }
    }
    
    if (dx.empty() || dy.empty()) {
        printf("警告：无法计算间距向量，跳过局部一致性剔除\n");
        inlier_mask.assign(pts.size(), 1);
        return;
    }
    
    printf("计算了 %zu 个间距向量\n", dx.size());
    
    auto median = [](std::vector<float>& v){
        if (v.empty()) return 0.0f;
        std::nth_element(v.begin(), v.begin()+v.size()/2, v.end());
        return v[v.size()/2];
    };
    
    float mdx = median(dx), mdy = median(dy);
    printf("中位数间距: dx=%.2f, dy=%.2f\n", mdx, mdy);

    // MAD (Median Absolute Deviation)
    std::vector<float> dx_mad = dx, dy_mad = dy;
    for (auto& v : dx_mad) v = fabs(v - mdx);
    for (auto& v : dy_mad) v = fabs(v - mdy);
    float sdx = 1.4826f * median(dx_mad);
    float sdy = 1.4826f * median(dy_mad);
    
    printf("MAD标准差: sdx=%.2f, sdy=%.2f\n", sdx, sdy);

    inlier_mask.assign(pts.size(), 1);
    int rejected_count = 0;
    
    for (size_t i = 0; i < pts.size(); ++i) {
        // 与其右邻、下邻误差
        int r = row_lbl[i], c = col_lbl[i];
        if (r < 0) { 
            inlier_mask[i] = 0; 
            rejected_count++;
            continue; 
        }
        
        auto neighbor = [&](int rr,int cc)->int{
            for (size_t k=0;k<pts.size();++k)
                if (row_lbl[k]==rr && col_lbl[k]==cc) return (int)k;
            return -1;
        };
        
        int right = neighbor(r, c+1);
        int down  = neighbor(r+1, c);
        
        auto check = [&](int j){
            if (j<0) return;
            cv::Point2f d = pts[j] - pts[i];
            float dx_error = fabs(d.x - mdx);
            float dy_error = fabs(d.y - mdy);
            
            if (dx_error > k_sigma*sdx || dy_error > k_sigma*sdy) {
                if (inlier_mask[i] == 1) {
                    inlier_mask[i] = 0;
                    rejected_count++;
                }
                if (inlier_mask[j] == 1) {
                    inlier_mask[j] = 0;
                    rejected_count++;
                }
            }
        };
        
        check(right); 
        check(down);
    }
    
    int remaining_count = std::count(inlier_mask.begin(), inlier_mask.end(), 1);
    printf("局部一致性剔除完成: 剔除 %d 个角点，保留 %d 个角点\n", 
           rejected_count, remaining_count);
}

// ---------- 3) 改进的局部一致性剔除（使用方向投影） ----------
void rejectOutliers2(const std::vector<cv::Point2f>& pts,
                     const std::vector<int>& row_lbl,
                     const std::vector<int>& col_lbl,
                     const cv::Point2f& dir_row,
                     const cv::Point2f& dir_col,
                     int rows, int cols,
                     std::vector<char>& inlier_mask,
                     const std::vector<char>& point_types,  // 新增: 0预测/1实测/2补全
                     float k_sigma = 3.0f)
{
    printf("开始改进的局部一致性剔除（方向投影），角点数量: %zu, 网格尺寸: %dx%d\n", 
           pts.size(), rows, cols);
    
    std::vector<float> row_sp, col_sp;
    struct Edge { int i,j; bool is_row; };          // 记下谁跟谁
    std::vector<Edge> edges;

    // 收集行/列邻边
    auto idxOf = [&](int r,int c)->int{
        for (size_t i=0;i<pts.size();++i)
            if (row_lbl[i]==r && col_lbl[i]==c) return (int)i;
        return -1;
    };
    
    for (int r=0;r<rows;++r) {
        for (int c=0;c<cols;++c) {
            int i = idxOf(r,c);
            if (i<0) continue;
            int j = idxOf(r, c+1);      // 行邻
            if (j>=0) {
                row_sp.push_back( (pts[j]-pts[i]).dot(dir_row) );
                edges.push_back({i,j,true});
            }
            j = idxOf(r+1, c);          // 列邻
            if (j>=0) {
                col_sp.push_back( (pts[j]-pts[i]).dot(dir_col) );
                edges.push_back({i,j,false});
            }
        }
    }

    if (row_sp.empty() && col_sp.empty()) {
        printf("警告：无法找到有效的邻边，跳过局部一致性剔除\n");
        inlier_mask.assign(pts.size(), 1);
        return;
    }

    auto robustStats = [](std::vector<float>& v, float& med, float& mad){
        if (v.empty()) {
            med = 0.0f;
            mad = 1.0f;
            return;
        }
        auto mid = v.begin()+v.size()/2;
        std::nth_element(v.begin(), mid, v.end());
        med = *mid;
        for (auto& x: v) x = std::fabs(x - med);
        std::nth_element(v.begin(), mid, v.end());
        mad = 1.4826f * (*mid);
    };

    float med_row, sigma_row, med_col, sigma_col;
    robustStats(row_sp, med_row, sigma_row);
    robustStats(col_sp, med_col, sigma_col);

    printf("行间距统计: 中位数=%.2f, MAD=%.2f (样本数=%zu)\n", med_row, sigma_row, row_sp.size());
    printf("列间距统计: 中位数=%.2f, MAD=%.2f (样本数=%zu)\n", med_col, sigma_col, col_sp.size());

    inlier_mask.assign(pts.size(), 1);
    int rejected_pairs = 0;
    
    /* 在最终决定剔除时，若点为"补全"(2) 则再降低阈值 */
    auto penalty = [&](int idx){
        return point_types.empty()?1.0f:
              (point_types[idx]==2?0.5f:1.0f);
    };
    
    for (const auto& e : edges) {
        float d = (pts[e.j]-pts[e.i]).dot(e.is_row ? dir_row : dir_col);
        float med  = e.is_row ? med_row  : med_col;
        float sigma= e.is_row ? sigma_row: sigma_col;
        if (sigma < 1e-3) sigma = 1;                 // 防 0
        if (std::fabs(d - med) > k_sigma * sigma * penalty(e.i)) {
            inlier_mask[e.i] = inlier_mask[e.j] = 0;
            rejected_pairs++;
        }
    }
    
    int retained_count = std::count(inlier_mask.begin(), inlier_mask.end(), 1);
    printf("改进的局部一致性剔除完成: 剔除 %d 个角点，保留 %d 个角点 (拒绝 %d 对邻边)\n", 
           (int)pts.size() - retained_count, retained_count, rejected_pairs);
}

// 动态调整检测参数的函数
void adjustDetectionParams(const cbdetect::Corner& corners, 
                          int expected_corners,
                          cbdetect::Params& params) {
    int detected_count = corners.p.size();
    double ratio = static_cast<double>(detected_count) / expected_corners;
    
    printf("\n=== 动态参数调整 ===\n");
    printf("期望角点数: %d, 检测到: %d, 比例: %.2f\n", expected_corners, detected_count, ratio);
    
    if (ratio < 0.9) { // 检测不足，放宽标准
        printf("检测不足，放宽检测标准...\n");
        params.score_thr *= 0.7; // 降低分数阈值
        params.init_loc_thr *= 0.7; // 降低位置阈值
        printf("调整后参数: score_thr=%.4f, init_loc_thr=%.4f\n", params.score_thr, params.init_loc_thr);
    } else if (ratio > 1.1) { // 检测过多，收紧标准
        printf("检测过多，收紧检测标准...\n");
        params.score_thr *= 1.3; // 提高分数阈值
        params.init_loc_thr *= 1.3; // 提高位置阈值
        printf("调整后参数: score_thr=%.4f, init_loc_thr=%.4f\n", params.score_thr, params.init_loc_thr);
    } else {
        printf("检测数量在合理范围内，保持当前参数\n");
    }
}

// 创建相关性模板核函数
void create_correlation_patch(std::vector<cv::Mat>& template_kernel,
                              double angle_1, double angle_2, int radius) {
    int width  = 2 * radius + 1;
    int height = 2 * radius + 1;
    template_kernel[0] = cv::Mat::zeros(height, width, CV_64F);
    template_kernel[1] = cv::Mat::zeros(height, width, CV_64F);
    template_kernel[2] = cv::Mat::zeros(height, width, CV_64F);
    template_kernel[3] = cv::Mat::zeros(height, width, CV_64F);
    int mu = radius;  // Matlab 中中心索引从 0 开始
    int mv = radius;
    cv::Point2d n1(-std::sin(angle_1), std::cos(angle_1));
    cv::Point2d n2(-std::sin(angle_2), std::cos(angle_2));
    double sigma = radius / 2.0;
    auto normpdf = [sigma](double d) {
        return std::exp(-0.5 * (d * d) / (sigma * sigma));
    };
    for (int v = 0; v < height; ++v) {
        for (int u = 0; u < width; ++u) {
            cv::Point2d vec(u - mu, v - mv);
            double s1 = vec.dot(n1);
            double s2 = vec.dot(n2);
            double dist = cv::norm(vec);
            if (s1 <= -0.1 && s2 <= -0.1) {
                template_kernel[0].at<double>(v, u) = normpdf(dist);
            } else if (s1 >= 0.1 && s2 >= 0.1) {
                template_kernel[1].at<double>(v, u) = normpdf(dist);
            } else if (s1 <= -0.1 && s2 >= 0.1) {
                template_kernel[2].at<double>(v, u) = normpdf(dist);
            } else if (s1 >= 0.1 && s2 <= -0.1) {
                template_kernel[3].at<double>(v, u) = normpdf(dist);
            }
        }
    }
    // 归一化
    for (int i = 0; i < 4; ++i) {
        double sum = cv::sum(template_kernel[i])[0];
        if (sum > 1e-8) template_kernel[i] /= sum;
    }
}

// 计算角点相关性评分
double compute_correlation_score(const cv::Mat& img, const cv::Point2f& corner, 
                                const std::vector<cv::Mat>& template_kernels) {
    int radius = (template_kernels[0].rows - 1) / 2;
    cv::Rect roi(corner.x - radius, corner.y - radius, 2*radius+1, 2*radius+1);
    
    // 检查边界
    if (roi.x < 0 || roi.y < 0 || 
        roi.x + roi.width > img.cols || 
        roi.y + roi.height > img.rows) {
        return 0.0;
    }
    
    cv::Mat patch = img(roi);
    cv::Mat patch_float;
    patch.convertTo(patch_float, CV_64F);
    
    double max_score = 0.0;
    for (const auto& kernel : template_kernels) {
        cv::Mat result;
        cv::filter2D(patch_float, result, CV_64F, kernel);
        double score = result.at<double>(radius, radius);
        max_score = std::max(max_score, score);
    }
    
    return max_score;
}

// 可视化相关性模板
void visualize_correlation_templates(const std::vector<cv::Mat>& template_kernels, 
                                    const std::string& filename = "correlation_templates.png") {
    int num_templates = template_kernels.size();
    int template_size = template_kernels[0].rows;
    int display_size = 200;
    
    cv::Mat display(display_size, display_size * num_templates, CV_8UC3);
    display.setTo(0);
    
    for (int i = 0; i < num_templates; ++i) {
        cv::Mat normalized;
        cv::normalize(template_kernels[i], normalized, 0, 255, cv::NORM_MINMAX);
        normalized.convertTo(normalized, CV_8UC1);
        
        cv::Mat resized;
        cv::resize(normalized, resized, cv::Size(display_size, display_size));
        
        cv::Mat color;
        cv::applyColorMap(resized, color, cv::COLORMAP_JET);
        
        cv::Rect roi(i * display_size, 0, display_size, display_size);
        color.copyTo(display(roi));
    }
    
    cv::imwrite(filename, display);
    printf("相关性模板已保存为: %s\n", filename.c_str());
}

// 基于相关性评分的角点过滤（支持动态阈值调整）
void filter_corners_by_correlation(const cv::Mat& img, 
                                  std::vector<cv::Point2f>& corners,
                                  double correlation_threshold = 0.1,
                                  int expected_corners = 88) {
    if (corners.empty()) return;
    
    // 创建模板核
    std::vector<cv::Mat> template_kernels(4);
    double angle_1 = 0.0;  // 水平方向
    double angle_2 = M_PI / 2.0;  // 垂直方向
    int radius = 5;
    create_correlation_patch(template_kernels, angle_1, angle_2, radius);
    
    // 可视化模板
    visualize_correlation_templates(template_kernels);
    
    // 计算每个角点的相关性评分
    std::vector<std::pair<double, int>> scores;
    for (size_t i = 0; i < corners.size(); ++i) {
        double score = compute_correlation_score(img, corners[i], template_kernels);
        scores.emplace_back(score, i);
    }
    
    // 按评分排序
    std::sort(scores.begin(), scores.end(), std::greater<std::pair<double, int>>());
    
    // 动态调整阈值以接近期望角点数
    double adjusted_threshold = correlation_threshold;
    if (scores.size() > expected_corners) {
        // 如果角点过多，提高阈值
        adjusted_threshold = scores[expected_corners].first;
        printf("角点过多，调整阈值从 %.3f 到 %.3f\n", correlation_threshold, adjusted_threshold);
    } else if (scores.size() < expected_corners * 0.8) {
        // 如果角点过少，降低阈值
        adjusted_threshold = std::max(0.01, scores.back().first * 0.5);
        printf("角点过少，调整阈值从 %.3f 到 %.3f\n", correlation_threshold, adjusted_threshold);
    }
    
    // 保留评分高的角点
    std::vector<cv::Point2f> filtered_corners;
    for (const auto& score_pair : scores) {
        if (score_pair.first > adjusted_threshold) {
            filtered_corners.push_back(corners[score_pair.second]);
        }
    }
    
    corners = filtered_corners;
    printf("相关性过滤: 保留 %zu 个角点 (调整后阈值: %.3f)\n", corners.size(), adjusted_threshold);
}

// 改进的主方向估算函数（整合RANSAC方法）
void estimateMainAxesImproved(const std::vector<cv::Point2f>& pts, cv::Point2f& dir1, cv::Point2f& dir2) {
    if (pts.size() < 2) return;
    
    printf("开始主方向估算，角点数量: %zu\n", pts.size());
    
    // 方法1：尝试使用RANSAC主方向估算（更鲁棒）
    cv::Point2f ransac_dir1, ransac_dir2;
    estimateMainAxesRansac(pts, ransac_dir1, ransac_dir2, 500, 5.0f);
    
    // 检查RANSAC结果的角度
    float ransac_angle1 = std::atan2(ransac_dir1.y, ransac_dir1.x) * 180.0f / M_PI;
    float ransac_angle2 = std::atan2(ransac_dir2.y, ransac_dir2.x) * 180.0f / M_PI;
    
    // 将角度标准化到[-90, 90]范围
    while (ransac_angle1 > 90) ransac_angle1 -= 180;
    while (ransac_angle1 < -90) ransac_angle1 += 180;
    while (ransac_angle2 > 90) ransac_angle2 -= 180;
    while (ransac_angle2 < -90) ransac_angle2 += 180;
    
    printf("RANSAC主方向角度: %.1f°, %.1f°\n", ransac_angle1, ransac_angle2);
    
    // 方法2：尝试使用PCA
    cv::Mat data(pts.size(), 2, CV_32F);
    for (size_t i = 0; i < pts.size(); ++i) {
        data.at<float>(i, 0) = pts[i].x;
        data.at<float>(i, 1) = pts[i].y;
    }
    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);
    cv::Vec2f axis1 = pca.eigenvectors.row(0);
    cv::Vec2f axis2 = pca.eigenvectors.row(1);
    
    float pca_angle1 = std::atan2(axis1[1], axis1[0]) * 180.0f / M_PI;
    float pca_angle2 = std::atan2(axis2[1], axis2[0]) * 180.0f / M_PI;
    
    while (pca_angle1 > 90) pca_angle1 -= 180;
    while (pca_angle1 < -90) pca_angle1 += 180;
    while (pca_angle2 > 90) pca_angle2 -= 180;
    while (pca_angle2 < -90) pca_angle2 += 180;
    
    printf("PCA主方向角度: %.1f°, %.1f°\n", pca_angle1, pca_angle2);
    
    // 选择最佳方法
    bool ransac_good = std::abs(ransac_angle1) < 45 && std::abs(ransac_angle2) < 45;
    bool pca_good = std::abs(pca_angle1) < 45 && std::abs(pca_angle2) < 45;
    
    if (ransac_good) {
        dir1 = ransac_dir1;
        dir2 = ransac_dir2;
        printf("使用RANSAC主方向（更鲁棒）\n");
    } else if (pca_good) {
        dir1 = cv::Point2f(axis1[0], axis1[1]);
        dir2 = cv::Point2f(axis2[0], axis2[1]);
        printf("使用PCA主方向\n");
    } else {
        // 否则使用简单的水平/垂直方向
        dir1 = cv::Point2f(1.0f, 0.0f);  // 水平方向
        dir2 = cv::Point2f(0.0f, 1.0f);  // 垂直方向
        printf("使用标准水平/垂直方向\n");
    }
    
    // 归一化方向向量
    dir1 *= 1.0f / cv::norm(dir1);
    dir2 *= 1.0f / cv::norm(dir2);
    
    printf("最终主方向: 行(%.3f, %.3f), 列(%.3f, %.3f)\n", 
           dir1.x, dir1.y, dir2.x, dir2.y);
}

// 行/列标签 —— DBSCAN 聚类 (替换 assignRowColLabels 原实现)
void assignRowColLabels(const std::vector<cv::Point2f>& pts,
                        const cv::Point2f& org,
                        const cv::Point2f& dir_row,
                        const cv::Point2f& dir_col,
                        int num_rows, int num_cols,
                        std::vector<int>& row_labels,
                        std::vector<int>& col_labels)
{
    const int N = static_cast<int>(pts.size());
    std::vector<float> proj_r(N), proj_c(N);
    for (int i=0;i<N;++i)
    {
        cv::Point2f v = pts[i] - org;
        proj_r[i] = v.dot(dir_row);
        proj_c[i] = v.dot(dir_col);
    }

    /*--- 估计平均 spacing 用于 eps ---*/
    auto spacing = [](const std::vector<float>& v){
        std::vector<float> d;
        for (size_t i=1;i<v.size();++i)
            d.push_back(fabs(v[i]-v[i-1]));
        if (d.empty()) return 30.f;
        std::nth_element(d.begin(), d.begin()+d.size()/2, d.end());
        return d[d.size()/2];
    };
    std::vector<float> tmp = proj_r;
    float eps_r = 0.3f * spacing(tmp);           // 30 % 间距
    tmp = proj_c;
    float eps_c = 0.3f * spacing(tmp);

    /*--- DBSCAN ---*/
    dbscan1D(proj_r, eps_r, 2, row_labels);
    dbscan1D(proj_c, eps_c, 2, col_labels);

    /*--- 未归簇的点保持 -1，后续补全时使用 ---*/
    printf("DBSCAN 聚类完成: 行簇 %d, 列簇 %d (目标 %d×%d)\n",
           1+*std::max_element(row_labels.begin(),row_labels.end()),
           1+*std::max_element(col_labels.begin(),col_labels.end()),
           num_rows,num_cols);
}

// 构建有序二维角点网格
void buildSortedGrid(const std::vector<cv::Point2f>& pts,
                     const std::vector<int>& row_labels,
                     const std::vector<int>& col_labels,
                     int num_rows, int num_cols,
                     std::vector<std::vector<cv::Point2f>>& grid) {
    grid.assign(num_rows, std::vector<cv::Point2f>(num_cols, cv::Point2f(-1, -1)));
    for (size_t i = 0; i < pts.size(); ++i) {
        int r = row_labels[i], c = col_labels[i];
        if (r >= 0 && r < num_rows && c >= 0 && c < num_cols) {
            grid[r][c] = pts[i];
        }
    }
}

// 验证网格质量并计算统计信息
void validateGridQuality(const std::vector<std::vector<cv::Point2f>>& grid,
                         int num_rows, int num_cols,
                         double& fill_rate, double& avg_row_spacing, double& avg_col_spacing) {
    int filled_cells = 0;
    std::vector<double> row_spacings, col_spacings;
    
    // 计算行间距
    for (int r = 0; r < num_rows; ++r) {
        std::vector<cv::Point2f> valid_points;
        for (int c = 0; c < num_cols; ++c) {
            if (grid[r][c].x >= 0 && grid[r][c].y >= 0) {
                valid_points.push_back(grid[r][c]);
                filled_cells++;
            }
        }
        if (valid_points.size() >= 2) {
            for (size_t i = 1; i < valid_points.size(); ++i) {
                double spacing = cv::norm(valid_points[i] - valid_points[i-1]);
                row_spacings.push_back(spacing);
            }
        }
    }
    
    // 计算列间距
    for (int c = 0; c < num_cols; ++c) {
        std::vector<cv::Point2f> valid_points;
        for (int r = 0; r < num_rows; ++r) {
            if (grid[r][c].x >= 0 && grid[r][c].y >= 0) {
                valid_points.push_back(grid[r][c]);
            }
        }
        if (valid_points.size() >= 2) {
            for (size_t i = 1; i < valid_points.size(); ++i) {
                double spacing = cv::norm(valid_points[i] - valid_points[i-1]);
                col_spacings.push_back(spacing);
            }
        }
    }
    
    // 计算统计信息
    fill_rate = static_cast<double>(filled_cells) / (num_rows * num_cols);
    
    if (!row_spacings.empty()) {
        avg_row_spacing = std::accumulate(row_spacings.begin(), row_spacings.end(), 0.0) / row_spacings.size();
    } else {
        avg_row_spacing = 0.0;
    }
    
    if (!col_spacings.empty()) {
        avg_col_spacing = std::accumulate(col_spacings.begin(), col_spacings.end(), 0.0) / col_spacings.size();
    } else {
        avg_col_spacing = 0.0;
    }
}

// 自动推断行列标签的函数（改进版本）
void inferRowColLabels(const std::vector<cv::Point2f>& corners, 
                      std::vector<int>& row_lbl, 
                      std::vector<int>& col_lbl,
                      int expected_rows = 8, int expected_cols = 11) {
    if (corners.empty()) return;
    
    // 1. 使用改进的PCA主方向估算
    cv::Point2f dir_row, dir_col;
    estimateMainAxesImproved(corners, dir_row, dir_col);
    
    // 2. 选择原点（左上角或均值）
    cv::Point2f origin = *std::min_element(corners.begin(), corners.end(),
        [](const cv::Point2f& a, const cv::Point2f& b) { 
            return a.x + a.y < b.x + b.y; 
        });
    
    // 3. 使用改进的角点行列分组
    assignRowColLabels(corners, origin, dir_row, dir_col, expected_rows, expected_cols, row_lbl, col_lbl);
    
    printf("改进的行列标签分配完成，有效标签数: %zu\n", 
           std::count_if(row_lbl.begin(), row_lbl.end(), [](int x) { return x >= 0; }));
}

// 测试改进的网格构建功能
void testImprovedGridBuilding() {
    printf("\n=== 测试改进的网格构建功能 ===\n");
    
    // 创建模拟的角点数据（8x11棋盘格）
    std::vector<cv::Point2f> test_corners;
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 11; ++c) {
            // 添加一些随机偏移来模拟真实检测
            float x = c * 50.0f + (rand() % 10 - 5);
            float y = r * 50.0f + (rand() % 10 - 5);
            test_corners.emplace_back(x, y);
        }
    }
    
    printf("创建了 %zu 个测试角点\n", test_corners.size());
    
    // 测试改进的主方向估算
    cv::Point2f dir_row, dir_col;
    estimateMainAxesImproved(test_corners, dir_row, dir_col);
    printf("主方向估算: 行方向(%.3f, %.3f), 列方向(%.3f, %.3f)\n", 
           dir_row.x, dir_row.y, dir_col.x, dir_col.y);
    
    // 测试行列标签分配
    std::vector<int> row_lbl, col_lbl;
    inferRowColLabels(test_corners, row_lbl, col_lbl, 8, 11);
    
    // 构建网格
    std::vector<std::vector<cv::Point2f>> grid;
    buildSortedGrid(test_corners, row_lbl, col_lbl, 8, 11, grid);
    
    // 验证网格质量
    double fill_rate, avg_row_spacing, avg_col_spacing;
    validateGridQuality(grid, 8, 11, fill_rate, avg_row_spacing, avg_col_spacing);
    
    printf("测试结果:\n");
    printf("  填充率: %.1f%%\n", 100.0 * fill_rate);
    printf("  平均行间距: %.2f 像素\n", avg_row_spacing);
    printf("  平均列间距: %.2f 像素\n", avg_col_spacing);
    
    printf("改进的网格构建功能测试完成\n");
}

void detect(const char* str, cbdetect::CornerType corner_type) {
  cbdetect::Corner corners;
  std::vector<cbdetect::Board> boards;
  cbdetect::Params params;
  params.corner_type = corner_type;

  cv::Mat img = cv::imread(str, cv::IMREAD_COLOR);
  if (img.empty()) {
    printf("Error: Could not load image '%s'\n", str);
    return;
  }

  // 图像预处理（可调试不同参数）
  img = preprocess_image(img, true, false, false); // 先只用CLAHE

  printf("Processing image: %s (size: %dx%d)\n", str, img.cols, img.rows);

  auto t1 = high_resolution_clock::now();
  cbdetect::find_corners(img, corners, params);
  auto t2 = high_resolution_clock::now();
  cbdetect::plot_corners(img, corners);
  auto t3 = high_resolution_clock::now();
  cbdetect::boards_from_corners(img, corners, boards, params);
  auto t4 = high_resolution_clock::now();
  printf("Find corners took: %.3f ms\n", duration_cast<microseconds>(t2 - t1).count() / 1000.0);
  printf("Find boards took: %.3f ms\n", duration_cast<microseconds>(t4 - t3).count() / 1000.0);
  printf("Total took: %.3f ms\n", duration_cast<microseconds>(t2 - t1).count() / 1000.0 + duration_cast<microseconds>(t4 - t3).count() / 1000.0);
  printf("Detected %zu corners and %zu boards\n", corners.p.size(), boards.size());
  cbdetect::plot_boards(img, corners, boards, params);
}

// 计算角点质量评分
double calculate_corner_quality(const cbdetect::Corner& corners, const cv::Size& img_size) {
    if (corners.p.empty()) return 0.0;

    // 1. 角点数量
    double corner_count_score = static_cast<double>(corners.p.size()) / (img_size.width * img_size.height);

    // 2. 角点分布均匀性 (使用Harris角点响应值)
    if (corners.score.empty()) return 0.0; // 确保有分数

    // 计算分数的标准差
    double mean_score = 0.0;
    for (const auto& s : corners.score) {
        mean_score += s;
    }
    mean_score /= corners.score.size();

    double variance_score = 0.0;
    for (const auto& s : corners.score) {
        variance_score += (s - mean_score) * (s - mean_score);
    }
    variance_score /= corners.score.size();

    double std_dev_score = std::sqrt(variance_score);

    // 3. 综合评分
    double quality_score = corner_count_score * 0.6 + (1.0 - std_dev_score) * 0.4; // 数量和均匀性权重
    return quality_score;
}

int main(int argc, char* argv[]) {
    // 棋盘格先验信息
    const int EXPECTED_ROWS = 8;
    const int EXPECTED_COLS = 11;
    const int EXPECTED_CORNERS = EXPECTED_ROWS * EXPECTED_COLS; // 88个角点
    
    // 测试改进的网格构建功能
    testImprovedGridBuilding();
    
    // 直接使用最优参数
    double best_clip = 3.6;
    int best_tile = 12;
    bool best_sharpen = false;
    bool best_denoise = false;

    const char* path = (argc >= 2) ? argv[1] : "example_data/04.png";
    printf("Testing image: %s\n", path);

    cv::Mat img_best = cv::imread(path, cv::IMREAD_COLOR);
    if (!img_best.empty()) {
        cv::Mat img_proc = preprocess_image(img_best, true, best_sharpen, best_denoise, best_clip, best_tile);
        
        // 初始检测
        cbdetect::Corner corners;
        cbdetect::Params params;
        params.corner_type = cbdetect::SaddlePoint;
        params.score_thr = 0.008;
        params.init_loc_thr = 0.005;
        
        printf("\n=== 初始检测 ===\n");
        cbdetect::find_corners(img_proc, corners, params);
        
        // 动态调整参数并重新检测
        int max_iterations = 3;
        for (int iter = 1; iter <= max_iterations; ++iter) {
            adjustDetectionParams(corners, EXPECTED_CORNERS, params);
            
            // 如果检测数量已经接近目标，停止迭代
            double ratio = static_cast<double>(corners.p.size()) / EXPECTED_CORNERS;
            if (ratio >= 0.9 && ratio <= 1.1) {
                printf("检测数量已接近目标，停止迭代\n");
                break;
            }
            
            if (iter < max_iterations) {
                printf("\n=== 第%d次重新检测 ===\n", iter);
                cbdetect::find_corners(img_proc, corners, params);
            }
        }
        
        // 可视化角点并保存图片
        cv::Mat result_img = img_best.clone(); // 用原图
        cbdetect::plot_corners(result_img, corners);
        
        // 输出角点坐标信息
        printf("\n=== 角点坐标信息 ===\n");
        printf("检测到 %zu 个角点:\n", corners.p.size());
        for (size_t i = 0; i < std::min(corners.p.size(), size_t(10)); ++i) {
            printf("角点 %zu: (%.1f, %.1f), 分数: %.3f\n", 
                   i, corners.p[i].x, corners.p[i].y, corners.score[i]);
        }
        if (corners.p.size() > 10) {
            printf("... 还有 %zu 个角点\n", corners.p.size() - 10);
        }
        
        // === 新增：相关性过滤、RANSAC主方向估算、局部一致性剔除、棋盘格拓扑拟合和BA优化 ===
        std::vector<cv::Point2f> cv_corners;
        for (const auto& pt : corners.p) {
            cv_corners.emplace_back(pt.x, pt.y);
        }
        
        // 相关性过滤
        printf("\n=== 相关性过滤 ===\n");
        filter_corners_by_correlation(result_img, cv_corners, 0.05, EXPECTED_CORNERS);
        
        // 改进的角点评分
        printf("\n=== 改进的角点评分 ===\n");
        cv::Mat gray_img;
        cv::cvtColor(result_img, gray_img, cv::COLOR_BGR2GRAY);
        
        std::vector<float> corner_scores = computeCornerScores(gray_img, cv_corners);
        
        // 统计评分信息
        if (!corner_scores.empty()) {
            auto minmax = std::minmax_element(corner_scores.begin(), corner_scores.end());
            float avg_score = std::accumulate(corner_scores.begin(), corner_scores.end(), 0.0f) / corner_scores.size();
            
            printf("角点评分统计: 最小值=%.3f, 最大值=%.3f, 平均值=%.3f\n", 
                   *minmax.first, *minmax.second, avg_score);
            
            // 基于评分过滤角点
            float score_threshold = avg_score * 0.5f; // 使用平均值的50%作为阈值
            std::vector<cv::Point2f> score_filtered_corners;
            std::vector<float> score_filtered_scores;
            
            for (size_t i = 0; i < cv_corners.size(); ++i) {
                if (corner_scores[i] >= score_threshold) {
                    score_filtered_corners.push_back(cv_corners[i]);
                    score_filtered_scores.push_back(corner_scores[i]);
                }
            }
            
            printf("基于评分的角点过滤: 保留 %zu/%zu 个角点 (阈值=%.3f)\n", 
                   score_filtered_corners.size(), cv_corners.size(), score_threshold);
            
            // 如果过滤后角点数量合理，使用过滤后的结果
            if (score_filtered_corners.size() >= EXPECTED_CORNERS * 0.7) {
                cv_corners = score_filtered_corners;
                printf("使用评分过滤后的角点\n");
            } else {
                printf("评分过滤过于严格，保持原有角点\n");
            }
        }
        
        if (!cv_corners.empty()) {
            printf("\n=== RANSAC主方向估算和局部一致性剔除 ===\n");
            
            // 1. RANSAC主方向估算（使用更宽松的参数）
            cv::Point2f dir_row, dir_col;
            estimateMainAxesRansac(cv_corners, dir_row, dir_col, 500, 5.0f); // 增加容差到5.0
            printf("RANSAC主方向估算完成\n");
            
            // 2. 自动推断行列标签
            std::vector<int> row_lbl, col_lbl;
            inferRowColLabels(cv_corners, row_lbl, col_lbl, EXPECTED_ROWS, EXPECTED_COLS);
            printf("行列标签推断完成，有效标签数: %zu\n", 
                   std::count_if(row_lbl.begin(), row_lbl.end(), [](int x) { return x >= 0; }));
            
            // 3. 自适应局部一致性剔除
            std::vector<char> inlier_mask;
            float k_sigma = 3.0f;
            int target_keep = EXPECTED_ROWS * EXPECTED_COLS * 0.9;   // 至少保留90%
            
            printf("开始自适应局部一致性剔除...\n");
            printf("目标保留角点数: %d (%.1f%%)\n", target_keep, 90.0);
            
            // 使用无限循环，直到达到目标或超过最大阈值
            std::vector<char> point_types; // 空向量表示所有点都是实测点
            for (;; k_sigma += 0.5f) {
                rejectOutliers2(cv_corners, row_lbl, col_lbl, dir_row, dir_col, EXPECTED_ROWS, EXPECTED_COLS, inlier_mask, point_types, k_sigma);
                int keep = (int)std::count(inlier_mask.begin(), inlier_mask.end(), 1);
                printf("  k_sigma=%.1f: 保留 %d 个角点 (%.1f%%)\n", k_sigma, keep, 100.0 * keep / cv_corners.size());
                
                if (keep >= target_keep || k_sigma > 6.0f) {
                    printf("  搜索完成: k_sigma=%.1f, 保留 %d 个角点\n", k_sigma, keep);
                    if (k_sigma > 6.0f) {
                        printf("  达到最大k_sigma限制 (6.0)\n");
                    } else {
                        printf("  达到目标保留率 (≥90%%)\n");
                    }
                    break;
                }
            }
            
            // 保留内点
            std::vector<cv::Point2f> filtered_corners;
            for (size_t i = 0; i < cv_corners.size(); ++i) {
                if (inlier_mask[i]) {
                    filtered_corners.push_back(cv_corners[i]);
                }
            }
            
            // 如果剔除后角点太少，使用更宽松的策略
            int min_expected = static_cast<int>(EXPECTED_CORNERS * 0.5); // 至少保留50%的期望角点
            if (filtered_corners.size() < min_expected) {
                printf("警告：局部一致性剔除过于严格，使用备选策略 (保留角点数: %zu < %d)\n", 
                       filtered_corners.size(), min_expected);
                
                // 如果局部一致性剔除后角点太少，直接使用相关性过滤后的角点
                if (filtered_corners.size() < 10) {
                    printf("局部一致性剔除过于严格，恢复相关性过滤后的角点\n");
                    filtered_corners = cv_corners; // 恢复相关性过滤后的角点
                } else {
                    // 使用简单的距离阈值过滤
                    std::vector<cv::Point2f> distance_filtered;
                    for (size_t i = 0; i < cv_corners.size(); ++i) {
                        bool is_good = true;
                        for (size_t j = 0; j < cv_corners.size(); ++j) {
                            if (i != j) {
                                float dist = cv::norm(cv_corners[i] - cv_corners[j]);
                                if (dist < 5.0f) { // 太近的点可能是重复的
                                    is_good = false;
                                    break;
                                }
                            }
                        }
                        if (is_good) {
                            distance_filtered.push_back(cv_corners[i]);
                        }
                    }
                    filtered_corners = distance_filtered;
                }
                printf("备选策略后保留角点数: %zu\n", filtered_corners.size());
            }
            
            cv_corners = filtered_corners;
            printf("局部一致性剔除完成，保留角点数: %zu (原: %zu)\n", 
                   cv_corners.size(), corners.p.size());
            
            // 角点补全：基于仿射变换预测缺失角点
            if (cv_corners.size() > 0) {
                // 重新计算行列标签（因为可能有些角点被剔除了）
                std::vector<int> final_row_lbl, final_col_lbl;
                inferRowColLabels(cv_corners, final_row_lbl, final_col_lbl, EXPECTED_ROWS, EXPECTED_COLS);
                
                // 创建内点掩码
                std::vector<char> final_inlier_mask(cv_corners.size(), 1);
                
                // 执行角点补全
                completeMissingCorners(gray_img, cv_corners, final_row_lbl, final_col_lbl, 
                                     final_inlier_mask, EXPECTED_ROWS, EXPECTED_COLS);
                
                printf("补全后总角点数: %zu\n", cv_corners.size());
            }
        }
        cv::Size boardSize(8, 11); // 8行11列内角点
        std::vector<cv::Point2f> refined;
        bool found = cv::findChessboardCorners(result_img, boardSize, refined,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
        if (found && refined.size() == cv_corners.size()) {
            cv::cornerSubPix(result_img, refined, cv::Size(5,5), cv::Size(-1,-1),
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 40, 0.01));
            cv_corners = refined;
        }
        if (cv_corners.size() == boardSize.area()) {
            std::vector<cv::Point3f> objectPoints;
            for(int i=0; i<boardSize.height; i++)
                for(int j=0; j<boardSize.width; j++)
                    objectPoints.emplace_back(j, i, 0);
            cv::Mat cameraMatrix = cv::Mat::eye(3,3,CV_64F);
            cv::Mat distCoeffs = cv::Mat::zeros(8,1,CV_64F);
            cv::Mat rvec, tvec;
            cv::solvePnP(objectPoints, cv_corners, cameraMatrix, distCoeffs, rvec, tvec);
            
            // 计算初始重投影误差
            double initial_error = computeReprojError(cv_corners, objectPoints, cameraMatrix, distCoeffs, rvec, tvec);
            printf("初始重投影误差: %.4f 像素\n", initial_error);
            
            // 使用Ceres进行Bundle Adjustment优化
#ifdef USE_CERES
            printf("\n=== 开始Ceres Bundle Adjustment优化 ===\n");
            
            // 构建网格用于BA优化
            std::vector<std::vector<cv::Point2f>> ba_grid(EXPECTED_ROWS, std::vector<cv::Point2f>(EXPECTED_COLS, cv::Point2f(-1, -1)));
            for (size_t i = 0; i < cv_corners.size(); ++i) {
                int r = i / EXPECTED_COLS;
                int c = i % EXPECTED_COLS;
                if (r < EXPECTED_ROWS && c < EXPECTED_COLS) {
                    ba_grid[r][c] = cv_corners[i];
                }
            }
            
            double final_cost;
            bool ba_success = optimizeChessboardPose(ba_grid, cameraMatrix, distCoeffs, 
                                                   rvec, tvec, objectPoints, final_cost);
            
            if (ba_success) {
                // 计算优化后的重投影误差
                double optimized_error = computeReprojError(cv_corners, objectPoints, cameraMatrix, distCoeffs, rvec, tvec);
                printf("优化后重投影误差: %.4f 像素 (改进: %.4f)\n", 
                       optimized_error, initial_error - optimized_error);
                
                // 重新投影并过滤角点
                std::vector<cv::Point2f> projected;
                cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, projected);
                std::vector<cv::Point2f> ba_refined;
                for(size_t i=0; i<cv_corners.size(); i++) {
                    double error = cv::norm(cv_corners[i] - projected[i]);
                    if (error < 3.0) // 使用更严格的错误阈值
                        ba_refined.push_back(cv_corners[i]);
                }
                if (ba_refined.size() > cv_corners.size() * 0.8) {
                    cv_corners = ba_refined;
                    printf("Ceres BA优化完成，保留 %zu 个角点\n", cv_corners.size());
                } else {
                    printf("Ceres BA优化过滤过于严格，保持原有角点\n");
                }
            } else {
                printf("Ceres BA优化失败，使用传统方法\n");
                // 使用传统的重投影过滤方法
            std::vector<cv::Point2f> projected;
            cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, projected);
            std::vector<cv::Point2f> ba_refined;
            for(size_t i=0; i<cv_corners.size(); i++) {
                double error = cv::norm(cv_corners[i] - projected[i]);
                if (error < 5.0) // 放宽BA优化的错误阈值
                    ba_refined.push_back(cv_corners[i]);
            }
                if (ba_refined.size() > cv_corners.size() * 0.8) {
                cv_corners = ba_refined;
                    printf("传统BA优化完成，保留 %zu 个角点\n", cv_corners.size());
            } else {
                    printf("传统BA优化过滤过于严格，保持原有角点\n");
                }
            }
#else
            // 不使用Ceres时的传统方法
            std::vector<cv::Point2f> projected;
            cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, projected);
            std::vector<cv::Point2f> ba_refined;
            for(size_t i=0; i<cv_corners.size(); i++) {
                double error = cv::norm(cv_corners[i] - projected[i]);
                if (error < 5.0) // 放宽BA优化的错误阈值
                    ba_refined.push_back(cv_corners[i]);
            }
            if (ba_refined.size() > cv_corners.size() * 0.8) {
                cv_corners = ba_refined;
                printf("传统BA优化完成，保留 %zu 个角点\n", cv_corners.size());
            } else {
                printf("传统BA优化过滤过于严格，保持原有角点\n");
            }
#endif
        }
        // 5. 可视化优化后角点
        printf("\n=== 最终角点统计 ===\n");
        printf("RANSAC + 局部一致性剔除后角点数: %zu\n", cv_corners.size());
        
        // === 新增：改进的网格构建和可视化 ===
        if (!cv_corners.empty()) {
            printf("\n=== 改进的网格构建和可视化 ===\n");
            
            // 1. 使用改进的行列标签分配
            std::vector<int> improved_row_lbl, improved_col_lbl;
            inferRowColLabels(cv_corners, improved_row_lbl, improved_col_lbl, EXPECTED_ROWS, EXPECTED_COLS);
            
            // 2. 构建有序二维角点网格
            std::vector<std::vector<cv::Point2f>> sorted_grid;
            buildSortedGrid(cv_corners, improved_row_lbl, improved_col_lbl, EXPECTED_ROWS, EXPECTED_COLS, sorted_grid);
            
            // 3. 验证网格质量并统计信息
            double fill_rate, avg_row_spacing, avg_col_spacing;
            validateGridQuality(sorted_grid, EXPECTED_ROWS, EXPECTED_COLS, 
                               fill_rate, avg_row_spacing, avg_col_spacing);
            
            printf("网格质量统计:\n");
            printf("  填充率: %.1f%% (%d/%d 个单元格)\n", 
                   100.0 * fill_rate, (int)(fill_rate * EXPECTED_ROWS * EXPECTED_COLS), 
                   EXPECTED_ROWS * EXPECTED_COLS);
            printf("  平均行间距: %.2f 像素\n", avg_row_spacing);
            printf("  平均列间距: %.2f 像素\n", avg_col_spacing);
            
            // 4. 在原图上绘制最终优化后的角点
        for (const auto& pt : cv_corners) {
            cv::circle(result_img, pt, 6, cv::Scalar(0,0,255), 2);
            }
            
            // 5. 绘制网格连线（如果网格填充率足够高）
            if (fill_rate > 0.6) { // 降低到60%以上填充率就绘制网格
                printf("绘制网格连线...\n");
                
                // 绘制行连线（蓝色）
                for (int r = 0; r < EXPECTED_ROWS; ++r) {
                    std::vector<cv::Point2f> row_points;
                    for (int c = 0; c < EXPECTED_COLS; ++c) {
                        if (sorted_grid[r][c].x >= 0 && sorted_grid[r][c].y >= 0) {
                            row_points.push_back(sorted_grid[r][c]);
                        }
                    }
                    if (row_points.size() >= 2) {
                        for (size_t i = 1; i < row_points.size(); ++i) {
                            cv::line(result_img, row_points[i-1], row_points[i], cv::Scalar(255,0,0), 2);
                        }
                    }
                }
                
                // 绘制列连线（绿色）- 只连接相邻的有效点
                for (int c = 0; c < EXPECTED_COLS; ++c) {
                    std::vector<cv::Point2f> col_points;
                    for (int r = 0; r < EXPECTED_ROWS; ++r) {
                        if (sorted_grid[r][c].x >= 0 && sorted_grid[r][c].y >= 0) {
                            col_points.push_back(sorted_grid[r][c]);
                        }
                    }
                    if (col_points.size() >= 2) {
                        // 只连接相邻的点，避免对角线连接
                        for (size_t i = 1; i < col_points.size(); ++i) {
                            // 检查两个点是否在相邻的行
                            cv::Point2f pt1 = col_points[i-1];
                            cv::Point2f pt2 = col_points[i];
                            
                            // 找到这两个点在网格中的位置
                            int r1 = -1, r2 = -1;
                            for (int r = 0; r < EXPECTED_ROWS; ++r) {
                                if (sorted_grid[r][c].x == pt1.x && sorted_grid[r][c].y == pt1.y) {
                                    r1 = r;
                                }
                                if (sorted_grid[r][c].x == pt2.x && sorted_grid[r][c].y == pt2.y) {
                                    r2 = r;
                                }
                            }
                            
                            // 只有当两个点在相邻行时才连接
                            if (r1 >= 0 && r2 >= 0 && std::abs(r1 - r2) == 1) {
                                cv::line(result_img, pt1, pt2, cv::Scalar(0,255,0), 2);
                            } else {
                                // 对于非相邻的点，用虚线或不同颜色表示
                                cv::line(result_img, pt1, pt2, cv::Scalar(0,255,255), 1);
                            }
                        }
                    }
                }
                
                // 6. 打印网格坐标信息
                printf("\n=== 网格坐标信息 ===\n");
                for (int r = 0; r < std::min(5, EXPECTED_ROWS); ++r) { // 只打印前5行
                    for (int c = 0; c < std::min(5, EXPECTED_COLS); ++c) { // 只打印前5列
                        cv::Point2f pt = sorted_grid[r][c];
                        if (pt.x >= 0 && pt.y >= 0) {
                            printf("(%d,%d):(%.1f,%.1f) ", r, c, pt.x, pt.y);
                        } else {
                            printf("(%d,%d):(---,---) ", r, c);
                        }
                    }
                    printf("\n");
                }
                if (EXPECTED_ROWS > 5 || EXPECTED_COLS > 5) {
                    printf("... (网格较大，只显示前5x5部分)\n");
                }
            } else {
                printf("网格填充率不足，跳过网格连线绘制\n");
            }
        }
        
        // 保存最终优化结果图片
        std::string filename = std::string(path);
        size_t last_slash = filename.find_last_of("/\\");
        size_t last_dot = filename.find_last_of(".");
        std::string basename = filename.substr(last_slash + 1, last_dot - last_slash - 1);
        std::string output_name2 = basename + "_ransac_optimized_result.png";
        cv::imwrite(output_name2, result_img);
        printf("RANSAC优化结果已保存为: %s\n", output_name2.c_str());
        
        // 显示结果
        cv::imshow("RANSAC Optimized Result", result_img);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    return 0;
} 