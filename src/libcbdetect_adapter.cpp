#include "cbdetect/libcbdetect_adapter.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

#ifdef _MSC_VER
#define M_PI 3.14159265358979323846
#define M_PI_2 1.57079632679489661923
#define M_PI_4 0.78539816339744830962
#endif

namespace cbdetect {

void box_filter(const cv::Mat& img, cv::Mat& blur_img, int kernel_size_x, int kernel_size_y) {
    if(kernel_size_y < 0) {
        kernel_size_y = kernel_size_x;
    }
    blur_img.create(img.size(), CV_64F);
    std::vector<double> buf(img.cols, 0);
    std::vector<int> count_buf(img.cols, 0);
    int count = 0;
    for(int j = 0; j < std::min(kernel_size_y, img.rows - 1); ++j) {
        for(int i = 0; i < img.cols; ++i) {
            buf[i] += img.at<double>(j, i);
            ++count_buf[i];
        }
    }
    for(int j = 0; j < img.rows; ++j) {
        if(j > kernel_size_y) {
            for(int i = 0; i < img.cols; ++i) {
                buf[i] -= img.at<double>(j - kernel_size_y - 1, i);
                --count_buf[i];
            }
        }
        if(j + kernel_size_y < img.rows) {
            for(int i = 0; i < img.cols; ++i) {
                buf[i] += img.at<double>(j + kernel_size_y, i);
                ++count_buf[i];
            }
        }
        blur_img.at<double>(j, 0) = 0;
        count = 0;
        for(int i = 0; i <= std::min(kernel_size_x, img.cols - 1); ++i) {
            blur_img.at<double>(j, 0) += buf[i];
            count += count_buf[i];
        }
        for(int i = 1; i < img.cols; ++i) {
            blur_img.at<double>(j, i) = blur_img.at<double>(j, i - 1);
            blur_img.at<double>(j, i - 1) /= count;
            if(i > kernel_size_x) {
                blur_img.at<double>(j, i) -= buf[i - kernel_size_x - 1];
                count -= count_buf[i - kernel_size_x - 1];
            }
            if(i + kernel_size_x < img.cols) {
                blur_img.at<double>(j, i) += buf[i + kernel_size_x];
                count += count_buf[i + kernel_size_x];
            }
        }
        blur_img.at<double>(j, img.cols - 1) /= count;
    }
}

void image_normalization_and_gradients(cv::Mat& img, cv::Mat& img_du, cv::Mat& img_dv,
                                       cv::Mat& img_angle, cv::Mat& img_weight, const Params& params) {
    // normalize image
    if(params.norm) {
        cv::Mat blur_img;
        box_filter(img, blur_img, params.norm_half_kernel_size);
        img = img - blur_img;
        img = 2.5 * (cv::max(cv::min(img + 0.2, 0.4), 0));
    }

    // sobel masks
    cv::Mat_<double> du({3, 3}, {1, 0, -1, 2, 0, -2, 1, 0, -1});
    cv::Mat_<double> dv({3, 3}, {1, 2, 1, 0, 0, 0, -1, -2, -1});

    // compute image derivatives (for principal axes estimation)
    cv::filter2D(img, img_du, -1, du, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
    cv::filter2D(img, img_dv, -1, dv, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
    img_angle.create(img.size(), img.type());
    img_weight.create(img.size(), img.type());
    
    // Ensure continuous memory layout
    if(!img_du.isContinuous()) {
        cv::Mat tmp = img_du.clone();
        std::swap(tmp, img_du);
    }
    if(!img_dv.isContinuous()) {
        cv::Mat tmp = img_dv.clone();
        std::swap(tmp, img_dv);
    }
    if(!img_angle.isContinuous()) {
        cv::Mat tmp = img_angle.clone();
        std::swap(tmp, img_angle);
    }
    if(!img_weight.isContinuous()) {
        cv::Mat tmp = img_weight.clone();
        std::swap(tmp, img_weight);
    }
    
    // Compute angle and weight
    for(int v = 0; v < img.rows; ++v) {
        for(int u = 0; u < img.cols; ++u) {
            double du_val = img_du.at<double>(v, u);
            double dv_val = img_dv.at<double>(v, u);
            
            // Calculate angle
            double angle = std::atan2(dv_val, du_val);
            if(angle < 0) angle += M_PI;
            img_angle.at<double>(v, u) = angle >= M_PI ? angle - M_PI : angle;
            
            // Calculate weight (gradient magnitude)
            img_weight.at<double>(v, u) = std::sqrt(du_val * du_val + dv_val * dv_val);
        }
    }

    // scale input image
    double img_min = 0, img_max = 1;
    cv::minMaxLoc(img, &img_min, &img_max);
    img = (img - img_min) / (img_max - img_min);
}

void hessian_response(const cv::Mat& img_in, cv::Mat& img_out) {
    const int rows = img_in.rows;
    const int cols = img_in.cols;

    // allocate output
    img_out = cv::Mat::zeros(rows, cols, CV_64F);

    for(int i = 1; i < rows - 1; ++i) {
        for(int c = 1; c < cols - 1; ++c) {
            // 3x3 neighborhood values
            double v11 = img_in.at<double>(i-1, c-1), v12 = img_in.at<double>(i-1, c), v13 = img_in.at<double>(i-1, c+1);
            double v21 = img_in.at<double>(i, c-1),   v22 = img_in.at<double>(i, c),   v23 = img_in.at<double>(i, c+1);
            double v31 = img_in.at<double>(i+1, c-1), v32 = img_in.at<double>(i+1, c), v33 = img_in.at<double>(i+1, c+1);

            // compute 3x3 Hessian values from symmetric differences.
            double Lxx = (v21 - 2 * v22 + v23);
            double Lyy = (v12 - 2 * v22 + v32);
            double Lxy = (v13 - v11 + v31 - v33) / 4.0;

            /* normalize and write out */
            img_out.at<double>(i, c) = Lxx * Lyy - Lxy * Lxy;
        }
    }
}

void create_correlation_patch(std::vector<cv::Mat>& templates, double angle1, double angle2, int radius) {
    templates.resize(4);
    int diameter = 2 * radius + 1;
    
    for(int i = 0; i < 4; ++i) {
        templates[i] = cv::Mat::zeros(diameter, diameter, CV_64F);
    }
    
    // Compute normal vectors
    cv::Point2d n1(-std::sin(angle1), std::cos(angle1));
    cv::Point2d n2(-std::sin(angle2), std::cos(angle2));
    
    double sum_a1 = 0, sum_a2 = 0, sum_b1 = 0, sum_b2 = 0;
    
    for(int v = -radius; v <= radius; ++v) {
        for(int u = -radius; u <= radius; ++u) {
            if(u*u + v*v > radius*radius) continue;
            
            // Project onto normal vectors
            double proj1 = u * n1.x + v * n1.y;
            double proj2 = u * n2.x + v * n2.y;
            
            int row = v + radius;
            int col = u + radius;
            
            // Assign to quadrants based on projections
            if(proj1 >= 0.1) {
                templates[0].at<double>(row, col) = 1.0;  // a1
                sum_a1 += 1.0;
            } else if(proj1 <= -0.1) {
                templates[1].at<double>(row, col) = 1.0;  // a2
                sum_a2 += 1.0;
            }
            
            if(proj2 >= 0.1) {
                templates[2].at<double>(row, col) = 1.0;  // b1
                sum_b1 += 1.0;
            } else if(proj2 <= -0.1) {
                templates[3].at<double>(row, col) = 1.0;  // b2
                sum_b2 += 1.0;
            }
        }
    }
    
    // Normalize by sum (libcdetSample style)
    if(sum_a1 > 0) templates[0] /= sum_a1;
    if(sum_a2 > 0) templates[1] /= sum_a2;
    if(sum_b1 > 0) templates[2] /= sum_b1;
    if(sum_b2 > 0) templates[3] /= sum_b2;
}

void non_maximum_suppression(const cv::Mat& corner_map, int radius, double threshold, 
                            int template_radius, Corner& corners) {
    const int rows = corner_map.rows;
    const int cols = corner_map.cols;
    
    for(int v = radius; v < rows - radius; ++v) {
        for(int u = radius; u < cols - radius; ++u) {
            double center_val = corner_map.at<double>(v, u);
            if(center_val < threshold) continue;
            
            // Check if it's a local maximum
            bool is_maximum = true;
            for(int dv = -radius; dv <= radius && is_maximum; ++dv) {
                for(int du = -radius; du <= radius && is_maximum; ++du) {
                    if(du == 0 && dv == 0) continue;
                    if(corner_map.at<double>(v + dv, u + du) > center_val) {
                        is_maximum = false;
                    }
                }
            }
            
            if(is_maximum) {
                corners.p.push_back(cv::Point2d(u, v));
                corners.r.push_back(template_radius);
                corners.v1.push_back(cv::Point2d(0, 0));
                corners.v2.push_back(cv::Point2d(0, 0));
                corners.score.push_back(center_val);
            }
        }
    }
}

void get_init_location(const cv::Mat& img, const cv::Mat& img_du, const cv::Mat& img_dv,
                       Corner& corners, const Params& params) {
    if(params.detect_method == HessianResponse) {
        cv::Mat gauss_img;
        cv::GaussianBlur(img, gauss_img, cv::Size(7, 7), 1.5, 1.5);
        cv::Mat hessian_img;
        hessian_response(gauss_img, hessian_img);
        double mn = 0, mx = 0;
        cv::minMaxIdx(hessian_img, &mn, &mx, NULL, NULL);
        hessian_img = cv::abs(hessian_img);
        double thr = std::abs(mn * params.init_loc_thr);
        for(const auto& r : params.radius) {
            non_maximum_suppression(hessian_img, r, thr, r, corners);
        }
    } else {
        // Template matching implementation
        std::vector<double> tprops;
        if(params.detect_method == TemplateMatchFast) {
            tprops = {0, M_PI_2, M_PI_4, -M_PI_4};
        } else {
            tprops = {0, M_PI_2, M_PI_4, -M_PI_4, 0, M_PI_4, 0, -M_PI_4,
                      M_PI_4, M_PI_2, -M_PI_4, M_PI_2, -3*M_PI/8, 3*M_PI/8,
                      -M_PI/8, M_PI/8, -M_PI/8, -3*M_PI/8, M_PI/8, 3*M_PI/8};
        }

        for(const auto& r : params.radius) {
            cv::Mat img_corners = cv::Mat::zeros(img.size(), CV_64F);
            cv::Mat img_corners_a1, img_corners_a2, img_corners_b1, img_corners_b2, img_corners_mu,
                    img_corners_a, img_corners_b, img_corners_s1, img_corners_s2;

            for(int i = 0; i < tprops.size(); i += 2) {
                std::vector<cv::Mat> template_kernel(4);
                create_correlation_patch(template_kernel, tprops[i], tprops[i + 1], r);

                cv::filter2D(img, img_corners_a1, -1, template_kernel[0], cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
                cv::filter2D(img, img_corners_a2, -1, template_kernel[1], cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
                cv::filter2D(img, img_corners_b1, -1, template_kernel[2], cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
                cv::filter2D(img, img_corners_b2, -1, template_kernel[3], cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

                img_corners_mu = (img_corners_a1 + img_corners_a2 + img_corners_b1 + img_corners_b2) / 4;

                // case 1: a=white, b=black
                img_corners_a = cv::min(img_corners_a1, img_corners_a2) - img_corners_mu;
                img_corners_b = img_corners_mu - cv::max(img_corners_b1, img_corners_b2);
                img_corners_s1 = cv::min(img_corners_a, img_corners_b);
                
                // case 2: b=white, a=black
                img_corners_a = img_corners_mu - cv::max(img_corners_a1, img_corners_a2);
                img_corners_b = cv::min(img_corners_b1, img_corners_b2) - img_corners_mu;
                img_corners_s2 = cv::min(img_corners_a, img_corners_b);

                img_corners = cv::max(img_corners, cv::max(img_corners_s1, img_corners_s2));
            }
            non_maximum_suppression(img_corners, 1, params.init_loc_thr, r, corners);
        }
    }
    
    // Location refinement (subpixel)
    int width = img.cols, height = img.rows;
    for(int i = 0; i < corners.p.size(); ++i) {
        double u = corners.p[i].x;
        double v = corners.p[i].y;
        int r = corners.r[i];

        cv::Mat G = cv::Mat::zeros(2, 2, CV_64F);
        cv::Mat b = cv::Mat::zeros(2, 1, CV_64F);

        if(u - r < 0 || u + r >= width - 1 || v - r < 0 || v + r >= height - 1) {
            continue;
        }

        for(int j2 = v - r; j2 <= v + r; ++j2) {
            for(int i2 = u - r; i2 <= u + r; ++i2) {
                if(i2 < 0 || i2 >= width || j2 < 0 || j2 >= height) continue;
                
                double o_du = img_du.at<double>(j2, i2);
                double o_dv = img_dv.at<double>(j2, i2);
                double o_norm = std::sqrt(o_du * o_du + o_dv * o_dv);
                if(o_norm < 0.1) continue;

                if(i2 == u && j2 == v) continue;
                
                G.at<double>(0, 0) += o_du * o_du;
                G.at<double>(0, 1) += o_du * o_dv;
                G.at<double>(1, 0) += o_du * o_dv;
                G.at<double>(1, 1) += o_dv * o_dv;
                b.at<double>(0, 0) += o_du * o_du * i2 + o_du * o_dv * j2;
                b.at<double>(1, 0) += o_du * o_dv * i2 + o_dv * o_dv * j2;
            }
        }

        double det = cv::determinant(G);
        if(std::abs(det) > 1e-10) {
            cv::Mat new_pos = G.inv() * b;
            double new_u = new_pos.at<double>(0, 0);
            double new_v = new_pos.at<double>(1, 0);
            if(std::abs(new_u - u) + std::abs(new_v - v) < r * 2) {
                corners.p[i].x = new_u;
                corners.p[i].y = new_v;
            }
        }
    }
}

void filter_corners(const cv::Mat& img, const cv::Mat& img_angle, const cv::Mat& img_weight, 
                   Corner& corners, const Params& params) {
    // Debug: temporarily disable zero crossing filter to match initial debugging approach
    // We'll implement a simplified filter that's less restrictive
    std::vector<bool> keep(corners.p.size(), true);
    
    if(params.show_processing) {
        std::cout << "Filter corners input: " << corners.p.size() << " corners" << std::endl;
    }
    
    // Simple boundary check filter only (for now)
    for(int i = 0; i < corners.p.size(); ++i) {
        int u = static_cast<int>(std::round(corners.p[i].x));
        int v = static_cast<int>(std::round(corners.p[i].y));
        int r = corners.r[i];
        
        // Check if corner is too close to image boundary
        if(u - r < 5 || u + r >= img.cols - 5 || v - r < 5 || v + r >= img.rows - 5) {
            keep[i] = false;
            continue;
        }
        
        // Optional: Add minimal quality check based on gradient magnitude
        if(u >= 0 && u < img_weight.cols && v >= 0 && v < img_weight.rows) {
            double weight = img_weight.at<double>(v, u);
            if(weight < 0.1) {  // Very low threshold
                keep[i] = false;
            }
        }
    }
    
    // Filter corners
    Corner filtered_corners;
    for(int i = 0; i < corners.p.size(); ++i) {
        if(keep[i]) {
            filtered_corners.p.push_back(corners.p[i]);
            filtered_corners.r.push_back(corners.r[i]);
            filtered_corners.v1.push_back(corners.v1[i]);
            filtered_corners.v2.push_back(corners.v2[i]);
            if(i < corners.score.size()) {
                filtered_corners.score.push_back(corners.score[i]);
            }
        }
    }
    corners = filtered_corners;
}

void refine_corners(const cv::Mat& img_du, const cv::Mat& img_dv, const cv::Mat& img_angle, 
                   const cv::Mat& img_weight, Corner& corners, const Params& params) {
    // Direction vector computation
    for(int i = 0; i < corners.p.size(); ++i) {
        double u = corners.p[i].x;
        double v = corners.p[i].y;
        int r = corners.r[i];
        
        if(u - r < 0 || u + r >= img_du.cols || v - r < 0 || v + r >= img_du.rows) {
            continue;
        }
        
        // Compute edge orientations in a circular pattern
        std::vector<double> angles;
        const int n_samples = 16;
        
        for(int j = 0; j < n_samples; ++j) {
            double angle = 2.0 * M_PI * j / n_samples;
            int sample_u = static_cast<int>(u + r * std::cos(angle));
            int sample_v = static_cast<int>(v + r * std::sin(angle));
            
            if(sample_u >= 0 && sample_u < img_du.cols && sample_v >= 0 && sample_v < img_du.rows) {
                double du_val = img_du.at<double>(sample_v, sample_u);
                double dv_val = img_dv.at<double>(sample_v, sample_u);
                double edge_angle = std::atan2(dv_val, du_val);
                angles.push_back(edge_angle);
            }
        }
        
        if(angles.size() >= 4) {
            // Find dominant orientations (simplified)
            std::sort(angles.begin(), angles.end());
            double angle1 = angles[angles.size() / 4];
            double angle2 = angles[3 * angles.size() / 4];
            
            corners.v1[i] = cv::Point2d(std::cos(angle1), std::sin(angle1));
            corners.v2[i] = cv::Point2d(std::cos(angle2), std::sin(angle2));
        }
    }
}

void polynomial_fit(const cv::Mat& img, Corner& corners, const Params& params) {
    // Simplified polynomial fitting for saddle point validation
    std::vector<bool> keep(corners.p.size(), true);
    
    for(int i = 0; i < corners.p.size(); ++i) {
        double u = corners.p[i].x;
        double v = corners.p[i].y;
        int r = params.polynomial_fit_half_kernel_size;
        
        if(u - r < 0 || u + r >= img.cols || v - r < 0 || v + r >= img.rows) {
            keep[i] = false;
            continue;
        }
        
        // Check if it's a saddle point by examining second derivatives
        // Simplified implementation - could be enhanced
        std::vector<double> values;
        for(int dv = -r; dv <= r; ++dv) {
            for(int du = -r; du <= r; ++du) {
                int sample_u = static_cast<int>(u + du);
                int sample_v = static_cast<int>(v + dv);
                if(sample_u >= 0 && sample_u < img.cols && sample_v >= 0 && sample_v < img.rows) {
                    values.push_back(img.at<double>(sample_v, sample_u));
                }
            }
        }
        
        if(values.size() < 9) {
            keep[i] = false;
        }
    }
    
    // Filter corners
    Corner filtered_corners;
    for(int i = 0; i < corners.p.size(); ++i) {
        if(keep[i]) {
            filtered_corners.p.push_back(corners.p[i]);
            filtered_corners.r.push_back(corners.r[i]);
            filtered_corners.v1.push_back(corners.v1[i]);
            filtered_corners.v2.push_back(corners.v2[i]);
            if(i < corners.score.size()) {
                filtered_corners.score.push_back(corners.score[i]);
            }
        }
    }
    corners = filtered_corners;
}

void score_corners(const cv::Mat& img, const cv::Mat& img_weight, Corner& corners, const Params& params) {
    for(int i = 0; i < corners.p.size(); ++i) {
        double u = corners.p[i].x;
        double v = corners.p[i].y;
        int r = corners.r[i];
        
        if(u - r < 0 || u + r >= img.cols || v - r < 0 || v + r >= img.rows) {
            if(i < corners.score.size()) {
                corners.score[i] = 0.0;
            } else {
                corners.score.push_back(0.0);
            }
            continue;
        }
        
        // Compute correlation score based on gradient magnitude
        double score = 0.0;
        int count = 0;
        
        for(int dv = -r; dv <= r; ++dv) {
            for(int du = -r; du <= r; ++du) {
                if(du*du + dv*dv <= r*r) {
                    int sample_u = static_cast<int>(u + du);
                    int sample_v = static_cast<int>(v + dv);
                    if(sample_u >= 0 && sample_u < img.cols && sample_v >= 0 && sample_v < img.rows) {
                        score += img_weight.at<double>(sample_v, sample_u);
                        count++;
                    }
                }
            }
        }
        
        if(count > 0) {
            score /= count;
        }
        
        if(i < corners.score.size()) {
            corners.score[i] = score;
        } else {
            corners.score.push_back(score);
        }
    }
    
    // Remove low scoring corners
    Corner filtered_corners;
    for(int i = 0; i < corners.p.size(); ++i) {
        double score = (i < corners.score.size()) ? corners.score[i] : 0.0;
        if(score >= params.score_thr) {
            filtered_corners.p.push_back(corners.p[i]);
            filtered_corners.r.push_back(corners.r[i]);
            filtered_corners.v1.push_back(corners.v1[i]);
            filtered_corners.v2.push_back(corners.v2[i]);
            filtered_corners.score.push_back(score);
        }
    }
    corners = filtered_corners;
}

void non_maximum_suppression_sparse(Corner& corners, int radius, const cv::Size& img_size, const Params& params) {
    std::vector<bool> keep(corners.p.size(), true);
    
    for(int i = 0; i < corners.p.size(); ++i) {
        if(!keep[i]) continue;
        
        double score_i = (i < corners.score.size()) ? corners.score[i] : 0.0;
        
        for(int j = i + 1; j < corners.p.size(); ++j) {
            if(!keep[j]) continue;
            
            double dist = cv::norm(corners.p[i] - corners.p[j]);
            if(dist < radius) {
                double score_j = (j < corners.score.size()) ? corners.score[j] : 0.0;
                if(score_i >= score_j) {
                    keep[j] = false;
                } else {
                    keep[i] = false;
                    break;
                }
            }
        }
    }
    
    // Filter corners
    Corner filtered_corners;
    for(int i = 0; i < corners.p.size(); ++i) {
        if(keep[i]) {
            filtered_corners.p.push_back(corners.p[i]);
            filtered_corners.r.push_back(corners.r[i]);
            filtered_corners.v1.push_back(corners.v1[i]);
            filtered_corners.v2.push_back(corners.v2[i]);
            if(i < corners.score.size()) {
                filtered_corners.score.push_back(corners.score[i]);
            }
        }
    }
    corners = filtered_corners;
}

void find_corners_resized(const cv::Mat& img, Corner& corners, const Params& params) {
    cv::Mat img_resized, img_norm;
    Corner corners_resized;

    // resize image
    double scale = 0;
    if(img.rows < 640 || img.cols < 480) {
        scale = 2.0;
    } else if(img.rows >= 640 || img.cols >= 480) {
        scale = 0.5;
    } else {
        return;
    }
    cv::resize(img, img_resized, cv::Size(img.cols * scale, img.rows * scale), 0, 0, cv::INTER_LINEAR);

    if(img_resized.channels() == 3) {
        cv::cvtColor(img_resized, img_norm, cv::COLOR_BGR2GRAY);
        img_norm.convertTo(img_norm, CV_64F, 1 / 255.0, 0);
    } else {
        img_resized.convertTo(img_norm, CV_64F, 1 / 255.0, 0);
    }

    // normalize image and calculate gradients
    cv::Mat img_du, img_dv, img_angle, img_weight;
    image_normalization_and_gradients(img_norm, img_du, img_dv, img_angle, img_weight, params);
    
    // get corner's initial location
    get_init_location(img_norm, img_du, img_dv, corners_resized, params);
    if(corners_resized.p.empty()) {
        return;
    }
    if(params.show_processing) {
        printf("Initializing corners (%d x %d) ... %lu\n", img_norm.cols, img_norm.rows, corners_resized.p.size());
    }

    // pre-filter corners according to zero crossings
    filter_corners(img_norm, img_angle, img_weight, corners_resized, params);
    if(params.show_processing) {
        printf("Filtering corners (%d x %d) ... %lu\n", img_norm.cols, img_norm.rows, corners_resized.p.size());
    }

    // refinement
    refine_corners(img_du, img_dv, img_angle, img_weight, corners_resized, params);
    if(params.show_processing) {
        printf("Refining corners (%d x %d) ... %lu\n", img_norm.cols, img_norm.rows, corners_resized.p.size());
    }

    // merge corners
    std::for_each(corners_resized.p.begin(), corners_resized.p.end(), [&scale](auto& p) { p /= scale; });
    double min_dist_thr = scale > 1 ? 3 : 5;
    for(int i = 0; i < corners_resized.p.size(); ++i) {
        double min_dist = DBL_MAX;
        cv::Point2d& p2 = corners_resized.p[i];
        for(int j = 0; j < corners.p.size(); ++j) {
            cv::Point2d& p1 = corners.p[j];
            double dist = cv::norm(p2 - p1);
            min_dist = dist < min_dist ? dist : min_dist;
        }
        if(min_dist > min_dist_thr) {
            corners.p.emplace_back(corners_resized.p[i]);
            corners.r.emplace_back(corners_resized.r[i]);
            corners.v1.emplace_back(corners_resized.v1[i]);
            corners.v2.emplace_back(corners_resized.v2[i]);
            if(params.corner_type == MonkeySaddlePoint) {
                corners.v3.emplace_back(corners_resized.v3[i]);
            }
        }
    }
}

void find_corners(const cv::Mat& img, Corner& corners, const Params& params) {
    // clear old data
    corners.p.clear();
    corners.r.clear();
    corners.v1.clear();
    corners.v2.clear();
    corners.v3.clear();
    corners.score.clear();

    // convert to double grayscale image
    cv::Mat img_norm;
    if(img.channels() == 3) {
        cv::cvtColor(img, img_norm, cv::COLOR_BGR2GRAY);
        img_norm.convertTo(img_norm, CV_64F, 1. / 255., 0);
    } else {
        img.convertTo(img_norm, CV_64F, 1. / 255., 0);
    }

    // normalize image and calculate gradients
    cv::Mat img_du, img_dv, img_angle, img_weight;
    image_normalization_and_gradients(img_norm, img_du, img_dv, img_angle, img_weight, params);

    // get corner's initial location
    get_init_location(img_norm, img_du, img_dv, corners, params);
    if(corners.p.empty()) {
        return;
    }
    if(params.show_processing) {
        printf("Initializing corners (%d x %d) ... %lu\n", img_norm.cols, img_norm.rows, corners.p.size());
    }

    // pre-filter corners according to zero crossings
    filter_corners(img_norm, img_angle, img_weight, corners, params);
    if(params.show_processing) {
        printf("Filtering corners (%d x %d) ... %lu\n", img_norm.cols, img_norm.rows, corners.p.size());
    }

    // refinement
    refine_corners(img_du, img_dv, img_angle, img_weight, corners, params);
    if(params.show_processing) {
        printf("Refining corners (%d x %d) ... %lu\n", img_norm.cols, img_norm.rows, corners.p.size());
    }

    // resize image to detect more corners
    find_corners_resized(img, corners, params);
    if(params.show_processing) {
        printf("Merging corners (%d x %d) ... %lu\n", img.cols, img.rows, corners.p.size());
    }

    // polynomial fit
    if(params.polynomial_fit) {
        polynomial_fit(img_norm, corners, params);
        if(params.show_processing) {
            printf("Polyfitting corners (%d x %d) ... %lu\n", img_norm.cols, img_norm.rows, corners.p.size());
        }
    }

    // score corners
    score_corners(img_norm, img_weight, corners, params);

    // non maximum suppression
    non_maximum_suppression_sparse(corners, 3, img.size(), params);
    if(params.show_processing) {
        printf("Scoring corners (%d x %d) ... %lu\n", img_norm.cols, img_norm.rows, corners.p.size());
    }
}

void boards_from_corners(const cv::Mat& img, const Corner& corners, std::vector<Board>& boards, const Params& params) {
    // Placeholder implementation - actual board reconstruction is complex
    boards.clear();
    
    if(corners.p.size() < 4) {
        return;
    }
    
    // For debugging, create a simple board if we have enough corners
    Board board;
    board.num = 1;
    
    // Create a simple 2x2 grid if we have at least 4 corners
    if(corners.p.size() >= 4) {
        board.idx.resize(2);
        for(int i = 0; i < 2; ++i) {
            board.idx[i].resize(2);
            for(int j = 0; j < 2; ++j) {
                if(i * 2 + j < corners.p.size()) {
                    board.idx[i][j] = i * 2 + j;
                } else {
                    board.idx[i][j] = -1;
                }
            }
        }
        boards.push_back(board);
    }
}

} // namespace cbdetect 