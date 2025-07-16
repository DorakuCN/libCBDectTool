#include "cbdetect/corner.h"
#include <algorithm>

namespace cbdetect {

// Corners class implementation

void Corners::clear() {
    corners.clear();
}

size_t Corners::size() const {
    return corners.size();
}

bool Corners::empty() const {
    return corners.empty();
}

Corner& Corners::operator[](size_t index) {
    return corners[index];
}

const Corner& Corners::operator[](size_t index) const {
    return corners[index];
}

std::vector<Corner>::iterator Corners::begin() {
    return corners.begin();
}

std::vector<Corner>::iterator Corners::end() {
    return corners.end();
}

std::vector<Corner>::const_iterator Corners::begin() const {
    return corners.begin();
}

std::vector<Corner>::const_iterator Corners::end() const {
    return corners.end();
}

void Corners::push_back(const Corner& corner) {
    corners.push_back(corner);
}

void Corners::filterByScore(float threshold) {
    corners.erase(
        std::remove_if(corners.begin(), corners.end(),
                      [threshold](const Corner& corner) {
                          return corner.quality_score < threshold;
                      }),
        corners.end()
    );
}

std::vector<cv::Point2f> Corners::getPoints() const {
    std::vector<cv::Point2f> points;
    points.reserve(corners.size());
    
    for (const auto& corner : corners) {
        points.push_back(corner.pt);
    }
    
    return points;
}

std::vector<cv::Vec2f> Corners::getDirections1() const {
    std::vector<cv::Vec2f> directions;
    directions.reserve(corners.size());
    
    for (const auto& corner : corners) {
        directions.push_back(corner.v1);
    }
    
    return directions;
}

std::vector<cv::Vec2f> Corners::getDirections2() const {
    std::vector<cv::Vec2f> directions;
    directions.reserve(corners.size());
    
    for (const auto& corner : corners) {
        directions.push_back(corner.v2);
    }
    
    return directions;
}

std::vector<float> Corners::getScores() const {
    std::vector<float> scores;
    scores.reserve(corners.size());
    
    for (const auto& corner : corners) {
        scores.push_back(corner.quality_score);
    }
    
    return scores;
}

} // namespace cbdetect 