#ifndef CBDETECT_CHESSBOARD_H
#define CBDETECT_CHESSBOARD_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

namespace cbdetect {

/**
 * @brief Chessboard structure representing a detected chessboard pattern
 */
class Chessboard {
public:
    // 2D grid storing corner indices (-1 for empty cells)
    std::vector<std::vector<int>> grid;
    
    // Energy/quality score of the chessboard
    float energy;
    
    Chessboard();
    Chessboard(int rows, int cols);
    
    // Basic operations
    void clear();
    void resize(int rows, int cols);
    int rows() const;
    int cols() const;
    bool empty() const;
    
    // Access operators
    std::vector<int>& operator[](int row);
    const std::vector<int>& operator[](int row) const;
    
    // Get/set corner index at position
    int getCornerIndex(int row, int col) const;
    void setCornerIndex(int row, int col, int corner_idx);
    
    // Check if position is valid and occupied
    bool isValidPosition(int row, int col) const;
    bool isOccupied(int row, int col) const;
    
    // Get all corner indices in the chessboard
    std::vector<int> getAllCornerIndices() const;
    
    // Get chessboard size (number of corners)
    int getCornerCount() const;
    
    // Check overlap with another chessboard
    bool hasOverlap(const Chessboard& other) const;
    
    // Grow chessboard in specified direction
    // direction: 0=right, 1=left, 2=down, 3=up
    Chessboard grow(int direction) const;
    
    // Copy constructor and assignment
    Chessboard(const Chessboard& other);
    Chessboard& operator=(const Chessboard& other);
    
    // Move constructor and assignment
    Chessboard(Chessboard&& other) noexcept;
    Chessboard& operator=(Chessboard&& other) noexcept;
};

/**
 * @brief Collection of detected chessboards
 */
class Chessboards {
public:
    std::vector<std::shared_ptr<Chessboard>> chessboards;
    
    void clear();
    size_t size() const;
    bool empty() const;
    
    // Access operators
    std::shared_ptr<Chessboard> operator[](size_t index);
    const std::shared_ptr<Chessboard> operator[](size_t index) const;
    
    // Add chessboard
    void push_back(std::shared_ptr<Chessboard> chessboard);
    void push_back(const Chessboard& chessboard);
    
    // Remove chessboard
    void erase(size_t index);
    
    // Find overlapping chessboards
    std::vector<size_t> findOverlapping(const Chessboard& chessboard) const;
    
    // Remove overlapping chessboards with higher energy
    void removeWorseOverlapping(const Chessboard& new_chessboard);
    
    // Filter by energy threshold
    void filterByEnergy(float threshold);
    
    // Iterator support
    std::vector<std::shared_ptr<Chessboard>>::iterator begin();
    std::vector<std::shared_ptr<Chessboard>>::iterator end();
    std::vector<std::shared_ptr<Chessboard>>::const_iterator begin() const;
    std::vector<std::shared_ptr<Chessboard>>::const_iterator end() const;
};

} // namespace cbdetect

#endif // CBDETECT_CHESSBOARD_H 