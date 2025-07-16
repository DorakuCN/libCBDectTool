#include "cbdetect/chessboard.h"
#include <algorithm>
#include <set>

namespace cbdetect {

// Chessboard class implementation

Chessboard::Chessboard() : energy(0.0f) {
}

Chessboard::Chessboard(int rows, int cols) : energy(0.0f) {
    resize(rows, cols);
}

void Chessboard::clear() {
    grid.clear();
    energy = 0.0f;
}

void Chessboard::resize(int rows, int cols) {
    grid.resize(rows);
    for (int i = 0; i < rows; ++i) {
        grid[i].resize(cols, -1);  // -1 indicates empty cell
    }
}

int Chessboard::rows() const {
    return static_cast<int>(grid.size());
}

int Chessboard::cols() const {
    return grid.empty() ? 0 : static_cast<int>(grid[0].size());
}

bool Chessboard::empty() const {
    return grid.empty();
}

std::vector<int>& Chessboard::operator[](int row) {
    return grid[row];
}

const std::vector<int>& Chessboard::operator[](int row) const {
    return grid[row];
}

int Chessboard::getCornerIndex(int row, int col) const {
    if (!isValidPosition(row, col)) {
        return -1;
    }
    return grid[row][col];
}

void Chessboard::setCornerIndex(int row, int col, int corner_idx) {
    if (isValidPosition(row, col)) {
        grid[row][col] = corner_idx;
    }
}

bool Chessboard::isValidPosition(int row, int col) const {
    return row >= 0 && row < rows() && col >= 0 && col < cols();
}

bool Chessboard::isOccupied(int row, int col) const {
    return isValidPosition(row, col) && grid[row][col] >= 0;
}

std::vector<int> Chessboard::getAllCornerIndices() const {
    std::vector<int> indices;
    
    for (const auto& row : grid) {
        for (int idx : row) {
            if (idx >= 0) {
                indices.push_back(idx);
            }
        }
    }
    
    return indices;
}

int Chessboard::getCornerCount() const {
    int count = 0;
    
    for (const auto& row : grid) {
        for (int idx : row) {
            if (idx >= 0) {
                count++;
            }
        }
    }
    
    return count;
}

bool Chessboard::hasOverlap(const Chessboard& other) const {
    std::vector<int> my_indices = getAllCornerIndices();
    std::vector<int> other_indices = other.getAllCornerIndices();
    
    std::set<int> my_set(my_indices.begin(), my_indices.end());
    
    for (int idx : other_indices) {
        if (my_set.find(idx) != my_set.end()) {
            return true;
        }
    }
    
    return false;
}

Chessboard Chessboard::grow(int direction) const {
    Chessboard result(*this);
    
    switch (direction) {
        case 0: { // Grow right
            result.resize(rows(), cols() + 1);
            // Copy existing data
            for (int r = 0; r < rows(); ++r) {
                for (int c = 0; c < cols(); ++c) {
                    result[r][c] = grid[r][c];
                }
            }
            break;
        }
        case 1: { // Grow left
            result.resize(rows(), cols() + 1);
            // Shift existing data right
            for (int r = 0; r < rows(); ++r) {
                for (int c = 0; c < cols(); ++c) {
                    result[r][c + 1] = grid[r][c];
                }
            }
            break;
        }
        case 2: { // Grow down
            result.resize(rows() + 1, cols());
            // Copy existing data
            for (int r = 0; r < rows(); ++r) {
                for (int c = 0; c < cols(); ++c) {
                    result[r][c] = grid[r][c];
                }
            }
            break;
        }
        case 3: { // Grow up
            result.resize(rows() + 1, cols());
            // Shift existing data down
            for (int r = 0; r < rows(); ++r) {
                for (int c = 0; c < cols(); ++c) {
                    result[r + 1][c] = grid[r][c];
                }
            }
            break;
        }
    }
    
    return result;
}

// Copy constructor
Chessboard::Chessboard(const Chessboard& other) 
    : grid(other.grid), energy(other.energy) {
}

// Assignment operator
Chessboard& Chessboard::operator=(const Chessboard& other) {
    if (this != &other) {
        grid = other.grid;
        energy = other.energy;
    }
    return *this;
}

// Move constructor
Chessboard::Chessboard(Chessboard&& other) noexcept 
    : grid(std::move(other.grid)), energy(other.energy) {
    other.energy = 0.0f;
}

// Move assignment operator
Chessboard& Chessboard::operator=(Chessboard&& other) noexcept {
    if (this != &other) {
        grid = std::move(other.grid);
        energy = other.energy;
        other.energy = 0.0f;
    }
    return *this;
}

// Chessboards class implementation

void Chessboards::clear() {
    chessboards.clear();
}

size_t Chessboards::size() const {
    return chessboards.size();
}

bool Chessboards::empty() const {
    return chessboards.empty();
}

std::shared_ptr<Chessboard> Chessboards::operator[](size_t index) {
    return chessboards[index];
}

const std::shared_ptr<Chessboard> Chessboards::operator[](size_t index) const {
    return chessboards[index];
}

void Chessboards::push_back(std::shared_ptr<Chessboard> chessboard) {
    chessboards.push_back(chessboard);
}

void Chessboards::push_back(const Chessboard& chessboard) {
    chessboards.push_back(std::make_shared<Chessboard>(chessboard));
}

void Chessboards::erase(size_t index) {
    if (index < chessboards.size()) {
        chessboards.erase(chessboards.begin() + index);
    }
}

std::vector<size_t> Chessboards::findOverlapping(const Chessboard& chessboard) const {
    std::vector<size_t> overlapping_indices;
    
    for (size_t i = 0; i < chessboards.size(); ++i) {
        if (chessboards[i]->hasOverlap(chessboard)) {
            overlapping_indices.push_back(i);
        }
    }
    
    return overlapping_indices;
}

void Chessboards::removeWorseOverlapping(const Chessboard& new_chessboard) {
    std::vector<size_t> overlapping = findOverlapping(new_chessboard);
    
    // Remove from back to front to maintain valid indices
    std::sort(overlapping.rbegin(), overlapping.rend());
    
    for (size_t idx : overlapping) {
        if (chessboards[idx]->energy > new_chessboard.energy) {
            erase(idx);
        }
    }
}

void Chessboards::filterByEnergy(float threshold) {
    chessboards.erase(
        std::remove_if(chessboards.begin(), chessboards.end(),
                      [threshold](const std::shared_ptr<Chessboard>& cb) {
                          return cb->energy > threshold;
                      }),
        chessboards.end()
    );
}

std::vector<std::shared_ptr<Chessboard>>::iterator Chessboards::begin() {
    return chessboards.begin();
}

std::vector<std::shared_ptr<Chessboard>>::iterator Chessboards::end() {
    return chessboards.end();
}

std::vector<std::shared_ptr<Chessboard>>::const_iterator Chessboards::begin() const {
    return chessboards.begin();
}

std::vector<std::shared_ptr<Chessboard>>::const_iterator Chessboards::end() const {
    return chessboards.end();
}

} // namespace cbdetect 