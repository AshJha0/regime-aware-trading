#ifndef REGIME_MATRIX_HPP
#define REGIME_MATRIX_HPP

/// \file matrix.hpp
/// \brief Minimal dense row-major matrix used throughout the regime library.
///
/// The library needs nothing more than a (T x D) container of doubles with
/// element access, so a tiny struct keeps the dependency footprint at zero
/// and makes the summation order of every algorithm explicit.

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace regime {

/// Dense row-major matrix of doubles.
struct Matrix {
    std::size_t rows{0};        ///< Number of rows.
    std::size_t cols{0};        ///< Number of columns.
    std::vector<double> data;   ///< Row-major storage, size rows*cols.

    Matrix() = default;

    /// Construct a rows x cols matrix filled with \p fill.
    Matrix(std::size_t r, std::size_t c, double fill = 0.0)
        : rows(r), cols(c), data(r * c, fill) {}

    /// Mutable element access (no bounds check; hot path).
    double& operator()(std::size_t i, std::size_t j) { return data[i * cols + j]; }

    /// Const element access.
    double operator()(std::size_t i, std::size_t j) const { return data[i * cols + j]; }

    /// True when the matrix holds no elements.
    bool empty() const { return rows == 0 || cols == 0; }
};

/// Wrap a 1-D series as a (T x 1) observation matrix.
inline Matrix to_matrix(const std::vector<double>& series) {
    Matrix m(series.size(), 1);
    for (std::size_t t = 0; t < series.size(); ++t) m(t, 0) = series[t];
    return m;
}

}  // namespace regime

#endif  // REGIME_MATRIX_HPP
