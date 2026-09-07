#pragma once

#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xsort.hpp>

#include <cstddef>
#include <limits>

namespace xbioclim {

/// 2-D array: shape [N_pixels, 12]
using Array2D = xt::xtensor<float, 2>;

/// 1-D array: shape [N_pixels]
using Array1D = xt::xtensor<float, 1>;

/// Integer index array: shape [N_pixels]
using IndexArray = xt::xtensor<std::size_t, 1>;

/// Bundle of four input variable blocks
struct ClimateBlock {
    Array2D tas;    ///< Monthly mean temperature  [N_pixels, 12]
    Array2D tasmax; ///< Monthly max temperature   [N_pixels, 12]
    Array2D tasmin; ///< Monthly min temperature   [N_pixels, 12]
    Array2D pr;     ///< Monthly precipitation     [N_pixels, 12]
    std::size_t n_pixels() const { return tas.shape(0); }
};

/// Scale/offset parameters for CHELSA-style raw data decoding.
/// Decoded value = raw * scale + offset.
struct ScaleOffset {
    float scale  = 1.0f;
    float offset = 0.0f;
};

/// Bundle of 19 output BIO variable arrays
struct BioBlock {
    Array1D bio01, bio02, bio03, bio04, bio05,
            bio06, bio07, bio08, bio09, bio10,
            bio11, bio12, bio13, bio14, bio15,
            bio16, bio17, bio18, bio19;
};

// ---------------------------------------------------------------------------
// Core primitives (row-wise operations on Array2D → Array1D)
// ---------------------------------------------------------------------------

/// Row-wise mean: shape [N_pixels, 12] → [N_pixels]
Array1D row_mean(const Array2D& A);

/// Row-wise sum: shape [N_pixels, 12] → [N_pixels]
Array1D row_sum(const Array2D& A);

/// Row-wise maximum: shape [N_pixels, 12] → [N_pixels]
Array1D row_max(const Array2D& A);

/// Row-wise minimum: shape [N_pixels, 12] → [N_pixels]
Array1D row_min(const Array2D& A);

/// Row-wise population standard deviation (ddof=0, denominator N=12)
Array1D row_std(const Array2D& A);

/// Element-wise difference: A − B, both shape [N_pixels, 12]
Array2D elementwise_diff(const Array2D& A, const Array2D& B);

/// 0-based start index of the rolling 3-month window with maximum circular sum
IndexArray rolling_quarter_argmax(const Array2D& A);

/// 0-based start index of the rolling 3-month window with minimum circular sum
IndexArray rolling_quarter_argmin(const Array2D& A);

/// Mean of the circular 3-month window starting at each pixel's start index
Array1D quarter_mean(const Array2D& A, const IndexArray& starts);

/// Sum of the circular 3-month window starting at each pixel's start index
Array1D quarter_sum(const Array2D& A, const IndexArray& starts);

} // namespace xbioclim
