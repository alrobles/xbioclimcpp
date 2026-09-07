#include "xbioclim/primitives.hpp"

#include <cmath>
#include <cstddef>
#include <limits>

#if defined(XBIOCLIM_USE_OPENMP) || defined(XBIOCLIM_USE_OPENMP_OFFLOAD)
#include <omp.h>
#endif

namespace xbioclim {

Array1D row_mean(const Array2D& A) {
    return xt::mean(A, /*axis=*/1);
}

Array1D row_sum(const Array2D& A) {
    return xt::sum(A, /*axis=*/1);
}

Array1D row_max(const Array2D& A) {
    return xt::amax(A, /*axis=*/1);
}

Array1D row_min(const Array2D& A) {
    return xt::amin(A, /*axis=*/1);
}

Array1D row_std(const Array2D& A) {
    // Population standard deviation: ddof=0, denominator N=12 (xtensor default)
    // Use initializer-list form {1} instead of bare integer 1 for axis argument.
    return xt::stddev(A, {1});
}

Array2D elementwise_diff(const Array2D& A, const Array2D& B) {
    return A - B;
}

IndexArray rolling_quarter_argmax(const Array2D& A) {
    const std::size_t N = A.shape(0);
    IndexArray result = xt::zeros<std::size_t>({N});

#if defined(XBIOCLIM_USE_OPENMP_OFFLOAD)
    {
        const float* __restrict__ A_ptr = A.data();
        std::size_t* __restrict__ res_ptr = result.data();
        const std::size_t total = N * 12;
        static constexpr std::size_t SENTINEL = std::numeric_limits<std::size_t>::max();
        static constexpr float FLOAT_LOWEST = std::numeric_limits<float>::lowest();
#pragma omp target teams distribute parallel for \
        map(to: A_ptr[0:total]) map(from: res_ptr[0:N])
        for (std::size_t p = 0; p < N; ++p) {
            float best = FLOAT_LOWEST;
            std::size_t bi = 0;
            bool has_nan = false;
            for (std::size_t i = 0; i < 12; ++i) {
                float a = A_ptr[p * 12 + i];
                float b = A_ptr[p * 12 + (i + 1) % 12];
                float c = A_ptr[p * 12 + (i + 2) % 12];
                if ((a != a) || (b != b) || (c != c)) {
                    has_nan = true;
                    break;
                }
                float s = a + b + c;
                if (s > best) {
                    best = s;
                    bi = i;
                }
            }
            res_ptr[p] = has_nan ? SENTINEL : bi;
        }
    }
#elif defined(XBIOCLIM_USE_OPENMP)
#pragma omp parallel for schedule(static)
    for (std::size_t p = 0; p < N; ++p) {
        float best = std::numeric_limits<float>::lowest();
        std::size_t bi = 0;
        bool has_nan = false;
        for (std::size_t i = 0; i < 12; ++i) {
            float a = A(p, i);
            float b = A(p, (i + 1) % 12);
            float c = A(p, (i + 2) % 12);
            if (std::isnan(a) || std::isnan(b) || std::isnan(c)) {
                has_nan = true;
                break;
            }
            float s = a + b + c;
            if (s > best) {
                best = s;
                bi = i;
            }
        }
        result(p) = has_nan ? std::numeric_limits<std::size_t>::max() : bi;
    }
#else
    for (std::size_t p = 0; p < N; ++p) {
        float best = std::numeric_limits<float>::lowest();
        std::size_t bi = 0;
        bool has_nan = false;
        for (std::size_t i = 0; i < 12; ++i) {
            float a = A(p, i);
            float b = A(p, (i + 1) % 12);
            float c = A(p, (i + 2) % 12);
            if (std::isnan(a) || std::isnan(b) || std::isnan(c)) {
                has_nan = true;
                break;
            }
            float s = a + b + c;
            if (s > best) {
                best = s;
                bi = i;
            }
        }
        result(p) = has_nan ? std::numeric_limits<std::size_t>::max() : bi;
    }
#endif
    return result;
}

IndexArray rolling_quarter_argmin(const Array2D& A) {
    const std::size_t N = A.shape(0);
    IndexArray result = xt::zeros<std::size_t>({N});

#if defined(XBIOCLIM_USE_OPENMP_OFFLOAD)
    {
        const float* __restrict__ A_ptr = A.data();
        std::size_t* __restrict__ res_ptr = result.data();
        const std::size_t total = N * 12;
        static constexpr std::size_t SENTINEL = std::numeric_limits<std::size_t>::max();
        static constexpr float FLOAT_MAX = std::numeric_limits<float>::max();
#pragma omp target teams distribute parallel for \
        map(to: A_ptr[0:total]) map(from: res_ptr[0:N])
        for (std::size_t p = 0; p < N; ++p) {
            float best = FLOAT_MAX;
            std::size_t bi = 0;
            bool has_nan = false;
            for (std::size_t i = 0; i < 12; ++i) {
                float a = A_ptr[p * 12 + i];
                float b = A_ptr[p * 12 + (i + 1) % 12];
                float c = A_ptr[p * 12 + (i + 2) % 12];
                if ((a != a) || (b != b) || (c != c)) {
                    has_nan = true;
                    break;
                }
                float s = a + b + c;
                if (s < best) {
                    best = s;
                    bi = i;
                }
            }
            res_ptr[p] = has_nan ? SENTINEL : bi;
        }
    }
#elif defined(XBIOCLIM_USE_OPENMP)
#pragma omp parallel for schedule(static)
    for (std::size_t p = 0; p < N; ++p) {
        float best = std::numeric_limits<float>::max();
        std::size_t bi = 0;
        bool has_nan = false;
        for (std::size_t i = 0; i < 12; ++i) {
            float a = A(p, i);
            float b = A(p, (i + 1) % 12);
            float c = A(p, (i + 2) % 12);
            if (std::isnan(a) || std::isnan(b) || std::isnan(c)) {
                has_nan = true;
                break;
            }
            float s = a + b + c;
            if (s < best) {
                best = s;
                bi = i;
            }
        }
        result(p) = has_nan ? std::numeric_limits<std::size_t>::max() : bi;
    }
#else
    for (std::size_t p = 0; p < N; ++p) {
        float best = std::numeric_limits<float>::max();
        std::size_t bi = 0;
        bool has_nan = false;
        for (std::size_t i = 0; i < 12; ++i) {
            float a = A(p, i);
            float b = A(p, (i + 1) % 12);
            float c = A(p, (i + 2) % 12);
            if (std::isnan(a) || std::isnan(b) || std::isnan(c)) {
                has_nan = true;
                break;
            }
            float s = a + b + c;
            if (s < best) {
                best = s;
                bi = i;
            }
        }
        result(p) = has_nan ? std::numeric_limits<std::size_t>::max() : bi;
    }
#endif
    return result;
}

Array1D quarter_mean(const Array2D& A, const IndexArray& starts) {
    const std::size_t N = A.shape(0);
    Array1D result = xt::zeros<float>({N});
    static constexpr std::size_t SENTINEL = std::numeric_limits<std::size_t>::max();

#if defined(XBIOCLIM_USE_OPENMP_OFFLOAD)
    {
        const float* __restrict__ A_ptr = A.data();
        const std::size_t* __restrict__ starts_ptr = starts.data();
        float* __restrict__ res_ptr = result.data();
        const std::size_t total = N * 12;
        static constexpr float NAN_VAL = std::numeric_limits<float>::quiet_NaN();
#pragma omp target teams distribute parallel for \
        map(to: A_ptr[0:total], starts_ptr[0:N]) map(from: res_ptr[0:N])
        for (std::size_t p = 0; p < N; ++p) {
            const std::size_t i = starts_ptr[p];
            if (i == SENTINEL) {
                res_ptr[p] = NAN_VAL;
            } else {
                res_ptr[p] = (A_ptr[p * 12 + i % 12]
                            + A_ptr[p * 12 + (i + 1) % 12]
                            + A_ptr[p * 12 + (i + 2) % 12]) / 3.0f;
            }
        }
    }
#elif defined(XBIOCLIM_USE_OPENMP)
#pragma omp parallel for schedule(static)
    for (std::size_t p = 0; p < N; ++p) {
        const std::size_t i = starts(p);
        if (i == SENTINEL) {
            result(p) = std::numeric_limits<float>::quiet_NaN();
        } else {
            result(p) = (A(p, i % 12)
                       + A(p, (i + 1) % 12)
                       + A(p, (i + 2) % 12)) / 3.0f;
        }
    }
#else
    for (std::size_t p = 0; p < N; ++p) {
        const std::size_t i = starts(p);
        if (i == SENTINEL) {
            result(p) = std::numeric_limits<float>::quiet_NaN();
        } else {
            result(p) = (A(p, i % 12)
                       + A(p, (i + 1) % 12)
                       + A(p, (i + 2) % 12)) / 3.0f;
        }
    }
#endif
    return result;
}

Array1D quarter_sum(const Array2D& A, const IndexArray& starts) {
    const std::size_t N = A.shape(0);
    Array1D result = xt::zeros<float>({N});
    static constexpr std::size_t SENTINEL = std::numeric_limits<std::size_t>::max();

#if defined(XBIOCLIM_USE_OPENMP_OFFLOAD)
    {
        const float* __restrict__ A_ptr = A.data();
        const std::size_t* __restrict__ starts_ptr = starts.data();
        float* __restrict__ res_ptr = result.data();
        const std::size_t total = N * 12;
        static constexpr float NAN_VAL = std::numeric_limits<float>::quiet_NaN();
#pragma omp target teams distribute parallel for \
        map(to: A_ptr[0:total], starts_ptr[0:N]) map(from: res_ptr[0:N])
        for (std::size_t p = 0; p < N; ++p) {
            const std::size_t i = starts_ptr[p];
            if (i == SENTINEL) {
                res_ptr[p] = NAN_VAL;
            } else {
                res_ptr[p] = (A_ptr[p * 12 + i % 12]
                            + A_ptr[p * 12 + (i + 1) % 12]
                            + A_ptr[p * 12 + (i + 2) % 12]);
            }
        }
    }
#elif defined(XBIOCLIM_USE_OPENMP)
#pragma omp parallel for schedule(static)
    for (std::size_t p = 0; p < N; ++p) {
        const std::size_t i = starts(p);
        if (i == SENTINEL) {
            result(p) = std::numeric_limits<float>::quiet_NaN();
        } else {
            result(p) = (A(p, i % 12)
                       + A(p, (i + 1) % 12)
                       + A(p, (i + 2) % 12));
        }
    }
#else
    for (std::size_t p = 0; p < N; ++p) {
        const std::size_t i = starts(p);
        if (i == SENTINEL) {
            result(p) = std::numeric_limits<float>::quiet_NaN();
        } else {
            result(p) = (A(p, i % 12)
                       + A(p, (i + 1) % 12)
                       + A(p, (i + 2) % 12));
        }
    }
#endif
    return result;
}

} // namespace xbioclim
