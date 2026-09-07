#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "xbioclim/bioclim.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace xbioclim;

// ---------------------------------------------------------------------------
// Build the synthetic ClimateBlock matching docs/ROADMAP.md §1.1
//
//   tas[m]    = m+1   (0-based m → values 1..12)
//   tasmax[m] = m+2
//   tasmin[m] = m
//   pr[m]     = m+1
//
// All pixels identical; N=1 pixel for simplicity.
// ---------------------------------------------------------------------------
static ClimateBlock make_test_block(std::size_t N = 1) {
    ClimateBlock data;
    data.tas    = Array2D::from_shape({N, 12});
    data.tasmax = Array2D::from_shape({N, 12});
    data.tasmin = Array2D::from_shape({N, 12});
    data.pr     = Array2D::from_shape({N, 12});

    for (std::size_t p = 0; p < N; ++p) {
        for (std::size_t m = 0; m < 12; ++m) {
            data.tas   (p, m) = static_cast<float>(m + 1);
            data.tasmax(p, m) = static_cast<float>(m + 2);
            data.tasmin(p, m) = static_cast<float>(m);
            data.pr    (p, m) = static_cast<float>(m + 1);
        }
    }
    return data;
}

// Expected values from ROADMAP §1.3
TEST_CASE("BIO01 - Mean Annual Temperature", "[bioclim]") {
    auto bio = compute_bioclim(make_test_block());
    CHECK_THAT(bio.bio01(0), WithinAbs(6.5f, 1e-3f));
}

TEST_CASE("BIO02 - Mean Diurnal Range", "[bioclim]") {
    // diurnal = tasmax - tasmin = [2..2], mean = 2.0
    auto bio = compute_bioclim(make_test_block());
    CHECK_THAT(bio.bio02(0), WithinAbs(2.0f, 1e-3f));
}

TEST_CASE("BIO03 - Isothermality", "[bioclim]") {
    // 100 * BIO02 / BIO07 = 100 * 2 / 13 ≈ 15.3846
    auto bio = compute_bioclim(make_test_block());
    CHECK_THAT(bio.bio03(0), WithinAbs(200.0f / 13.0f, 1e-3f));
}

TEST_CASE("BIO04 - Temperature Seasonality", "[bioclim]") {
    // 100 * pop_sd(tas) = 100 * sqrt(143/12) ≈ 345.21
    const float expected = 100.0f * std::sqrt(143.0f / 12.0f);
    auto bio = compute_bioclim(make_test_block());
    CHECK_THAT(bio.bio04(0), WithinAbs(expected, 1e-1f));
}

TEST_CASE("BIO05 - Max Temperature of Warmest Month", "[bioclim]") {
    // max(tasmax) = max(2..13) = 13
    auto bio = compute_bioclim(make_test_block());
    CHECK_THAT(bio.bio05(0), WithinAbs(13.0f, 1e-3f));
}

TEST_CASE("BIO06 - Min Temperature of Coldest Month", "[bioclim]") {
    // min(tasmin) = min(0..11) = 0
    auto bio = compute_bioclim(make_test_block());
    CHECK_THAT(bio.bio06(0), WithinAbs(0.0f, 1e-3f));
}

TEST_CASE("BIO07 - Annual Temperature Range", "[bioclim]") {
    // BIO05 - BIO06 = 13 - 0 = 13
    auto bio = compute_bioclim(make_test_block());
    CHECK_THAT(bio.bio07(0), WithinAbs(13.0f, 1e-3f));
}

TEST_CASE("BIO08 - Mean Temp of Wettest Quarter", "[bioclim]") {
    // Wettest quarter: start index 9 (months 10,11,12) → mean tas = 11.0
    auto bio = compute_bioclim(make_test_block());
    CHECK_THAT(bio.bio08(0), WithinAbs(11.0f, 1e-3f));
}

TEST_CASE("BIO09 - Mean Temp of Driest Quarter", "[bioclim]") {
    // Driest quarter: start index 0 (months 1,2,3) → mean tas = 2.0
    auto bio = compute_bioclim(make_test_block());
    CHECK_THAT(bio.bio09(0), WithinAbs(2.0f, 1e-3f));
}

TEST_CASE("BIO10 - Mean Temp of Warmest Quarter", "[bioclim]") {
    // Warmest quarter: start index 9 → mean tas = 11.0
    auto bio = compute_bioclim(make_test_block());
    CHECK_THAT(bio.bio10(0), WithinAbs(11.0f, 1e-3f));
}

TEST_CASE("BIO11 - Mean Temp of Coldest Quarter", "[bioclim]") {
    // Coldest quarter: start index 0 → mean tas = 2.0
    auto bio = compute_bioclim(make_test_block());
    CHECK_THAT(bio.bio11(0), WithinAbs(2.0f, 1e-3f));
}

TEST_CASE("BIO12 - Annual Precipitation", "[bioclim]") {
    // sum(1..12) = 78.0
    auto bio = compute_bioclim(make_test_block());
    CHECK_THAT(bio.bio12(0), WithinAbs(78.0f, 1e-3f));
}

TEST_CASE("BIO13 - Precipitation of Wettest Month", "[bioclim]") {
    // max(pr) = 12
    auto bio = compute_bioclim(make_test_block());
    CHECK_THAT(bio.bio13(0), WithinAbs(12.0f, 1e-3f));
}

TEST_CASE("BIO14 - Precipitation of Driest Month", "[bioclim]") {
    // min(pr) = 1
    auto bio = compute_bioclim(make_test_block());
    CHECK_THAT(bio.bio14(0), WithinAbs(1.0f, 1e-3f));
}

TEST_CASE("BIO15 - Precipitation Seasonality", "[bioclim]") {
    // 100 * pop_sd(pr) / mean(pr) = 100 * sqrt(143/12) / 6.5
    const float expected = 100.0f * std::sqrt(143.0f / 12.0f) / 6.5f;
    auto bio = compute_bioclim(make_test_block());
    CHECK_THAT(bio.bio15(0), WithinAbs(expected, 1e-1f));
}

TEST_CASE("BIO16 - Precipitation of Wettest Quarter", "[bioclim]") {
    // Wettest quarter by pr: start 9 → sum pr(10,11,12) = 33.0
    auto bio = compute_bioclim(make_test_block());
    CHECK_THAT(bio.bio16(0), WithinAbs(33.0f, 1e-3f));
}

TEST_CASE("BIO17 - Precipitation of Driest Quarter", "[bioclim]") {
    // Driest quarter by pr: start 0 → sum pr(1,2,3) = 6.0
    auto bio = compute_bioclim(make_test_block());
    CHECK_THAT(bio.bio17(0), WithinAbs(6.0f, 1e-3f));
}

TEST_CASE("BIO18 - Precipitation of Warmest Quarter", "[bioclim]") {
    // Warmest quarter by tas: start 9 → sum pr(10,11,12) = 33.0
    auto bio = compute_bioclim(make_test_block());
    CHECK_THAT(bio.bio18(0), WithinAbs(33.0f, 1e-3f));
}

TEST_CASE("BIO19 - Precipitation of Coldest Quarter", "[bioclim]") {
    // Coldest quarter by tas: start 0 → sum pr(1,2,3) = 6.0
    auto bio = compute_bioclim(make_test_block());
    CHECK_THAT(bio.bio19(0), WithinAbs(6.0f, 1e-3f));
}

TEST_CASE("BIO03 is NoData when BIO07 is zero", "[bioclim]") {
    // Build a block where tasmax == tasmin everywhere
    ClimateBlock data;
    data.tas    = xt::ones<float>({std::size_t(1), std::size_t(12)});
    data.tasmax = xt::ones<float>({std::size_t(1), std::size_t(12)});
    data.tasmin = xt::ones<float>({std::size_t(1), std::size_t(12)});
    data.pr     = xt::ones<float>({std::size_t(1), std::size_t(12)});
    auto bio = compute_bioclim(data);
    // BIO07 == 0, so BIO03 must be NaN
    CHECK(std::isnan(bio.bio03(0)));
}

TEST_CASE("BIO15 is NoData when mean precipitation is zero", "[bioclim]") {
    ClimateBlock data;
    data.tas    = xt::ones<float>({std::size_t(1), std::size_t(12)});
    data.tasmax = xt::ones<float>({std::size_t(1), std::size_t(12)});
    data.tasmin = xt::zeros<float>({std::size_t(1), std::size_t(12)});
    data.pr     = xt::zeros<float>({std::size_t(1), std::size_t(12)});
    auto bio = compute_bioclim(data);
    CHECK(std::isnan(bio.bio15(0)));
}

TEST_CASE("compute_bioclim handles multiple pixels", "[bioclim]") {
    // All pixels identical → all results identical
    const std::size_t N = 100;
    auto bio = compute_bioclim(make_test_block(N));
    for (std::size_t p = 0; p < N; ++p) {
        CHECK_THAT(bio.bio01(p), WithinAbs(6.5f, 1e-3f));
        CHECK_THAT(bio.bio12(p), WithinAbs(78.0f, 1e-3f));
    }
}

TEST_CASE("NaN inputs propagate through BIO variables", "[bioclim]") {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    ClimateBlock data;
    data.tas    = Array2D::from_shape({std::size_t(1), std::size_t(12)});
    data.tasmax = Array2D::from_shape({std::size_t(1), std::size_t(12)});
    data.tasmin = Array2D::from_shape({std::size_t(1), std::size_t(12)});
    data.pr     = Array2D::from_shape({std::size_t(1), std::size_t(12)});
    data.tas.fill(nan);
    data.tasmax.fill(nan);
    data.tasmin.fill(nan);
    data.pr.fill(nan);

    auto bio = compute_bioclim(data);
    // NaN inputs should produce NaN outputs for all BIO variables
    CHECK(std::isnan(bio.bio01(0)));
    CHECK(std::isnan(bio.bio02(0)));
    CHECK(std::isnan(bio.bio04(0)));
    CHECK(std::isnan(bio.bio05(0)));
    CHECK(std::isnan(bio.bio06(0)));
    CHECK(std::isnan(bio.bio07(0)));
    CHECK(std::isnan(bio.bio08(0)));
    CHECK(std::isnan(bio.bio09(0)));
    CHECK(std::isnan(bio.bio10(0)));
    CHECK(std::isnan(bio.bio11(0)));
    CHECK(std::isnan(bio.bio12(0)));
    CHECK(std::isnan(bio.bio13(0)));
    CHECK(std::isnan(bio.bio14(0)));
    CHECK(std::isnan(bio.bio16(0)));
    CHECK(std::isnan(bio.bio17(0)));
    CHECK(std::isnan(bio.bio18(0)));
    CHECK(std::isnan(bio.bio19(0)));
}

TEST_CASE("ClimateBlock rejects mismatched n_pixels", "[bioclim]") {
    ClimateBlock data;
    data.tas    = Array2D::from_shape({std::size_t(2), std::size_t(12)});
    data.tasmax = Array2D::from_shape({std::size_t(3), std::size_t(12)});  // mismatch
    data.tasmin = Array2D::from_shape({std::size_t(2), std::size_t(12)});
    data.pr     = Array2D::from_shape({std::size_t(2), std::size_t(12)});
    CHECK_THROWS_AS(compute_bioclim(data), std::invalid_argument);
}

TEST_CASE("All-identical monthly values yield zero seasonality", "[bioclim]") {
    // tas = tasmax = tasmin = pr = 5.0 everywhere → BIO04=0, BIO15=0
    ClimateBlock data;
    data.tas    = Array2D::from_shape({std::size_t(1), std::size_t(12)});
    data.tasmax = Array2D::from_shape({std::size_t(1), std::size_t(12)});
    data.tasmin = Array2D::from_shape({std::size_t(1), std::size_t(12)});
    data.pr     = Array2D::from_shape({std::size_t(1), std::size_t(12)});
    data.tas.fill(5.0f);
    data.tasmax.fill(5.0f);
    data.tasmin.fill(5.0f);
    data.pr.fill(5.0f);

    auto bio = compute_bioclim(data);
    // Temperature seasonality: 100 * stddev(5,5,...) = 0
    CHECK_THAT(bio.bio04(0), WithinAbs(0.0f, 1e-3f));
    // Precipitation seasonality: 100 * stddev/mean = 100 * 0/5 = 0
    CHECK_THAT(bio.bio15(0), WithinAbs(0.0f, 1e-3f));
}

// ---------------------------------------------------------------------------
// OpenMP offload correctness: results from the offload code path (activated
// when XBIOCLIM_USE_OPENMP_OFFLOAD is defined) must match the expected
// analytical values and handle NaN/edge cases identically to the CPU path.
// ---------------------------------------------------------------------------

#ifdef XBIOCLIM_USE_OPENMP_OFFLOAD

TEST_CASE("Offload: BIO01/BIO12 match expected values", "[bioclim][offload]") {
    auto bio = compute_bioclim(make_test_block());
    CHECK_THAT(bio.bio01(0), WithinAbs(6.5f, 1e-3f));
    CHECK_THAT(bio.bio12(0), WithinAbs(78.0f, 1e-3f));
}

TEST_CASE("Offload: quarter-based variables match expected values", "[bioclim][offload]") {
    auto bio = compute_bioclim(make_test_block());
    // Wettest/warmest quarter starts at month 9 → mean tas = 11, sum pr = 33
    CHECK_THAT(bio.bio08(0), WithinAbs(11.0f, 1e-3f));
    CHECK_THAT(bio.bio10(0), WithinAbs(11.0f, 1e-3f));
    CHECK_THAT(bio.bio16(0), WithinAbs(33.0f, 1e-3f));
    CHECK_THAT(bio.bio18(0), WithinAbs(33.0f, 1e-3f));
    // Driest/coldest quarter starts at month 0 → mean tas = 2, sum pr = 6
    CHECK_THAT(bio.bio09(0), WithinAbs(2.0f, 1e-3f));
    CHECK_THAT(bio.bio11(0), WithinAbs(2.0f, 1e-3f));
    CHECK_THAT(bio.bio17(0), WithinAbs(6.0f, 1e-3f));
    CHECK_THAT(bio.bio19(0), WithinAbs(6.0f, 1e-3f));
}

TEST_CASE("Offload: NaN inputs produce NaN outputs", "[bioclim][offload]") {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    ClimateBlock data;
    data.tas    = Array2D::from_shape({std::size_t(1), std::size_t(12)});
    data.tasmax = Array2D::from_shape({std::size_t(1), std::size_t(12)});
    data.tasmin = Array2D::from_shape({std::size_t(1), std::size_t(12)});
    data.pr     = Array2D::from_shape({std::size_t(1), std::size_t(12)});
    data.tas.fill(nan);
    data.tasmax.fill(nan);
    data.tasmin.fill(nan);
    data.pr.fill(nan);

    auto bio = compute_bioclim(data);
    // Quarter-based variables must be NaN when all inputs are NaN
    CHECK(std::isnan(bio.bio08(0)));
    CHECK(std::isnan(bio.bio09(0)));
    CHECK(std::isnan(bio.bio10(0)));
    CHECK(std::isnan(bio.bio11(0)));
    CHECK(std::isnan(bio.bio16(0)));
    CHECK(std::isnan(bio.bio17(0)));
    CHECK(std::isnan(bio.bio18(0)));
    CHECK(std::isnan(bio.bio19(0)));
}

TEST_CASE("Offload: multiple pixels yield consistent results", "[bioclim][offload]") {
    const std::size_t N = 1000;
    auto bio = compute_bioclim(make_test_block(N));
    for (std::size_t p = 0; p < N; ++p) {
        CHECK_THAT(bio.bio01(p), WithinAbs(6.5f, 1e-3f));
        CHECK_THAT(bio.bio08(p), WithinAbs(11.0f, 1e-3f));
        CHECK_THAT(bio.bio12(p), WithinAbs(78.0f, 1e-3f));
    }
}

TEST_CASE("Offload: BIO03 is NoData when BIO07 is zero", "[bioclim][offload]") {
    ClimateBlock data;
    data.tas    = xt::ones<float>({std::size_t(1), std::size_t(12)});
    data.tasmax = xt::ones<float>({std::size_t(1), std::size_t(12)});
    data.tasmin = xt::ones<float>({std::size_t(1), std::size_t(12)});
    data.pr     = xt::ones<float>({std::size_t(1), std::size_t(12)});
    auto bio = compute_bioclim(data);
    CHECK(std::isnan(bio.bio03(0)));
}

TEST_CASE("Offload: BIO15 is NoData when mean precipitation is zero", "[bioclim][offload]") {
    ClimateBlock data;
    data.tas    = xt::ones<float>({std::size_t(1), std::size_t(12)});
    data.tasmax = xt::ones<float>({std::size_t(1), std::size_t(12)});
    data.tasmin = xt::zeros<float>({std::size_t(1), std::size_t(12)});
    data.pr     = xt::zeros<float>({std::size_t(1), std::size_t(12)});
    auto bio = compute_bioclim(data);
    CHECK(std::isnan(bio.bio15(0)));
}

#endif // XBIOCLIM_USE_OPENMP_OFFLOAD
