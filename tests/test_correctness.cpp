// tests/test_correctness.cpp
//
// Correctness test suite — CPU vs. GPU/Offload paths for compute_bioclim
// =======================================================================
//
// Strategy
// --------
// compute_bioclim() is the single public entry point.  The backend it uses
// (scalar CPU, OpenMP threads, OpenMP 4.5 target offload, CUDA) is selected
// at *build time* via CMake options and the resulting pre-processor macros.
//
// Every test case in this file runs against the *active* backend that the
// current binary was compiled for.  To compare backends against each other,
// the CI builds and runs this suite once per configuration.  Any deviation
// from the shared expected values signals a regression in that backend.
//
// Tolerance
// ---------
// All BIO variables are computed in single precision (float).  The default
// tolerance used throughout is 1e-4 (absolute).  BIO04 and BIO15 involve
// sqrt() and division, so they use a slightly wider tolerance of 1e-2.
//
// Sections
// --------
//  1. Active backend identification
//  2. All 19 BIO variables — correctness against known expected values
//     (includes consolidated deviation report per variable)
//  3. NaN propagation — all inputs NaN → all 19 outputs NaN
//  4. Partial-NaN regression — mixed-pixel block (NaN + valid side by side)
//  5. NoData edge cases: BIO03 when BIO07=0, BIO15 when mean(pr)=0
//  6. Uniform-value edge cases (zero seasonality, isothermality guards)
//  7. Negative temperature / zero precipitation edge cases
//  8. OpenMP-offload backend (skipped when not enabled)
//  9. CUDA backend (skipped when not enabled)

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "xbioclim/bioclim.hpp"

#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace xbioclim;

// ---------------------------------------------------------------------------
// Tolerances
// ---------------------------------------------------------------------------
static constexpr float kTolDefault = 1e-4f; // most BIO variables
static constexpr float kTolSqrt    = 1e-2f; // BIO04, BIO15 (involve sqrt)

// ---------------------------------------------------------------------------
// Section 1 — Active backend identification
// ---------------------------------------------------------------------------

static std::string active_backend() {
#if defined(XBIOCLIM_USE_CUDA)
    return "CUDA";
#elif defined(XBIOCLIM_USE_OPENMP_OFFLOAD)
    return "OpenMP-offload";
#elif defined(XBIOCLIM_USE_OPENMP)
    return "OpenMP-CPU";
#else
    return "scalar-CPU";
#endif
}

// ---------------------------------------------------------------------------
// Helper: standard "ramp" ClimateBlock
//   tas[m]    = m+1  (values 1..12, 0-based month index m)
//   tasmax[m] = m+2  (values 2..13)
//   tasmin[m] = m    (values 0..11)
//   pr[m]     = m+1  (values 1..12)
// N_pixels copies of the same data are placed in rows 0..N-1.
// ---------------------------------------------------------------------------
static ClimateBlock make_ramp_block(std::size_t N = 1) {
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

// ---------------------------------------------------------------------------
// Helper: collect deviations above tolerance and format them as strings.
// Returns "" when the value is within tolerance.
// Handles the NaN-equals-NaN convention used in NoData checks.
// ---------------------------------------------------------------------------
static std::string deviation_message(const char* name,
                                     float actual,
                                     float expected,
                                     float tol = kTolDefault) {
    // Both NaN → correct NoData propagation, no deviation
    if (std::isnan(expected) && std::isnan(actual)) {
        return "";
    }
    // Unexpected NaN or numerical deviation
    if (std::isnan(actual) || std::isnan(expected) ||
        std::abs(actual - expected) > tol) {
        std::ostringstream ss;
        ss << name << ": expected=" << expected
           << " actual=" << actual
           << " |diff|=" << (std::isnan(actual) || std::isnan(expected)
                                 ? std::numeric_limits<float>::quiet_NaN()
                                 : std::abs(actual - expected))
           << " tol=" << tol;
        return ss.str();
    }
    return "";
}

// ---------------------------------------------------------------------------
// Pre-computed expected values for the ramp block (single pixel)
// ---------------------------------------------------------------------------
namespace expected_ramp {
    // BIO01: mean(1..12) = 78/12 = 6.5
    static constexpr float bio01 = 6.5f;
    // BIO02: mean(tasmax - tasmin) = mean(2..2) = 2.0
    static constexpr float bio02 = 2.0f;
    // BIO03: 100 * BIO02 / BIO07 = 100 * 2 / 13
    static const float bio03 = 200.0f / 13.0f;
    // BIO04: 100 * pop_SD(1..12) = 100 * sqrt(143/12)
    static const float bio04 = 100.0f * std::sqrt(143.0f / 12.0f);
    // BIO05: max(tasmax) = max(2..13) = 13
    static constexpr float bio05 = 13.0f;
    // BIO06: min(tasmin) = min(0..11) = 0
    static constexpr float bio06 = 0.0f;
    // BIO07: BIO05 - BIO06 = 13
    static constexpr float bio07 = 13.0f;
    // BIO08: mean tas of wettest quarter (start 9 → months 10,11,12) = 11.0
    static constexpr float bio08 = 11.0f;
    // BIO09: mean tas of driest quarter (start 0 → months 1,2,3) = 2.0
    static constexpr float bio09 = 2.0f;
    // BIO10: mean tas of warmest quarter (start 9) = 11.0
    static constexpr float bio10 = 11.0f;
    // BIO11: mean tas of coldest quarter (start 0) = 2.0
    static constexpr float bio11 = 2.0f;
    // BIO12: sum(1..12) = 78
    static constexpr float bio12 = 78.0f;
    // BIO13: max(pr) = 12
    static constexpr float bio13 = 12.0f;
    // BIO14: min(pr) = 1
    static constexpr float bio14 = 1.0f;
    // BIO15: 100 * pop_SD(1..12) / mean(1..12) = bio04 / 6.5
    static const float bio15 = 100.0f * std::sqrt(143.0f / 12.0f) / 6.5f;
    // BIO16: sum pr of wettest quarter (start 9 → months 10,11,12) = 10+11+12 = 33.0
    static constexpr float bio16 = 33.0f;
    // BIO17: sum pr of driest quarter (start 0 → months 1,2,3) = 1+2+3 = 6.0
    static constexpr float bio17 = 6.0f;
    // BIO18: sum pr of warmest quarter (start 9) = 10+11+12 = 33.0
    static constexpr float bio18 = 33.0f;
    // BIO19: sum pr of coldest quarter (start 0) = 1+2+3 = 6.0
    static constexpr float bio19 = 6.0f;
} // namespace expected_ramp

// ===========================================================================
// Section 2 — All 19 BIO variables: correctness with deviation report
// ===========================================================================

TEST_CASE("correctness: all 19 BIO variables on active backend [ramp block]",
          "[correctness][backend]") {
    INFO("Active backend: " << active_backend());
    const auto bio = compute_bioclim(make_ramp_block());

    // Collect any deviations above tolerance and report them together so that
    // CI logs show *all* failing variables, not just the first.
    std::vector<std::string> failures;
    auto chk = [&](const char* name, float actual, float expected,
                   float tol = kTolDefault) {
        auto msg = deviation_message(name, actual, expected, tol);
        if (!msg.empty()) {
            failures.push_back(msg);
        }
    };

    chk("BIO01", bio.bio01(0), expected_ramp::bio01);
    chk("BIO02", bio.bio02(0), expected_ramp::bio02);
    chk("BIO03", bio.bio03(0), expected_ramp::bio03);
    chk("BIO04", bio.bio04(0), expected_ramp::bio04, kTolSqrt);
    chk("BIO05", bio.bio05(0), expected_ramp::bio05);
    chk("BIO06", bio.bio06(0), expected_ramp::bio06);
    chk("BIO07", bio.bio07(0), expected_ramp::bio07);
    chk("BIO08", bio.bio08(0), expected_ramp::bio08);
    chk("BIO09", bio.bio09(0), expected_ramp::bio09);
    chk("BIO10", bio.bio10(0), expected_ramp::bio10);
    chk("BIO11", bio.bio11(0), expected_ramp::bio11);
    chk("BIO12", bio.bio12(0), expected_ramp::bio12);
    chk("BIO13", bio.bio13(0), expected_ramp::bio13);
    chk("BIO14", bio.bio14(0), expected_ramp::bio14);
    chk("BIO15", bio.bio15(0), expected_ramp::bio15, kTolSqrt);
    chk("BIO16", bio.bio16(0), expected_ramp::bio16);
    chk("BIO17", bio.bio17(0), expected_ramp::bio17);
    chk("BIO18", bio.bio18(0), expected_ramp::bio18);
    chk("BIO19", bio.bio19(0), expected_ramp::bio19);

    // Report the count and details of any deviations before failing.
    INFO("Deviations above tolerance: " << failures.size());
    for (const auto& f : failures) {
        INFO(f);
    }
    REQUIRE(failures.empty());
}

// Individual per-variable correctness tests — these also serve as regression
// anchors when only specific variables are re-implemented on a new backend.

TEST_CASE("correctness: BIO01 mean annual temperature", "[correctness]") {
    INFO("Backend: " << active_backend());
    auto bio = compute_bioclim(make_ramp_block());
    CHECK_THAT(bio.bio01(0), WithinAbs(expected_ramp::bio01, kTolDefault));
}

TEST_CASE("correctness: BIO02 mean diurnal range", "[correctness]") {
    INFO("Backend: " << active_backend());
    auto bio = compute_bioclim(make_ramp_block());
    CHECK_THAT(bio.bio02(0), WithinAbs(expected_ramp::bio02, kTolDefault));
}

TEST_CASE("correctness: BIO03 isothermality", "[correctness]") {
    INFO("Backend: " << active_backend());
    auto bio = compute_bioclim(make_ramp_block());
    CHECK_THAT(bio.bio03(0), WithinAbs(expected_ramp::bio03, kTolDefault));
}

TEST_CASE("correctness: BIO04 temperature seasonality", "[correctness]") {
    INFO("Backend: " << active_backend());
    auto bio = compute_bioclim(make_ramp_block());
    CHECK_THAT(bio.bio04(0), WithinAbs(expected_ramp::bio04, kTolSqrt));
}

TEST_CASE("correctness: BIO05 max temperature of warmest month",
          "[correctness]") {
    INFO("Backend: " << active_backend());
    auto bio = compute_bioclim(make_ramp_block());
    CHECK_THAT(bio.bio05(0), WithinAbs(expected_ramp::bio05, kTolDefault));
}

TEST_CASE("correctness: BIO06 min temperature of coldest month",
          "[correctness]") {
    INFO("Backend: " << active_backend());
    auto bio = compute_bioclim(make_ramp_block());
    CHECK_THAT(bio.bio06(0), WithinAbs(expected_ramp::bio06, kTolDefault));
}

TEST_CASE("correctness: BIO07 annual temperature range", "[correctness]") {
    INFO("Backend: " << active_backend());
    auto bio = compute_bioclim(make_ramp_block());
    CHECK_THAT(bio.bio07(0), WithinAbs(expected_ramp::bio07, kTolDefault));
}

TEST_CASE("correctness: BIO08 mean temp of wettest quarter", "[correctness]") {
    INFO("Backend: " << active_backend());
    auto bio = compute_bioclim(make_ramp_block());
    CHECK_THAT(bio.bio08(0), WithinAbs(expected_ramp::bio08, kTolDefault));
}

TEST_CASE("correctness: BIO09 mean temp of driest quarter", "[correctness]") {
    INFO("Backend: " << active_backend());
    auto bio = compute_bioclim(make_ramp_block());
    CHECK_THAT(bio.bio09(0), WithinAbs(expected_ramp::bio09, kTolDefault));
}

TEST_CASE("correctness: BIO10 mean temp of warmest quarter", "[correctness]") {
    INFO("Backend: " << active_backend());
    auto bio = compute_bioclim(make_ramp_block());
    CHECK_THAT(bio.bio10(0), WithinAbs(expected_ramp::bio10, kTolDefault));
}

TEST_CASE("correctness: BIO11 mean temp of coldest quarter", "[correctness]") {
    INFO("Backend: " << active_backend());
    auto bio = compute_bioclim(make_ramp_block());
    CHECK_THAT(bio.bio11(0), WithinAbs(expected_ramp::bio11, kTolDefault));
}

TEST_CASE("correctness: BIO12 annual precipitation", "[correctness]") {
    INFO("Backend: " << active_backend());
    auto bio = compute_bioclim(make_ramp_block());
    CHECK_THAT(bio.bio12(0), WithinAbs(expected_ramp::bio12, kTolDefault));
}

TEST_CASE("correctness: BIO13 precipitation of wettest month", "[correctness]") {
    INFO("Backend: " << active_backend());
    auto bio = compute_bioclim(make_ramp_block());
    CHECK_THAT(bio.bio13(0), WithinAbs(expected_ramp::bio13, kTolDefault));
}

TEST_CASE("correctness: BIO14 precipitation of driest month", "[correctness]") {
    INFO("Backend: " << active_backend());
    auto bio = compute_bioclim(make_ramp_block());
    CHECK_THAT(bio.bio14(0), WithinAbs(expected_ramp::bio14, kTolDefault));
}

TEST_CASE("correctness: BIO15 precipitation seasonality", "[correctness]") {
    INFO("Backend: " << active_backend());
    auto bio = compute_bioclim(make_ramp_block());
    CHECK_THAT(bio.bio15(0), WithinAbs(expected_ramp::bio15, kTolSqrt));
}

TEST_CASE("correctness: BIO16 precipitation of wettest quarter",
          "[correctness]") {
    INFO("Backend: " << active_backend());
    auto bio = compute_bioclim(make_ramp_block());
    CHECK_THAT(bio.bio16(0), WithinAbs(expected_ramp::bio16, kTolDefault));
}

TEST_CASE("correctness: BIO17 precipitation of driest quarter",
          "[correctness]") {
    INFO("Backend: " << active_backend());
    auto bio = compute_bioclim(make_ramp_block());
    CHECK_THAT(bio.bio17(0), WithinAbs(expected_ramp::bio17, kTolDefault));
}

TEST_CASE("correctness: BIO18 precipitation of warmest quarter",
          "[correctness]") {
    INFO("Backend: " << active_backend());
    auto bio = compute_bioclim(make_ramp_block());
    CHECK_THAT(bio.bio18(0), WithinAbs(expected_ramp::bio18, kTolDefault));
}

TEST_CASE("correctness: BIO19 precipitation of coldest quarter",
          "[correctness]") {
    INFO("Backend: " << active_backend());
    auto bio = compute_bioclim(make_ramp_block());
    CHECK_THAT(bio.bio19(0), WithinAbs(expected_ramp::bio19, kTolDefault));
}

// ===========================================================================
// Section 3 — NaN propagation: all-NaN input → all 19 outputs NaN
// ===========================================================================

TEST_CASE("correctness: NaN inputs propagate to all 19 BIO outputs",
          "[correctness][nan]") {
    INFO("Backend: " << active_backend());
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

    // All simple statistics propagate NaN.
    CHECK(std::isnan(bio.bio01(0)));  // mean(NaN)
    CHECK(std::isnan(bio.bio02(0)));  // mean(NaN - NaN)
    // BIO03 = 100*BIO02/BIO07 — both operands NaN → NaN
    CHECK(std::isnan(bio.bio03(0)));
    CHECK(std::isnan(bio.bio04(0)));  // stddev(NaN)
    CHECK(std::isnan(bio.bio05(0)));  // max(NaN)
    CHECK(std::isnan(bio.bio06(0)));  // min(NaN)
    CHECK(std::isnan(bio.bio07(0)));  // NaN - NaN
    // Quarter variables use the sentinel path → NaN
    CHECK(std::isnan(bio.bio08(0)));
    CHECK(std::isnan(bio.bio09(0)));
    CHECK(std::isnan(bio.bio10(0)));
    CHECK(std::isnan(bio.bio11(0)));
    CHECK(std::isnan(bio.bio12(0)));  // sum(NaN)
    CHECK(std::isnan(bio.bio13(0)));  // max(NaN)
    CHECK(std::isnan(bio.bio14(0)));  // min(NaN)
    // BIO15: mean(pr)==NaN → guard triggers → NaN
    CHECK(std::isnan(bio.bio15(0)));
    CHECK(std::isnan(bio.bio16(0)));
    CHECK(std::isnan(bio.bio17(0)));
    CHECK(std::isnan(bio.bio18(0)));
    CHECK(std::isnan(bio.bio19(0)));
}

// ===========================================================================
// Section 4 — Partial-NaN regression: mixed NaN + valid pixels
// ===========================================================================

TEST_CASE("correctness: NaN pixel does not corrupt adjacent valid pixel",
          "[correctness][nan]") {
    INFO("Backend: " << active_backend());
    const float nan = std::numeric_limits<float>::quiet_NaN();

    // Two-pixel block: pixel 0 = all NaN, pixel 1 = ramp data
    ClimateBlock data;
    data.tas    = Array2D::from_shape({std::size_t(2), std::size_t(12)});
    data.tasmax = Array2D::from_shape({std::size_t(2), std::size_t(12)});
    data.tasmin = Array2D::from_shape({std::size_t(2), std::size_t(12)});
    data.pr     = Array2D::from_shape({std::size_t(2), std::size_t(12)});

    // Pixel 0: NaN
    for (std::size_t m = 0; m < 12; ++m) {
        data.tas   (0, m) = nan;
        data.tasmax(0, m) = nan;
        data.tasmin(0, m) = nan;
        data.pr    (0, m) = nan;
    }
    // Pixel 1: ramp
    for (std::size_t m = 0; m < 12; ++m) {
        data.tas   (1, m) = static_cast<float>(m + 1);
        data.tasmax(1, m) = static_cast<float>(m + 2);
        data.tasmin(1, m) = static_cast<float>(m);
        data.pr    (1, m) = static_cast<float>(m + 1);
    }

    auto bio = compute_bioclim(data);

    // Pixel 0 → all NaN
    CHECK(std::isnan(bio.bio01(0)));
    CHECK(std::isnan(bio.bio12(0)));

    // Pixel 1 → correct ramp values (NaN pixel must not contaminate)
    CHECK_THAT(bio.bio01(1), WithinAbs(expected_ramp::bio01, kTolDefault));
    CHECK_THAT(bio.bio02(1), WithinAbs(expected_ramp::bio02, kTolDefault));
    CHECK_THAT(bio.bio03(1), WithinAbs(expected_ramp::bio03, kTolDefault));
    CHECK_THAT(bio.bio04(1), WithinAbs(expected_ramp::bio04, kTolSqrt));
    CHECK_THAT(bio.bio05(1), WithinAbs(expected_ramp::bio05, kTolDefault));
    CHECK_THAT(bio.bio06(1), WithinAbs(expected_ramp::bio06, kTolDefault));
    CHECK_THAT(bio.bio07(1), WithinAbs(expected_ramp::bio07, kTolDefault));
    CHECK_THAT(bio.bio08(1), WithinAbs(expected_ramp::bio08, kTolDefault));
    CHECK_THAT(bio.bio09(1), WithinAbs(expected_ramp::bio09, kTolDefault));
    CHECK_THAT(bio.bio10(1), WithinAbs(expected_ramp::bio10, kTolDefault));
    CHECK_THAT(bio.bio11(1), WithinAbs(expected_ramp::bio11, kTolDefault));
    CHECK_THAT(bio.bio12(1), WithinAbs(expected_ramp::bio12, kTolDefault));
    CHECK_THAT(bio.bio13(1), WithinAbs(expected_ramp::bio13, kTolDefault));
    CHECK_THAT(bio.bio14(1), WithinAbs(expected_ramp::bio14, kTolDefault));
    CHECK_THAT(bio.bio15(1), WithinAbs(expected_ramp::bio15, kTolSqrt));
    CHECK_THAT(bio.bio16(1), WithinAbs(expected_ramp::bio16, kTolDefault));
    CHECK_THAT(bio.bio17(1), WithinAbs(expected_ramp::bio17, kTolDefault));
    CHECK_THAT(bio.bio18(1), WithinAbs(expected_ramp::bio18, kTolDefault));
    CHECK_THAT(bio.bio19(1), WithinAbs(expected_ramp::bio19, kTolDefault));
}

// Single-month NaN injection: one month is NaN, rest are valid.
// Quarter variables touch every month in their rolling window, so a single
// NaN must trigger the sentinel path and produce NaN output.
TEST_CASE("correctness: single-month NaN produces NaN in quarter variables",
          "[correctness][nan]") {
    INFO("Backend: " << active_backend());
    const float nan = std::numeric_limits<float>::quiet_NaN();

    ClimateBlock data;
    data.tas    = Array2D::from_shape({std::size_t(1), std::size_t(12)});
    data.tasmax = Array2D::from_shape({std::size_t(1), std::size_t(12)});
    data.tasmin = Array2D::from_shape({std::size_t(1), std::size_t(12)});
    data.pr     = Array2D::from_shape({std::size_t(1), std::size_t(12)});

    for (std::size_t m = 0; m < 12; ++m) {
        data.tas   (0, m) = static_cast<float>(m + 1);
        data.tasmax(0, m) = static_cast<float>(m + 2);
        data.tasmin(0, m) = static_cast<float>(m);
        data.pr    (0, m) = static_cast<float>(m + 1);
    }
    // Inject one NaN into the middle of tas and pr
    data.tas(0, 5) = nan;
    data.pr (0, 5) = nan;

    auto bio = compute_bioclim(data);

    // Quarter variables that depend on tas or pr with a NaN month
    CHECK(std::isnan(bio.bio08(0)));
    CHECK(std::isnan(bio.bio09(0)));
    CHECK(std::isnan(bio.bio10(0)));
    CHECK(std::isnan(bio.bio11(0)));
    CHECK(std::isnan(bio.bio16(0)));
    CHECK(std::isnan(bio.bio17(0)));
    CHECK(std::isnan(bio.bio18(0)));
    CHECK(std::isnan(bio.bio19(0)));

    // BIO15: mean(pr) with one NaN month → NaN (so guard fires → NaN)
    CHECK(std::isnan(bio.bio15(0)));
}

// ===========================================================================
// Section 5 — NoData edge cases
// ===========================================================================

TEST_CASE("correctness: BIO03 is NoData (NaN) when BIO07 is zero",
          "[correctness][nodata]") {
    INFO("Backend: " << active_backend());
    // tasmax == tasmin everywhere → diurnal = 0 → BIO02 = 0 → BIO07 = 0
    ClimateBlock data;
    data.tas    = xt::ones<float>({std::size_t(1), std::size_t(12)}) * 10.0f;
    data.tasmax = xt::ones<float>({std::size_t(1), std::size_t(12)}) * 10.0f;
    data.tasmin = xt::ones<float>({std::size_t(1), std::size_t(12)}) * 10.0f;
    data.pr     = xt::ones<float>({std::size_t(1), std::size_t(12)}) * 5.0f;

    auto bio = compute_bioclim(data);
    CHECK_THAT(bio.bio07(0), WithinAbs(0.0f, kTolDefault));
    CHECK(std::isnan(bio.bio03(0)));
}

TEST_CASE("correctness: BIO15 is NoData (NaN) when mean precipitation is zero",
          "[correctness][nodata]") {
    INFO("Backend: " << active_backend());
    ClimateBlock data;
    data.tas    = xt::ones<float>({std::size_t(1), std::size_t(12)});
    data.tasmax = xt::ones<float>({std::size_t(1), std::size_t(12)});
    data.tasmin = xt::zeros<float>({std::size_t(1), std::size_t(12)});
    data.pr     = xt::zeros<float>({std::size_t(1), std::size_t(12)});

    auto bio = compute_bioclim(data);
    CHECK(std::isnan(bio.bio15(0)));
    // BIO12 (annual precipitation) must still be 0, not NaN
    CHECK_THAT(bio.bio12(0), WithinAbs(0.0f, kTolDefault));
}

// BIO03 and BIO15 both triggered simultaneously
TEST_CASE("correctness: BIO03 and BIO15 both NoData when ranges are zero",
          "[correctness][nodata]") {
    INFO("Backend: " << active_backend());
    // tasmax == tasmin and pr == 0 → BIO07=0 and mean(pr)=0
    ClimateBlock data;
    data.tas    = xt::zeros<float>({std::size_t(1), std::size_t(12)});
    data.tasmax = xt::zeros<float>({std::size_t(1), std::size_t(12)});
    data.tasmin = xt::zeros<float>({std::size_t(1), std::size_t(12)});
    data.pr     = xt::zeros<float>({std::size_t(1), std::size_t(12)});

    auto bio = compute_bioclim(data);
    CHECK(std::isnan(bio.bio03(0)));
    CHECK(std::isnan(bio.bio15(0)));
    // Other BIO variables should still be valid (0)
    CHECK_THAT(bio.bio01(0), WithinAbs(0.0f, kTolDefault));
    CHECK_THAT(bio.bio12(0), WithinAbs(0.0f, kTolDefault));
    CHECK_THAT(bio.bio04(0), WithinAbs(0.0f, kTolDefault));
}

// ===========================================================================
// Section 6 — Uniform-value edge cases
// ===========================================================================

TEST_CASE("correctness: uniform monthly values yield zero seasonality",
          "[correctness][edge]") {
    INFO("Backend: " << active_backend());
    // All months identical → SD = 0 → BIO04 = 0, BIO15 = 0
    ClimateBlock data;
    data.tas    = xt::ones<float>({std::size_t(1), std::size_t(12)}) * 15.0f;
    data.tasmax = xt::ones<float>({std::size_t(1), std::size_t(12)}) * 20.0f;
    data.tasmin = xt::ones<float>({std::size_t(1), std::size_t(12)}) * 10.0f;
    data.pr     = xt::ones<float>({std::size_t(1), std::size_t(12)}) * 50.0f;

    auto bio = compute_bioclim(data);

    CHECK_THAT(bio.bio01(0), WithinAbs(15.0f, kTolDefault)); // mean(15)
    CHECK_THAT(bio.bio04(0), WithinAbs(0.0f,  kTolDefault)); // 100*SD(15)=0
    CHECK_THAT(bio.bio15(0), WithinAbs(0.0f,  kTolDefault)); // 100*0/50=0
    CHECK_THAT(bio.bio12(0), WithinAbs(600.0f, kTolDefault)); // 12*50
    // BIO03: 100 * 10 / 10 = 100
    CHECK_THAT(bio.bio03(0), WithinAbs(100.0f, kTolDefault));
    // Temperature quarter variables equal to uniform monthly value (mean)
    CHECK_THAT(bio.bio08(0), WithinAbs(15.0f, kTolDefault));
    CHECK_THAT(bio.bio09(0), WithinAbs(15.0f, kTolDefault));
    CHECK_THAT(bio.bio10(0), WithinAbs(15.0f, kTolDefault));
    CHECK_THAT(bio.bio11(0), WithinAbs(15.0f, kTolDefault));
    // Precipitation quarter variables are sums of 3 months
    CHECK_THAT(bio.bio16(0), WithinAbs(150.0f, kTolDefault));
    CHECK_THAT(bio.bio17(0), WithinAbs(150.0f, kTolDefault));
    CHECK_THAT(bio.bio18(0), WithinAbs(150.0f, kTolDefault));
    CHECK_THAT(bio.bio19(0), WithinAbs(150.0f, kTolDefault));
}

TEST_CASE("correctness: quarter wrap-around (start index 11)",
          "[correctness][edge]") {
    INFO("Backend: " << active_backend());
    // Arrange pr so the wettest quarter wraps from month 12 → 1 → 2.
    // pr = [100, 100, 1, 1, 1, 1, 1, 1, 1, 1, 1, 100]
    // Quarter sums (circular):
    //   start 0: 100+100+1=201
    //   start 10: 1+100+100=201
    //   start 11: 100+100+100=300  ← maximum
    // sum(pr at 11,0,1) = 100+100+100 = 300
    ClimateBlock data;
    data.tas    = xt::ones<float>({std::size_t(1), std::size_t(12)}) * 5.0f;
    data.tasmax = xt::ones<float>({std::size_t(1), std::size_t(12)}) * 6.0f;
    data.tasmin = xt::ones<float>({std::size_t(1), std::size_t(12)}) * 4.0f;
    data.pr     = xt::ones<float>({std::size_t(1), std::size_t(12)});
    data.pr(0,  0) = 100.0f;
    data.pr(0,  1) = 100.0f;
    data.pr(0, 11) = 100.0f;

    auto bio = compute_bioclim(data);

    // BIO16 = sum pr of wettest quarter = 300.0
    CHECK_THAT(bio.bio16(0), WithinAbs(300.0f, kTolDefault));
}

// ===========================================================================
// Section 7 — Negative temperatures / zero precipitation
// ===========================================================================

TEST_CASE("correctness: negative temperatures are handled correctly",
          "[correctness][edge]") {
    INFO("Backend: " << active_backend());
    // Sub-zero ramp: tas = [-6..5], tasmax = [-5..6], tasmin = [-7..4]
    ClimateBlock data;
    data.tas    = Array2D::from_shape({std::size_t(1), std::size_t(12)});
    data.tasmax = Array2D::from_shape({std::size_t(1), std::size_t(12)});
    data.tasmin = Array2D::from_shape({std::size_t(1), std::size_t(12)});
    data.pr     = Array2D::from_shape({std::size_t(1), std::size_t(12)});
    for (std::size_t m = 0; m < 12; ++m) {
        data.tas   (0, m) = static_cast<float>(m) - 6.0f;  // -6..5
        data.tasmax(0, m) = static_cast<float>(m) - 5.0f;  // -5..6
        data.tasmin(0, m) = static_cast<float>(m) - 7.0f;  // -7..4
        data.pr    (0, m) = static_cast<float>(m + 1);
    }

    auto bio = compute_bioclim(data);

    // BIO01: mean(-6..5) = -0.5
    CHECK_THAT(bio.bio01(0), WithinAbs(-0.5f, kTolDefault));
    // BIO02: mean(diurnal) = mean(2..2) = 2.0 (unchanged from ramp)
    CHECK_THAT(bio.bio02(0), WithinAbs(2.0f, kTolDefault));
    // BIO05: max(tasmax) = 6
    CHECK_THAT(bio.bio05(0), WithinAbs(6.0f, kTolDefault));
    // BIO06: min(tasmin) = -7
    CHECK_THAT(bio.bio06(0), WithinAbs(-7.0f, kTolDefault));
    // BIO07: 6 - (-7) = 13
    CHECK_THAT(bio.bio07(0), WithinAbs(13.0f, kTolDefault));
    // BIO03: 100 * 2 / 13 (same as ramp)
    CHECK_THAT(bio.bio03(0), WithinAbs(200.0f / 13.0f, kTolDefault));
}

TEST_CASE("correctness: large pixel block produces consistent results",
          "[correctness][multiPixel]") {
    INFO("Backend: " << active_backend());
    // All N pixels are identical ramp data → all BIO values must be identical.
    const std::size_t N = 200;
    auto bio = compute_bioclim(make_ramp_block(N));

    std::vector<std::string> failures;
    for (std::size_t p = 0; p < N; ++p) {
        auto chk = [&](const char* name, float actual, float expected,
                       float tol = kTolDefault) {
            auto msg = deviation_message(name, actual, expected, tol);
            if (!msg.empty()) {
                std::ostringstream ss;
                ss << "pixel " << p << " " << msg;
                failures.push_back(ss.str());
            }
        };
        chk("BIO01", bio.bio01(p), expected_ramp::bio01);
        chk("BIO02", bio.bio02(p), expected_ramp::bio02);
        chk("BIO03", bio.bio03(p), expected_ramp::bio03);
        chk("BIO04", bio.bio04(p), expected_ramp::bio04, kTolSqrt);
        chk("BIO05", bio.bio05(p), expected_ramp::bio05);
        chk("BIO06", bio.bio06(p), expected_ramp::bio06);
        chk("BIO07", bio.bio07(p), expected_ramp::bio07);
        chk("BIO08", bio.bio08(p), expected_ramp::bio08);
        chk("BIO09", bio.bio09(p), expected_ramp::bio09);
        chk("BIO10", bio.bio10(p), expected_ramp::bio10);
        chk("BIO11", bio.bio11(p), expected_ramp::bio11);
        chk("BIO12", bio.bio12(p), expected_ramp::bio12);
        chk("BIO13", bio.bio13(p), expected_ramp::bio13);
        chk("BIO14", bio.bio14(p), expected_ramp::bio14);
        chk("BIO15", bio.bio15(p), expected_ramp::bio15, kTolSqrt);
        chk("BIO16", bio.bio16(p), expected_ramp::bio16);
        chk("BIO17", bio.bio17(p), expected_ramp::bio17);
        chk("BIO18", bio.bio18(p), expected_ramp::bio18);
        chk("BIO19", bio.bio19(p), expected_ramp::bio19);
    }

    INFO("Total pixel-variable deviations: " << failures.size());
    for (std::size_t i = 0; i < std::min(failures.size(), std::size_t(10)); ++i) {
        INFO(failures[i]);
    }
    REQUIRE(failures.empty());
}

// ===========================================================================
// Section 8 — OpenMP offload backend
// ===========================================================================

TEST_CASE("correctness: OpenMP-offload path produces expected values [BIO01–19]",
          "[correctness][openmp-offload]") {
    INFO("Backend: " << active_backend());
#if !defined(XBIOCLIM_USE_OPENMP_OFFLOAD)
    SKIP("OpenMP offload not enabled in this build "
         "(rebuild with -DXBIOCLIM_USE_OPENMP_OFFLOAD=ON)");
#else
    // When offload is enabled the same compute_bioclim() entry point is used;
    // the offload pragmas redirect execution to the GPU.  The results must
    // match the CPU reference values to within kTolDefault.
    auto bio = compute_bioclim(make_ramp_block());
    std::vector<std::string> failures;
    auto chk = [&](const char* name, float actual, float expected,
                   float tol = kTolDefault) {
        auto msg = deviation_message(name, actual, expected, tol);
        if (!msg.empty()) failures.push_back(msg);
    };
    chk("BIO01", bio.bio01(0), expected_ramp::bio01);
    chk("BIO02", bio.bio02(0), expected_ramp::bio02);
    chk("BIO03", bio.bio03(0), expected_ramp::bio03);
    chk("BIO04", bio.bio04(0), expected_ramp::bio04, kTolSqrt);
    chk("BIO05", bio.bio05(0), expected_ramp::bio05);
    chk("BIO06", bio.bio06(0), expected_ramp::bio06);
    chk("BIO07", bio.bio07(0), expected_ramp::bio07);
    chk("BIO08", bio.bio08(0), expected_ramp::bio08);
    chk("BIO09", bio.bio09(0), expected_ramp::bio09);
    chk("BIO10", bio.bio10(0), expected_ramp::bio10);
    chk("BIO11", bio.bio11(0), expected_ramp::bio11);
    chk("BIO12", bio.bio12(0), expected_ramp::bio12);
    chk("BIO13", bio.bio13(0), expected_ramp::bio13);
    chk("BIO14", bio.bio14(0), expected_ramp::bio14);
    chk("BIO15", bio.bio15(0), expected_ramp::bio15, kTolSqrt);
    chk("BIO16", bio.bio16(0), expected_ramp::bio16);
    chk("BIO17", bio.bio17(0), expected_ramp::bio17);
    chk("BIO18", bio.bio18(0), expected_ramp::bio18);
    chk("BIO19", bio.bio19(0), expected_ramp::bio19);
    for (const auto& f : failures) { INFO(f); }
    REQUIRE(failures.empty());
#endif
}

TEST_CASE("correctness: OpenMP-offload NaN propagation",
          "[correctness][openmp-offload][nan]") {
    INFO("Backend: " << active_backend());
#if !defined(XBIOCLIM_USE_OPENMP_OFFLOAD)
    SKIP("OpenMP offload not enabled in this build");
#else
    const float nan = std::numeric_limits<float>::quiet_NaN();
    ClimateBlock data;
    data.tas    = Array2D::from_shape({std::size_t(1), std::size_t(12)});
    data.tasmax = Array2D::from_shape({std::size_t(1), std::size_t(12)});
    data.tasmin = Array2D::from_shape({std::size_t(1), std::size_t(12)});
    data.pr     = Array2D::from_shape({std::size_t(1), std::size_t(12)});
    data.tas.fill(nan); data.tasmax.fill(nan);
    data.tasmin.fill(nan); data.pr.fill(nan);
    auto bio = compute_bioclim(data);
    CHECK(std::isnan(bio.bio01(0)));
    CHECK(std::isnan(bio.bio08(0)));
    CHECK(std::isnan(bio.bio15(0)));
    CHECK(std::isnan(bio.bio19(0)));
#endif
}

// ===========================================================================
// Section 9 — CUDA backend
// ===========================================================================

TEST_CASE("correctness: CUDA path produces expected values [BIO01–19]",
          "[correctness][cuda]") {
    INFO("Backend: " << active_backend());
#if !defined(XBIOCLIM_USE_CUDA)
    SKIP("CUDA not enabled in this build "
         "(rebuild with -DXBIOCLIM_USE_CUDA=ON and a CUDA-capable GPU)");
#else
    auto bio = compute_bioclim(make_ramp_block());
    std::vector<std::string> failures;
    auto chk = [&](const char* name, float actual, float expected,
                   float tol = kTolDefault) {
        auto msg = deviation_message(name, actual, expected, tol);
        if (!msg.empty()) failures.push_back(msg);
    };
    chk("BIO01", bio.bio01(0), expected_ramp::bio01);
    chk("BIO02", bio.bio02(0), expected_ramp::bio02);
    chk("BIO03", bio.bio03(0), expected_ramp::bio03);
    chk("BIO04", bio.bio04(0), expected_ramp::bio04, kTolSqrt);
    chk("BIO05", bio.bio05(0), expected_ramp::bio05);
    chk("BIO06", bio.bio06(0), expected_ramp::bio06);
    chk("BIO07", bio.bio07(0), expected_ramp::bio07);
    chk("BIO08", bio.bio08(0), expected_ramp::bio08);
    chk("BIO09", bio.bio09(0), expected_ramp::bio09);
    chk("BIO10", bio.bio10(0), expected_ramp::bio10);
    chk("BIO11", bio.bio11(0), expected_ramp::bio11);
    chk("BIO12", bio.bio12(0), expected_ramp::bio12);
    chk("BIO13", bio.bio13(0), expected_ramp::bio13);
    chk("BIO14", bio.bio14(0), expected_ramp::bio14);
    chk("BIO15", bio.bio15(0), expected_ramp::bio15, kTolSqrt);
    chk("BIO16", bio.bio16(0), expected_ramp::bio16);
    chk("BIO17", bio.bio17(0), expected_ramp::bio17);
    chk("BIO18", bio.bio18(0), expected_ramp::bio18);
    chk("BIO19", bio.bio19(0), expected_ramp::bio19);
    for (const auto& f : failures) { INFO(f); }
    REQUIRE(failures.empty());
#endif
}

TEST_CASE("correctness: CUDA NaN propagation",
          "[correctness][cuda][nan]") {
    INFO("Backend: " << active_backend());
#if !defined(XBIOCLIM_USE_CUDA)
    SKIP("CUDA not enabled in this build");
#else
    const float nan = std::numeric_limits<float>::quiet_NaN();
    ClimateBlock data;
    data.tas    = Array2D::from_shape({std::size_t(1), std::size_t(12)});
    data.tasmax = Array2D::from_shape({std::size_t(1), std::size_t(12)});
    data.tasmin = Array2D::from_shape({std::size_t(1), std::size_t(12)});
    data.pr     = Array2D::from_shape({std::size_t(1), std::size_t(12)});
    data.tas.fill(nan); data.tasmax.fill(nan);
    data.tasmin.fill(nan); data.pr.fill(nan);
    auto bio = compute_bioclim(data);
    CHECK(std::isnan(bio.bio01(0)));
    CHECK(std::isnan(bio.bio08(0)));
    CHECK(std::isnan(bio.bio15(0)));
    CHECK(std::isnan(bio.bio19(0)));
#endif
}
