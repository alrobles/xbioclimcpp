#include "xbioclim/bioclim.hpp"

#include <xtensor/xoperation.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace xbioclim {

static constexpr float NODATA_FLOAT = std::numeric_limits<float>::quiet_NaN();

BioBlock compute_bioclim(const ClimateBlock& data) {
    BioBlock bio;

    // --- Validate ClimateBlock invariants ---
    const std::size_t N = data.tas.shape(0);
    if (data.tasmax.shape(0) != N ||
        data.tasmin.shape(0) != N ||
        data.pr.shape(0)     != N) {
        throw std::invalid_argument(
            "ClimateBlock: all fields must have the same n_pixels (shape(0))");
    }
    if (data.tas.shape(1)    != 12 ||
        data.tasmax.shape(1) != 12 ||
        data.tasmin.shape(1) != 12 ||
        data.pr.shape(1)     != 12) {
        throw std::invalid_argument(
            "ClimateBlock: all fields must have exactly 12 months (shape(1))");
    }

    // --- Diurnal range: [N_pixels, 12] ---
    Array2D diurnal = data.tasmax - data.tasmin;

    // --- Simple statistics ---
    bio.bio01 = row_mean(data.tas);
    bio.bio02 = row_mean(diurnal);
    bio.bio05 = row_max(data.tasmax);
    bio.bio06 = row_min(data.tasmin);
    bio.bio07 = bio.bio05 - bio.bio06;

    // BIO03: guard against division by zero (BIO07 == 0)
    bio.bio03 = xt::where(xt::not_equal(bio.bio07, 0.0f),
                          100.0f * bio.bio02 / bio.bio07,
                          NODATA_FLOAT);

    bio.bio04 = 100.0f * row_std(data.tas);
    bio.bio12 = row_sum(data.pr);
    bio.bio13 = row_max(data.pr);
    bio.bio14 = row_min(data.pr);

    // BIO15: precipitation CV; guard against mean(pr) == 0
    Array1D pr_mean = row_mean(data.pr);
    Array1D pr_std  = row_std(data.pr);
    bio.bio15 = xt::where(pr_mean > 0.0f,
                          100.0f * pr_std / pr_mean,
                          NODATA_FLOAT);

    // --- Rolling quarter indices ---
    IndexArray wet_q  = rolling_quarter_argmax(data.pr);
    IndexArray dry_q  = rolling_quarter_argmin(data.pr);
    IndexArray warm_q = rolling_quarter_argmax(data.tas);
    IndexArray cold_q = rolling_quarter_argmin(data.tas);

    // --- BIO08/09: mean temperature of wettest/driest quarter ---
    bio.bio08 = quarter_mean(data.tas, wet_q);
    bio.bio09 = quarter_mean(data.tas, dry_q);

    // --- BIO10/11: mean temperature of warmest/coldest quarter ---
    bio.bio10 = quarter_mean(data.tas, warm_q);
    bio.bio11 = quarter_mean(data.tas, cold_q);

    // --- BIO16/17: sum of precipitation of wettest/driest quarter ---
    bio.bio16 = quarter_sum(data.pr, wet_q);
    bio.bio17 = quarter_sum(data.pr, dry_q);

    // --- BIO18/19: sum of precipitation of warmest/coldest quarter ---
    bio.bio18 = quarter_sum(data.pr, warm_q);
    bio.bio19 = quarter_sum(data.pr, cold_q);

    return bio;
}

} // namespace xbioclim
