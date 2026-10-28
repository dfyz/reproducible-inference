#pragma once

#include "truncate.h"

double approx_square(double x) {
    // > [...] We are able to truncate the squarer output,
    // > such that we only require 15 output bits [...]
    // 15 output bits means dropping everything below 2^-27.
    constexpr int THRESHOLD = 27;

    double x_orig = x;
    double res = 0.0f;

    for (double place = 0x1p-7; place >= 0x1p-21; place *= 0.5) {
        if (x >= place) {
            // > [...] It uses a well-known technique of re-arranging
            // > the partial product matrix due to identical multiplier input [...]
            //
            // > [...] We further truncate [...] partial product bits themselves which
            // > contribute toward output bits which would ultimately be truncated [...]
            //
            // We don't rearrange the partial product matrix, so we have to increase
            // the truncation threshold.
            res += truncate(x_orig * place, THRESHOLD + 1);
            x -= place;
        }
    }
    return truncate(res, THRESHOLD);
}
