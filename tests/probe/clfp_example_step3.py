from mmasim_kernels.nv_ptx.turing import mma_kernels
from features.special_values import (
    is_negative_zero_supported,
    is_infinity_supported,
    is_nan_supported,
    is_subnormal_supported,
)
from features.fdpa_precision import probe_fdpa_c_precision, probe_fdpa_ab_precision
from features.fdpa_internal_order import probe_internal_summation_order
from features.fdpa_alignment_rounding import (
    probe_fdpa_c_rounding,
    probe_fdpa_ab_rounding,
)
from features.fdpa_product_renormalization import is_fdpa_ab_renormalized
from features.fdpa_product_overflow_underflow import (
    is_fdpa_ab_overflowing,
    is_fdpa_ab_underflowing,
)
from features.fdpa_output_rounding import (
    probe_fdpa_output_precision,
    probe_output_rounding,
    probe_output_rne_precision,
)
from features.fdpa_exponent_for_zero import probe_exponent_for_zero_rne_fp16

kernel = mma_kernels["m16n8k8.f16.f16.f16.f16"]

# special values
assert not is_negative_zero_supported(kernel)
assert is_infinity_supported(kernel)
assert is_nan_supported(kernel)
assert is_subnormal_supported(kernel)

# precision
F = 24
assert probe_fdpa_c_precision(kernel) == F
assert probe_fdpa_ab_precision(kernel) == F

# internal summation order
order_info = probe_internal_summation_order(kernel, F)
for i in range(kernel.k + 1):
    for j in range(i):
        assert order_info[i, j] == 0

# alignment rounding
RZ = {0.25: 0, 0.5: 0, 0.75: 0, 1.5: 1, -0.25: 0, -0.5: 0, -0.75: 0, -1.5: -1}
rounding_info = probe_fdpa_c_rounding(kernel, F)
assert rounding_info == RZ
rounding_info = probe_fdpa_ab_rounding(kernel, F)
assert rounding_info == RZ

# product renormalization
assert not is_fdpa_ab_renormalized(kernel, F)

# product overflow / underflow
assert not is_fdpa_ab_overflowing(kernel)
assert not is_fdpa_ab_underflowing(kernel)

# output rounding
RNE = {0.25: 0, 0.5: 0, 0.75: 1, 1.5: 2, -0.25: 0, -0.5: 0, -0.75: -1, -1.5: -2}
output_prec = probe_fdpa_output_precision(kernel)
assert output_prec == 10
rounding_info = probe_output_rounding(kernel, output_prec)
assert rounding_info == RNE
assert probe_output_rne_precision(kernel, output_prec) == F

# exponent for zero
assert probe_exponent_for_zero_rne_fp16(kernel, F) == -20


print("Step 3 passed.")
