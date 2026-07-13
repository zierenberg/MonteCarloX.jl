# Second-moment correlation length derived from the structure factor.

"""
    correlation_length(sys) -> Float64

Second-moment correlation length `ξ = (1/2sin(π/L)) √(S(0)/S(k_min) − 1)`, averaged over lattice
axes, with `S(0) = (Σσ)²`. Returns `0` when the structure factor vanishes (up to the roundoff of
its Fourier sums, so uniform configurations give `0` rather than a huge spurious `ξ`).
"""
function correlation_length(sys::SpinSystem)
    dims = _lattice_dims(sys)
    S0 = float(magnetization(sys))^2
    # The c/s sums in structure_factor carry O(N·eps) roundoff, so an exactly vanishing S(k) shows
    # up as O((N·eps)²) > 0; anything below this floor is noise, not signal.
    tol = (4 * length(sys.spins) * eps())^2
    acc = 0.0
    n = 0
    for d in eachindex(dims)
        L = dims[d]
        L < 2 && continue
        Sk = structure_factor(sys, d)
        Sk <= tol && continue
        acc += sqrt(max(S0 / Sk - 1, 0.0)) / (2 * sin(π / L))
        n += 1
    end
    return n > 0 ? acc / n : 0.0
end
