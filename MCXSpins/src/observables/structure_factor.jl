# Static structure factor at the smallest nonzero wavevector of a periodic hypercubic lattice.

"""
    structure_factor(sys, d) -> Float64

Static structure factor `|Σ_j σ_j e^{i k·r_j}|²` at the smallest wavevector along axis `d`
(`k = 2π/L_d`). These are the `cos`/`sin` Fourier sums used for the correlation length.
"""
function structure_factor(sys::SpinSystem, d::Int)
    dims = sys.topo.dims
    L = dims[d]
    L < 2 && return 0.0
    stride = d == 1 ? 1 : prod(dims[1:d-1])
    k = 2π / L
    c = 0.0
    s = 0.0
    @inbounds for site in eachindex(sys.spins)
        x = ((site - 1) ÷ stride) % L
        σ = Int(sys.spins[site])
        c += σ * cos(k * x)
        s += σ * sin(k * x)
    end
    return c^2 + s^2
end
