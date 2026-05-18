mutable struct MulticanonicalEnsemble{BO<:BinnedObject} <: AbstractEnsemble
    logweight::BO
    histogram::BO
    log_p_acc::Union{Nothing, Vector{Float64}}
    record_visits::Bool

    function MulticanonicalEnsemble(logweight::BO, histogram::BO; record_visits::Bool=true) where {BO<:BinnedObject}
        _assert_same_domain(logweight, histogram)
        new{BO}(logweight, histogram, nothing, record_visits)
    end
end
MulticanonicalEnsemble(logweight::BO; histogram=nothing, record_visits::Bool=true) where {BO<:BinnedObject} =
    MulticanonicalEnsemble(logweight, histogram === nothing ? zero(logweight) : histogram; record_visits=record_visits)

function MulticanonicalEnsemble(bins; init::Real=0.0, record_visits::Bool=true)
    lw = bins isa BinnedObject ? bins : BinnedObject(bins, float(init))
    histogram = zero(lw)
    return MulticanonicalEnsemble(lw, histogram; record_visits=record_visits)
end

@inline logweight(e::MulticanonicalEnsemble) = e.logweight # this is already a callable BinnedObject, so we can just return it
@inline logweight(e::MulticanonicalEnsemble, x) = e.logweight(x)

@inline should_record_visit(ens::MulticanonicalEnsemble) = ens.record_visits

@inline function record_visit!(ens::MulticanonicalEnsemble, x_vis)
    h = ens.histogram
    # multidimensional aspects should be handled by the BinnedObject indexing.
    h[x_vis] += 1
    return nothing
end

"""
    update!(e::MulticanonicalEnsemble; mode=:simple)

Update logweights from the current histogram.

# Modes
- `:simple` — trivial update: `W[i] -= log(H[i])` (Berg & Neuhaus, 1992)
- `:recursive` — precision-weighted recursive update that accumulates
  statistics across iterations, preventing overcorrection when the walker
  gets stuck in one region (Berg, J. Stat. Phys. 82, 323, 1996;
  Janke, Physica A 254, 164, 1998).
"""
function update!(e::MulticanonicalEnsemble; mode::Symbol=:simple)
    if mode === :simple
        _update_simple!(e)
    elseif mode === :recursive
        _update_recursive!(e)
    else
        throw(ArgumentError("unsupported mode=$(mode), use :simple or :recursive"))
    end
    return nothing
end

function _update_simple!(e::MulticanonicalEnsemble)
    @inbounds for idx in eachindex(e.histogram.values)
        h = e.histogram.values[idx]
        logh = h > 0 ? log(h) : 0.0
        e.logweight.values[idx] -= logh
    end
end

# Multicanonical recursion (Berg, J. Stat. Phys. 82, 323, 1996;
# Janke, Physica A 254, 164, 1998).
# For each adjacent bin pair (j-1, j), computes a transition precision
# p = H[j-1]*H[j] / (H[j-1]+H[j]) and accumulates it across iterations.
# The entropy difference ΔS(E) = S(E+ΔE) - S(E) is estimated as a
# precision-weighted average of previous and current estimates:
#   ΔS_new = (1-κ)*ΔS_old + κ*ΔS_measured,  κ = p / p_acc
# This prevents destroying well-converged weights in regions not visited
# during the current iteration.
function _update_recursive!(e::MulticanonicalEnsemble)
    n = length(e.logweight.values)
    n >= 2 || return

    # lazily initialize accumulator
    if e.log_p_acc === nothing
        e.log_p_acc = fill(-Inf, n)  # log(0) = -Inf
    end
    log_p_acc = e.log_p_acc

    # microcanonical entropy S = -W
    S = similar(e.logweight.values)
    @inbounds for i in 1:n
        S[i] = -e.logweight.values[i]
    end

    # sweep from high to low index: S[j-1] = S[j] + ΔS
    @inbounds for j in n:-1:2
        h_lo = e.histogram.values[j-1]
        h_hi = e.histogram.values[j]

        # default: use previous weights as ΔS estimate
        deltaS = -(e.logweight.values[j-1] - e.logweight.values[j])

        if h_lo > 0 && h_hi > 0
            # transition precision: p = H_lo * H_hi / (H_lo + H_hi)
            log_p = log(h_lo) + log(h_hi) - log(h_lo + h_hi)

            # accumulate precision
            log_p_acc[j] = log_p_acc[j] == -Inf ? log_p : log_sum(log_p_acc[j], log_p)

            # blending weight
            kappa = exp(log_p - log_p_acc[j])

            # measured entropy difference from histogram
            deltaS_measured = (log(h_lo) - e.logweight.values[j-1]) -
                              (log(h_hi) - e.logweight.values[j])

            deltaS = (1.0 - kappa) * deltaS + kappa * deltaS_measured
        end

        S[j-1] = S[j] + deltaS
    end

    # write back: W = -S
    @inbounds for i in 1:n
        e.logweight.values[i] = -S[i]
    end
end