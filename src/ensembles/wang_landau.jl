mutable struct WangLandauEnsemble{BO<:BinnedObject} <: AbstractEnsemble
    logweight::BO
    histogram::BO
    logf::Float64
    visited::BitVector
end

WangLandauEnsemble(logweight::BO, histogram::BO; logf::Real=1.0) where {BO<:BinnedObject} =
    WangLandauEnsemble(logweight, histogram, Float64(logf), histogram.values .> 0)

WangLandauEnsemble(logweight::BO, histogram::BO, logf::Real) where {BO<:BinnedObject} =
    WangLandauEnsemble(logweight, histogram, Float64(logf), histogram.values .> 0)

WangLandauEnsemble(logweight::BO; logf::Real=1.0) where {BO<:BinnedObject} =
    WangLandauEnsemble(logweight, zero(logweight), Float64(logf))

function WangLandauEnsemble(bins; init::Real=0.0, logf::Real=1.0)
    logweight = bins isa BinnedObject ? bins : BinnedObject(bins, float(init))
    return WangLandauEnsemble(logweight; logf=logf)
end

@inline logweight(e::WangLandauEnsemble) = e.logweight
@inline logweight(e::WangLandauEnsemble, arg) = e.logweight(arg)

@inline MonteCarloX.should_record_visit(e::WangLandauEnsemble) = true
@inline function MonteCarloX.record_visit!(e::WangLandauEnsemble, x_vis)
    e.histogram[x_vis] += 1
    return nothing
end

"""
    update_logweight!(e::WangLandauEnsemble; power=0.5, rebase=true)

Update Wang-Landau schedule by scaling the modification factor:
`logf <- power * logf` with default `power=0.5`. When `rebase=true`, shift only
visited logweight entries so their minimum is zero; unvisited entries are unchanged.
"""
function update_logweight!(e::WangLandauEnsemble; power::Real=0.5, rebase::Bool=true)
    e.logf *= power
    e.visited .|= e.histogram.values .> 0
    if rebase && any(e.visited)
        visited_values = e.logweight.values[e.visited]
        e.logweight.values[e.visited] .-= minimum(visited_values)
    end
    return nothing
end