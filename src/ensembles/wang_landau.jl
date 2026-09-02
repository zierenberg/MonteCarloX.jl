mutable struct WangLandauEnsemble{BO<:BinnedObject} <: AbstractEnsemble
    logweight::BO
    histogram::BO
    logf::Float64
end

WangLandauEnsemble(logweight::BO, histogram::BO; logf::Real=1.0) where {BO<:BinnedObject} =
    WangLandauEnsemble(logweight, histogram, Float64(logf))

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
    update_logweight!(e::WangLandauEnsemble; power=0.5)

Update Wang-Landau schedule by scaling the modification factor:
`logf <- power * logf` with default `power=0.5`.
"""
@inline update_logweight!(e::WangLandauEnsemble; power::Real=0.5) = (e.logf *= power; nothing)