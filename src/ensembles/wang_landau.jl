mutable struct WangLandauEnsemble{BO<:BinnedObject} <: AbstractEnsemble
    logweight::BO
    logf::Float64
end

WangLandauEnsemble(logweight::BO; logf::Real=1.0) where {BO<:BinnedObject} =
    WangLandauEnsemble(logweight, Float64(logf))

function WangLandauEnsemble(bins; init::Real=0.0, logf::Real=1.0)
    logweight = bins isa BinnedObject ? bins : BinnedObject(bins, float(init))
    return WangLandauEnsemble(logweight; logf=logf)
end

@inline logweight(e::WangLandauEnsemble) = e.logweight
@inline logweight(e::WangLandauEnsemble, x) = e.logweight(x)

"""
    update!(e::WangLandauEnsemble; power=0.5)

Update Wang-Landau schedule by scaling the modification factor:
`logf <- power * logf` with default `power=0.5`.
"""
@inline update!(e::WangLandauEnsemble; power::Real=0.5) = (e.logf *= power; nothing)