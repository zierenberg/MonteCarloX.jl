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

@inline get_centers(e::WangLandauEnsemble, dim::Int=1) = get_centers(e.logweight, dim)
@inline Base.values(e::WangLandauEnsemble) = Base.values(e.logweight)
@inline get_values(e::WangLandauEnsemble) = Base.values(e)
