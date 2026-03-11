mutable struct MulticanonicalEnsemble{BO<:BinnedObject} <: AbstractEnsemble
    logweight::BO
    histogram::BO
    record_visits::Bool

    function MulticanonicalEnsemble(logweight::BO, histogram::BO; record_visits::Bool=true) where {BO<:BinnedObject}
        _assert_same_domain(logweight, histogram)
        new{BO}(logweight, histogram, record_visits)
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

function record_visit!(ens::MulticanonicalEnsemble, x_vis)
    h = ens.histogram
    if x_vis isa Tuple
        idx_new = _binindex(h.bins, x_vis)
        if all(d -> (1 <= idx_new[d] <= size(h.values, d)), 1:length(idx_new))
            h[x_vis...] += 1
        end
    else
        idx_new = _binindex(h.bins[1], x_vis)
        if 1 <= idx_new <= size(h.values, 1)
            h[x_vis] += 1
        end
    end

    return nothing
end

function update!(e::MulticanonicalEnsemble; mode::Symbol=:simple)
    if mode != :simple
        throw(ArgumentError("unsupported mode=$(mode), currently only :simple"))
    end

    @inbounds for idx in eachindex(e.histogram.values)
        h = e.histogram.values[idx]
        logh = h > 0 ? log(h) : 0.0
        e.logweight.values[idx] -= logh
    end

    return nothing
end