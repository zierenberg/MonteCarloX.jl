mutable struct MulticanonicalEnsemble{BO<:BinnedObject} <: AbstractEnsemble
    logweight::BO
    histogram::BO
    record_visits::Bool

    function MulticanonicalEnsemble(logweight::BO, histogram::BO; record_visits::Bool=true) where {BO<:BinnedObject}
        _assert_same_domain(logweight, histogram)
        new{BO}(logweight, histogram, record_visits)
    end
end
MulticanonicalEnsemble(logweight::BO; histogram=nothing) where {BO<:BinnedObject} =
    MulticanonicalEnsemble(logweight, histogram === nothing ? zero(logweight) : histogram)

function MulticanonicalEnsemble(bins; init::Real=0.0)
    lw = bins isa BinnedObject ? bins : BinnedObject(bins, float(init))
    histogram = zero(lw)
    return MulticanonicalEnsemble(lw, histogram)
end

@inline logweight(e::MulticanonicalEnsemble) = e.logweight
@inline logweight(e::MulticanonicalEnsemble, x) = e.logweight(x)

@inline get_centers(e::MulticanonicalEnsemble, dim::Int=1) = get_centers(e.logweight, dim)
@inline Base.values(e::MulticanonicalEnsemble) = Base.values(e.logweight)
@inline get_values(e::MulticanonicalEnsemble) = Base.values(e)

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


function set!(
    e::MulticanonicalEnsemble,
    xrange::Union{Tuple{<:Real,<:Real},AbstractRange{<:Real}},
    f::Function,
)
    length(size(e.logweight.values)) == 1 ||
        throw(ArgumentError("`set!` currently supports only 1D binned log-weights"))

    cs = collect(e.logweight.bins[1])
    n = length(cs)

    xleft, xright = if xrange isa Tuple
        Float64(min(xrange[1], xrange[2])), Float64(max(xrange[1], xrange[2]))
    else
        Float64(min(first(xrange), last(xrange))), Float64(max(first(xrange), last(xrange)))
    end

    idx_left = clamp(searchsortedfirst(cs, xleft), 1, n)
    idx_right = clamp(searchsortedlast(cs, xright), 1, n)
    idx_left <= idx_right ||
        throw(ArgumentError("selected range does not overlap any bin centers"))

    if cs[idx_left] > xright || cs[idx_right] < xleft
        throw(ArgumentError("selected range does not overlap any bin centers"))
    end

    @inbounds for i in idx_left:idx_right
        x = cs[i]
        e.logweight.values[i] = Float64(f(x))
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