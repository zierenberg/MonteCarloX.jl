"""
    binary_search(sorted::AbstractVector{T}, value::T)::Int where {T<:Real}

Perform a binary search to return the index i of an sorted array
such that sorted[i-1] < value <= sorted[i]

# Examples
```julia
julia> MonteCarloX.binary_search([1.,2.,3.,4.],2.5)
3
```
```julia
julia> MonteCarloX.binary_search([1,2,3,4],2)
2
```
"""
function binary_search(sorted::AbstractVector{T}, value::T)::Int where {T <: Real}
    # catch lower-bound case that cannot be reached by binary search
    id = 1
    if value > sorted[1]
        index_l = 1
        index_r = length(sorted)
        while index_l < index_r - 1
            # this should be fine because build for integers!
            index_m = fld(index_l + index_r, 2)
            if sorted[index_m] < value
                index_l = index_m
            else
                index_r = index_m
            end
        end
        id = index_r
    end
    return id
end

"""
    log_sum(a::Real,b::Real)

Return result of logarithmic sum ``c = \\ln(A+B) = a + \\ln(1+e^{|b-a|})`` where ``C =
e^c = A+B = e^a + e^b``.

This is useful for sums that involve elements that span multiple orders of
magnitude, e.g., the partition sum that is required as normalization factor
during reweighting.

# Examples
```julia
julia> exp(MonteCarloX.log_sum(log(2.), log(3.)))
5.000000000000001

```
"""
@inline function log_sum(a::Real, b::Real)
    ap, bp = promote(a, b)
    return ap > bp ? ap + log1p(exp(bp - ap)) : bp + log1p(exp(ap - bp))
end

"""
    log_sum(values::AbstractVector{<:Real})

Stable reduction of log-weights:
`log(sum(exp.(values)))`.
"""
function log_sum(values::AbstractVector{<:Real})
    n = length(values)
    n > 0 || throw(ArgumentError("values must be non-empty"))
    acc = values[1]
    @inbounds for i in 2:n
        acc = log_sum(acc, values[i])
    end
    return acc
end

"""
    logistic(x::Real)

Numerically stable logistic sigmoid:

`σ(x) = 1 / (1 + exp(-x))`
"""
@inline function logistic(x::Real)
    if x ≥ 0
        return inv(1 + exp(-x))
    end
    ex = exp(x)
    return ex / (1 + ex)
end

"""
    distribution_from_logdos(logdos::BinnedObject, β::Real)

Given a log-DOS (logarithmic density of states) as a BinnedObject, return the corresponding distribution at inverse temperature `β` by reweighting with the Boltzmann factor `exp(-β * E)` and normalizing.
"""
function distribution_from_logdos(logdos::BinnedObject, β::Real)
    E = get_centers(logdos)
    log_w = logdos.values .- β .* E
    mask = isfinite.(log_w)
    any(mask) || throw(ArgumentError("logdos must contain at least one finite entry"))
    log_Z = log_sum(log_w[mask])

    out = zero(logdos)
    out.values[mask] .= exp.(log_w[mask] .- log_Z)
    return out
end




###### Additional utility of Histogram class from StatsBase (does not work with documenter right now)
# """
#   getindex(h::AbstractHistogram, x)

# access from histogram `h` the bin that corresponds to the coordinate `x`. `x`
# can be high-dimensional. Returns `missing` in case the bin does not exist.

# # Examples

# ```julia
# julia> using StatsBase
# julia> h = fit(Histogram, [1,1,1,2], 1:3);
# julia> h[1]
# 3
# julia> h[2]
# 1
# julia> h[3]
# missing
# ```
# """
Base.getindex(h::AbstractHistogram{T,1}, x::Real) where {T} = getindex(h, (x,))
function Base.getindex(h::Histogram{T,N}, xs::NTuple{N,Real}) where {T,N}
    idx = StatsBase.binindex(h, xs)
    if checkbounds(Bool, h.weights, idx...)
        return @inbounds h.weights[idx...]
    else
        return missing
    end
end

# """
#   setindex(h::AbstractHistogram, value x)

# asign from histogram `h` the bin that corresponds to the coordinate `x` a
# certain `value` . `x` can be high-dimensional.

# # Examples

# ```julia
# julia> using StatsBase
# julia> h = Histogram(1:3);
# julia> h[1]
# 0
# julia> h[1] = 3
# 3
# julia> h[1]
# 3
# julia> h[3] = 3
# missing
# """
Base.setindex!(h::AbstractHistogram{T,1}, value::Real, x::Real) where {T} = setindex!(h, value, (x,))
function Base.setindex!(h::Histogram{T,N}, value::Real, xs::NTuple{N,Real}) where {T,N}
    h.isdensity && error("Density histogram must have float-type weights")
    idx = StatsBase.binindex(h, xs)
    if checkbounds(Bool, h.weights, idx...)
        @inbounds h.weights[idx...] = value
    else
        throw(DomainError(idx,"the histogram is not defined for this value"))
    end
end



import StatsBase.kldivergence
"""
    kldivergence(P::Histogram, Q::Function)

Kullback-Leibler divergence between an empirical distribution (measured) and a reference distribution (analytic)

So far this is defined only for 1-dimensional distributions of type StatsBase.Histogram
"""
function kldivergence(P::Histogram, Q::Function)
    ##KL divergence sum P(args)logP(args)/Q(args)
    ## P=P_meas, Q=P_true s.t P(args)=0 simply ignored
    kld = 0.0
    for (i, x) in enumerate(P.edges[1])
        if i <= length(P.weights)
            @inbounds p = P.weights[i]
            if p > 0
                q = Q(x)
                kld += p * log(p / q)
            end
        end
    end
    return kld
end
