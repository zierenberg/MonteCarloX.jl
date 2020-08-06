"""
    binary_search(sorted::AbstractVector{T}, value::T)::Int where {T<:Real}

Perfom a binary search to return the index i of an sorted array
such that sorted[i-1] < value <= sorted[i]

# Examples
```jldoctest
julia> MonteCarloX.binary_search([1.,2.,3.,4.],2.5)
3
```
```jldoctest
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
    log_sum(a::T,b::T)

Return result of logarithmic sum ``c = \\ln(A+B) = a + \\ln(1+e^{|b-a|})`` where ``C =
e^c = A+B = e^a + e^b``. 

This is useful for sums that involve elements that span multiple orders of
magnitude, e.g., the partition sum that is required as normalization factor
during reweighting.

# Examples
```jldoctest
julia> exp(MonteCarloX.log_sum(log(2.), log(3.)))
5.000000000000001

```

 
"""
function log_sum(a::T, b::T)::T where T <: AbstractFloat
    if b < a
        return a + log(1 + exp(b - a)) 
    else
        return b + log(1 + exp(a - b))
    end
end


"""
    random_element(list_probabilities::Vector{T},rng::AbstractRNG)::Int where T<:AbstractFloat

Pick an index with probability defined by `list_probability` (which needs to be normalized). 

#Remark
Deprecated for use of StatsBase.sample

# Examples
```jldoctest
julia> using Random

julia> rng = MersenneTwister(1000);

julia> MonteCarloX.random_element([0.1,0.2,0.3,0.4],rng)
4
julia> MonteCarloX.random_element([0.1,0.2,0.3,0.4],rng)
4
julia> MonteCarloX.random_element([0.1,0.2,0.3,0.4],rng)
3
julia> MonteCarloX.random_element([0.1,0.2,0.3,0.4],rng)
4
```
"""
function random_element(list_probabilities::Vector{Float64}, rng::AbstractRNG)::Int 
    theta = rand(rng) * sum(list_probabilities)

    id = 1
    cumulated_prob = list_probabilities[id]
    while cumulated_prob < theta
        id += 1
        @inbounds cumulated_prob += list_probabilities[id]
    end

    return id
end

# TODO: iterate over n-dimensional StatsBase.Histogram
# for start, CartesianIndices(hist.weights) can give you a way to iterate over
# weights, and the cartesian index is your bin, if you want to convert the bin
# number back to edge value, you just need to access the corresponding element
# in hist.edges

# does this need performance boost? - maybe this is faseter when explicit for 1D
Base.getindex(h::AbstractHistogram{T,1}, x::Real) where {T} = getindex(h, (x,))

function Base.getindex(h::Histogram{T,N}, xs::NTuple{N,Real}) where {T,N}
    idx = StatsBase.binindex(h, xs)
    if checkbounds(Bool, h.weights, idx...)
        return @inbounds h.weights[idx...]
    else
        return missing
    end
end

Base.setindex!(h::AbstractHistogram{T,1}, value::Real, x::Real) where {T} = setindex!(h, value, (x,))

function Base.setindex!(h::Histogram{T,N}, value::Real, xs::NTuple{N,Real}) where {T,N}
    h.isdensity && error("Density histogram must have float-type weights")
    idx = StatsBase.binindex(h, xs)
    if checkbounds(Bool, h.weights, idx...)
        @inbounds h.weights[idx...] = value
    else
        return missing
    end
end
