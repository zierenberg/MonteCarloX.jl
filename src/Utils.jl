"""
    binary_search(cumulated_sum::AbstractVector{T}, value::T)::Int where {T<:Real}

Perfom a binary search to return the index i of an ordered array
(cumulated_sum) such that cumulated_sum[i-1] < value <= cumulated_sum[i]

# Examples
```jldoctest
julia> MonteCarloX.binary_search([[1.,2.,3.,4.],2.5)
3
```
```jldoctest
julia> MonteCarloX.binary_search([[1,2,3,4],2)
2
```
"""
function binary_search(cumulated_sum::AbstractVector{T}, value::T)::Int where {T<:Real}
  #catch lower-bound case that cannot be reached by binary search
  id = 1
  if value > cumulated_sum[1] 
    index_l = 1
    index_r = length(cumulated_sum)
    while index_l < index_r-1
      #this should be fine because build for integers!
      index_m = fld(index_l+index_r, 2)
      if cumulated_sum[index_m] < value
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

Return result of logarithmic sum ``c = ln(A+B) = a + ln(1+e^{|b-a|})`` where ``C =
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
function log_sum(a::T,b::T)::T where T<:AbstractFloat
  if b < a
    return a + log(1+exp(b-a)) 
  else
    return b + log(1+exp(a-b))
  end
end
