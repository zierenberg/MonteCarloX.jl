#Binary Search here
#
function binary_search(cumulated_sum::AbstractVector{T}, value::T)::Int where {T<:Real}
  #catch lower-bound case that cannot be reached by binary search
  id = 1
  if value >= cumulated_sum[1] 
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
logarithmic addition of type C = e^c = A+B = e^a + e^b
 c = ln(A+B) = a + ln(1+e^{b-a})
 with b-a < 1
"""
function log_sum(a,b)
  if b < a
    return a + log(1+exp(b-a)) 
  else
    return b + log(1+exp(a-b))
  end
end
