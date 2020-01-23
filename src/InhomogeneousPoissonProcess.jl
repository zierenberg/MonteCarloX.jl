"""
Module MonteCarloX.InhomogeneousPoissonProcess

basic functions to evaluate (a coupled set of) inhomogeneous Poisson processes with time-dependent rate function rate(t)
"""
module InhomogeneousPoissonProcess
using Random
using Distributions

"""
next_event_time(rate::Function, max_rate::Float64, rng::AbstractRNG)::Float64

Generate a new event from an inhomogeneous poisson process with rate Lambda(t).
Based on (Ogata’s Modified Thinning Algorithm: Ogata,  1981,  p.25,  Algorithm  2)
see also https://www.math.fsu.edu/~ychen/research/Thinning%20algorithm.pdf

# Arguments
- `rate`: rate(dt) has to be defined outside (e..g t-> rate(t+t0,args))
- `max_rate`: maximal rate in near future (has to be evaluated externally)
- `rng`: random number generator

API - output
* returns the next event time

"""
function next_event_time(rate::Function, max_rate::Float64, rng::AbstractRNG)::Float64
  dt = 0.0
  theta = 1.0/max_rate
  while true
    # generate next event from bounding homogeneous Poisson process with max_rate
    dt += rand(rng,Exponential(theta))
    # accept next event with probability rate(t)/rate_max [Thinning algorithm]
    if rand(rng) < rate(dt)/max_rate
      return dt
    end
  end
end

"""
Generate a new event from an inhomogeneous poisson process with rate Lambda(t) under
the assumption that rate(dt) is monotonically decreasing.
Based on (Ogata’s Modified Thinning Algorithm: Ogata,  1981,  p.25,  Algorithm  3)
see also https://www.math.fsu.edu/~ychen/research/Thinning%20algorithm.pdf

# Arguments
- `rate`: rate(dt) has to be defined outside (e..g t-> rate(t+t0,args))
- `rng`: random number generator

API - output
* returns the next event time

"""
function next_event_time_for_piece_wise_decreasing_rate(rate::Function, rng::AbstractRNG)::Float64
  dt = 0.0
  while true
    # future rate can only be smaller than current rate
    max_rate = rate(dt)
    # generate next event from bounding homogeneous Poisson process with max_rate
    dt += rand(rng,Exponential(1.0/max_rate))
    # accept next event with probability rate(t)/rate_max [Thinning algorithm]
    if rand(rng) < rate(dt)/max_rate
      return dt
    end
  end
end

"""
Generate a new event from a collection of inhomogeneous poisson processes with
rates Lambda(t).
# Arguments
- `rates`: rates(dt); Float -> [Float]
- `max_rate`: maximal rate in near future (has to be evaluated externally)
- `rng`: random number generator

API - output
* returns the next event time and its id ass tuple (dt, id)
"""
#TODO: FunctionWrapper.jl ?
function next_event(rates::Function, max_rate::Float64, rng::AbstractRNG)::Tuple{Float64,Int}
  rate(t)=sum(rates(t))
  dt = next_event_time(rate, max_rate, rng)

  next_index = 1
  cumulated_rates = cumsum(rates(dt))
  sum_rate = cumulated_rates[end]

  theta = rand(rng) * sum_rate
  id = 1
  #catch lower-bound case that cannot be reached by binary search
  if theta >= cumulated_rates[1]
    #binary search
    index_l = 1
    index_r = length(cumulated_rates)
    while index_l < index_r-1
      #index_m = floor(Int,(index_l+index_r)/2)
      index_m = fld(index_l+index_r, 2)
      if cumulated_rates[index_m] < theta
        index_l = index_m
      else
        index_r = index_m
      end
    end
    id = index_r
  end

  return dt, id
end

end

export InhomogeneousPoissonProcess
