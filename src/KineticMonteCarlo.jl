"""
# MonteCarloX.KineticMonteCarlo

Module that allows to implement (rejection-free) kinetic Monte Carlo algorithms in general
"""
module KineticMonteCarlo
using Random
using Distributions

include("Utils.jl")

"""
next event time for Poisson process with given rate

# Arguments
"""
function next_event_time(rate::Float64, rng::AbstractRNG)::Float64
  return rand(rng,Exponential(1.0/rate))
end

#TODO: make more abstract such that Inhomogeneous Poisson Process can also use this
"""
 next event id for list of stochastic events describes by cumulated rates vector
"""
function next_event_id(cumulated_rates::Vector{T},rng::AbstractRNG)::Int where {T<:AbstractFloat}
  theta = rand(rng)*cumulated_rates[end]

  id = binary_search(cumulated_rates,theta)

  return id
end

"""
 next event id for list of stochastic events described by occurence rates and the sum over those
"""
function next_event_id(list_rates::Vector{T}, sum_rates::Float64,rng::AbstractRNG)::Int where {T<:AbstractFloat}
  id::Int        = 1
  theta::Float64 = rand(rng)*sum_rates

  if theta < 0.5*sum_rates
    cumulated_rates = list_rates[id]
    while cumulated_rates < theta
      id += 1
      cumulated_rates += list_rates[id]
    end
  else
    id = length(list_rates);
    cumulated_rates_lower = sum_rates - list_rates[id]
    #cumulated_rates in this case belong to id-1
    while cumulated_rates_lower > theta
      id -= 1;
      cumulated_rates_lower -= list_rates[id]
    end
  end

  return id
end


"""
generate events (dt,id) from a list of rates such that their occurence corresponds with their rate
"""
function next_event(list_rates::Vector{T},rng::AbstractRNG)::Tuple{Float64,Int} where {T<:AbstractFloat}
  cumulated_rates = cumsum(list_rates)
  sum_rates = cumulated_rates[end]
  dt = next_event_time(sum_rates,rng)
  id = next_event_id(cumulated_rates, rng)
  return dt, id
end

"""
fast implementation of next_event_rate if sum(list_rates) is known
"""
function next_event(list_rates::Vector{T}, sum_rates::T, rng::AbstractRNG)::Tuple{T,Int} where T<:AbstractFloat
  dt = next_event_time(sum_rates, rng)
  id = next_event_id(list_rates, sum_rates, rng)
  return dt,id
end

end
export KineticMonteCarlo
