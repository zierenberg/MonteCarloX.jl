# KineticMonteCarlo
struct KineticMonteCarlo end

"""
    next(rng::AbstractRNG, weights::AbstractWeights)::Tuple{Float64,Int}

Next stochastic event (`\\Delta t`, index) drawn proportional to probability given in `weights`
"""
function next(alg::KineticMonteCarlo, rng::AbstractRNG, weights::AbstractWeights)::Tuple{Float64,Int}
  dtime = next_time(rng, sum(weights))
  index = next_event(rng, weights)
  return dtime, index
end

"""
    next(rng::AbstractRNG, event_handler::AbstractEventHandlerRate)

Next stochastic event (`\\Delta t`, event type) organized by `event_handler`
fast(to be tested, depends on overhead of EventList) implementation of next_event_rate if defined by EventList object
"""
function next(alg::KineticMonteCarlo, rng::AbstractRNG, event_handler::AbstractEventHandlerRate)
  dt = next_time(rng, sum(event_handler.list_rate))
  id = next_event(rng, event_handler)
  return dt,id
end

"""
    next_time(rng::AbstractRNG, rate::Float64)::Float64

Next stochastic `\\Delta t` for Poisson process with `rate`
"""
function next_time(rng::AbstractRNG, rate::Float64)::Float64
  return randexp(rng)/rate
end

"""
    next_event(rng::AbstractRNG, cumulated_rates::Vector{T})::Int where {T<:AbstractFloat}

Select a single random index in `1:length(cumulated_rates)` with cumulated probability given in `cumulated_rates`.

#Remarks
Deprecated unless we find a good data structure for (dynamic) cumulated weights
"""
function next_event(rng::AbstractRNG, cumulated_rates::Vector{T})::Int where {T<:AbstractFloat}
  theta = rand(rng)*cumulated_rates[end]
  index = MonteCarloX.binary_search(cumulated_rates,theta)
  return index
end

"""
    next_event(rng::AbstractRNG, weights::AbstractWeights)::Int 

Select a single random index in `1:length(weights)` with probability proportional to the entry in `weights`.

#Remarks
This is on average twice as fast as StatsBase.sampling because it can iterate from either beginning or end of weights
"""
function next_event(rng::AbstractRNG, weights::AbstractWeights)::Int 
  theta = rand(rng)*weights.sum
  N = length(weights)

  #this is for extra performance in case of large lists (factor of 2)
  if theta < 0.5*weights.sum
    index = 1
    cumulated_rates = weights[index]
    while cumulated_rates < theta && index < N
      index += 1
      @inbounds cumulated_rates += weights[index]
    end
    return index

  else
    index = N
    cumulated_rates_lower = weights.sum - weights[index]
    #cumulated_rates in this case belong to index-1
    while cumulated_rates_lower > theta && index > 1
      index -= 1;
      @inbounds cumulated_rates_lower -= weights[index]
    end
    return index

  end
end

#relevent to write this as extra function this for performance issues (compared to keyword argument with preset value on GLOBAL_RNG)
next_event(list_rates::AbstractWeights) = next_event(Random.GLOBAL_RNG, list_rates) 


"""
    next_event(rng::AbstractRNG, event_handler::AbstractEventHandlerRate{T})::T where T

Select a single random event with a given probability managed by `event_handler`.

The `event_handler` also manages the case that no valid events are left (e.g.
when all rates are equal to zero). This becomes relevant when working with the
advance!(alg::Gillespie) wrapper to advance time for a longer period.
"""
function next_event(rng::AbstractRNG, event_handler::AbstractEventHandlerRate{T})::T where T
  ne = length(event_handler)
  if ne > 1 
    theta::Float64 = rand(rng)*sum(event_handler.list_rate)
    index_last = last_index(event_handler) 
    index_first = first_index(event_handler)

    if theta < 0.5*sum(event_handler.list_rate)
      index = index_first 
      cumulated_rates = event_handler.list_rate[index]
      while cumulated_rates < theta && index < index_last
        index = next_index(event_handler, index)
        @inbounds cumulated_rates += event_handler.list_rate[index]
      end
    else
      index = index_last
      cumulated_rates_lower = sum(event_handler.list_rate) - event_handler.list_rate[index]
      #cumulated_rates in this case belong to id-1
      while cumulated_rates_lower > theta && index > index_first
        index = previous_index(event_handler, index) 
        @inbounds cumulated_rates_lower -= event_handler.list_rate[index]
      end
    end
    return index
  elseif ne == 1
    return index = first_index(event_handler)
  else
    return event_handler.noevent 
  end
end

function next_event(rng::AbstractRNG, event_handler::ListEventRateSimple{T})::T where T
  if length(event_handler) > 0
    index = next_event(rng, event_handler.list_rate)
    return event_handler.list_event[index] 
  else
    return event_handler.noevent
  end
end
