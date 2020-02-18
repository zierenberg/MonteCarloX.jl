"""
# MonteCarloX.KineticMonteCarlo

Module that allows to implement (rejection-free) kinetic Monte Carlo algorithms in general
"""
module KineticMonteCarlo
using Random
using Distributions

using ..MonteCarloX
using ..EventHandler

###############################################################################
###############################################################################
###############################################################################


"""
generate events (dt,id) from a list of rates such that their occurence corresponds with their rate
"""
function next_event(list_rates::Vector{T},rng::AbstractRNG)::Tuple{Float64,Int} where {T<:AbstractFloat}
  cumulated_rates = cumsum(list_rates)
  sum_rates = cumulated_rates[end]
  dtime = next_event_time(sum_rates,rng)
  index = next_event_index(cumulated_rates, rng)
  return dtime, index
end

"""
fast implementation of next_event_rate if sum(list_rates) is known
"""
function next_event(list_rates::Vector{T}, sum_rates::T, rng::AbstractRNG)::Tuple{T,Int} where T<:AbstractFloat
  dtime = next_event_time(sum_rates, rng)
  index = next_event_index(list_rates, sum_rates, rng)
  return dtime,index
end

"""
fast(to be tested, depends on overhead of EventList) implementation of next_event_rate if defined by EventList object
"""
function next_event(event_handler::AbstractEventHandlerRate, rng::AbstractRNG)
  dt = next_event_time(event_handler.sum_rates, rng)
  id = next_event_id(event_handler, rng)
  return dt,id
end

"""
next event time for Poisson process with given rate

# Arguments
"""
function next_event_time(rate::Float64, rng::AbstractRNG)::Float64
  return rand(rng,Exponential(1.0/rate))
end

"""
 next event id for list of stochastic events describes by cumulated rates vector
"""
function next_event_index(cumulated_rates::Vector{T},rng::AbstractRNG)::Int where {T<:AbstractFloat}
  theta = rand(rng)*cumulated_rates[end]
  index = MonteCarloX.binary_search(cumulated_rates,theta)
  return index
end

"""
 next event index for list of stochastic events described by occurence rates and the sum over those
"""
function next_event_index(list_rates::Vector{T}, sum_rates::T,rng::AbstractRNG)::Int where {T<:AbstractFloat}
  theta = rand(rng)*sum_rates

  if theta < 0.5*sum_rates
    index = 1
    cumulated_rates = list_rates[index]
    while cumulated_rates < theta
      index += 1
      cumulated_rates += list_rates[index]
    end
  else
    index = length(list_rates);
    cumulated_rates_lower = sum_rates - list_rates[index]
    #cumulated_rates in this case belong to index-1
    while cumulated_rates_lower > theta
      index -= 1;
      cumulated_rates_lower -= list_rates[index]
    end
  end

  return index
end

"""
 next event id for list of stochastic events described by occurence rates and the sum over those
"""
function next_event_id(event_handler::SimpleEventList{T}, rng::AbstractRNG)::T where {T}
  if event_handler.sum_rates > event_handler.threshold_min_rate
    index = next_event_index(event_handler.list_rate, event_handler.sum_rates, rng)
    return event_handler.list_event[index] 
  else
    return event_handler.noevent
  end
end

"""
next event id for dictionary of events (type T) and corresponding rates
TODO: this needs performance testing and should be identical to optimal versiob two with masked lists
"""
function next_event_id(event_handler::EventDict{T},rng::AbstractRNG)::T where {T}
  ne = num_events(event_handler)
  if ne > 1 
    theta::Float64 = rand(rng)*event_handler.sum_rates
    cumulated_rates = 0
    for (id, rate) in event_handler.dict_event_rate
      cumulated_rates += rate
      if ! (cumulated_rates < theta)
        return id
      end
    end
  elseif ne == 1
    return first(keys(event_handler.dict_event_rate))
  else
    return event_handler.noevent 
  end
end

"""
 next event id for list of stochastic events described by occurence rates and the sum over those
 #TODO: needs unit test!!!
 #ALTERNATIVE: DROP THE INDEX LIST AND JUST DO THE BOOL LIST: simply jump over false arrays
"""
#TODO: Maybe this can be unified with function for MaskedEventList if index is obtained by some first, next... functions
# this could be the default AbstractEventHandler solution
function next_event_id(event_handler::ActiveEventListSorted,rng::AbstractRNG)::Int
  ne = num_events(event_handler)
  if ne > 1 
    theta = rand(rng)*event_handler.sum_rates
    if theta < 0.5*event_handler.sum_rates
      i = 1
      index = event_handler.list_sorted_active_index[i]
      cumulated_rates = event_handler.list_rate[index]
      while cumulated_rates < theta
        i += 1
        index = event_handler.list_sorted_active_index[i]
        cumulated_rates += event_handler.list_rate[index]
      end
    else
      i = length(event_handler.list_sorted_active_index);
      index = event_handler.list_sorted_active_index[i]
      cumulated_rates_lower = event_handler.sum_rates - event_handler.list_rate[index]
      #cumulated_rates in this case belong to id-1
      while cumulated_rates_lower > theta
        i -= 1
        index = event_handler.list_sorted_active_index[i]
        cumulated_rates_lower -= event_handler.list_rate[index]
      end
    end
    return index
  elseif ne == 1
    return event_handler.list_sorted_active_index[1]
  else
    return event_handler.noevent 
  end
end

function next_event_id(event_handler::ActiveEventList,rng::AbstractRNG)::Int
  ne = num_events(event_handler)
  if ne > 1 
    index = next_event_index(event_handler.list_rate, event_handler.sum_rates, rng)
    return event_handler.list_rate_to_event[index]
  elseif ne == 1
    return event_handler.list_rate_to_event[1]
  else
    return event_handler.noevent 
  end
end

function next_event_id(event_handler::MaskedEventList,rng::AbstractRNG)::Int
  ne = num_events(event_handler)
  if ne > 1 
    theta::Float64 = rand(rng)*event_handler.sum_rates

    if theta < 0.5*event_handler.sum_rates
      index = first_active(event_handler)
      cumulated_rates = event_handler.list_rate[index]
      while cumulated_rates < theta
        index = next_active(event_handler, index)
        cumulated_rates += event_handler.list_rate[index]
      end
    else
      index = last_active(event_handler) 
      cumulated_rates_lower = event_handler.sum_rates - event_handler.list_rate[index]
      #cumulated_rates in this case belong to id-1
      while cumulated_rates_lower > theta
        index = previous_active(event_handler, index) 
        cumulated_rates_lower -= event_handler.list_rate[index]
      end
    end
    return index
  elseif ne == 1
    return index = next_active(event_handler, 0)
  else
    return event_handler.noevent 
  end
end




end
export KineticMonteCarlo
