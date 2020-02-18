"""
Module that manages events of different types (times, rates, probabilities)
"""
module EventHandler

#TOOD: check how to properly export functions like set! (or add! in Histograms)
#TODO: standardize all EventHanderRates

abstract type AbstractEventHandler end
abstract type AbstractEventHandlerTime <: AbstractEventHandler end
abstract type AbstractEventHandlerRate <: AbstractEventHandler end
abstract type AbstractEventHandlerProbability <: AbstractEventHandler end

#TODO: rename
# EventDict -> DictEventRate
# SimpleEventList -> ListEventRateSimple
# ActiveEventList -> ListEventRateActive
# ActiveEventListSorted -> ListEventRateActiveSorted
# MaskedEventList -> ListEventRateActiveMask

###############################################################################
###############################################################################
###############################################################################
mutable struct EventDict{T}<:AbstractEventHandlerRate
  internal_num_updates::Int
  dict_event_rate::Dict{T,Float64} 
  sum_rates::Float64 
  noevent::T
  function EventDict{T}(list_event::Vector{T}, list_rate::Vector{Float64}, noevent::T) where T
    dict_event_rate = Dict(list_event .=> list_rate)
    sum_rates = sum(values(dict_event_rate))
    new(0, dict_event_rate, sum_rates, noevent)
  end
end

function num_events(event_handler::EventDict{T}) where T
  return length(event_handler.dict_event_rate)
end

function set!(event_handler::EventDict, event::T, rate::Float64) where T
  rate_old = get(event_handler.dict_event_rate, event, 0.0)
  event_handler.dict_event_rate[event] = rate 
  event_handler.sum_rates += rate - rate_old
end

function remove!(event_handler::EventDict, event::T) where T
  rate_old = get(event_handler.dict_event_rate, event, 0) 
  delete!(event_handler.dict_event_rate, event)
  event_handler.sum_rates -= rate_old 
end

###############################################################################
###############################################################################
###############################################################################

mutable struct ActiveEventListSorted<:AbstractEventHandlerRate
  internal_num_updates::Int
  list_rate::Vector{Float64} 
  sum_rates::Float64 
  threshold_active::Float64
  list_active::Vector{Bool} 
  list_sorted_active_index::Vector{Int}

  function ActiveEventListSorted(list_rate::Vector{Float64}, threshold_active::Float64, initial::String="all_active")
    sum_rates = sum(list_rate)
    if initial=="all_active"
      list_sorted_active_index = [i for i=1:length(list_rate)]
      list_active = [true for i=1:length(list_rate)]
    elseif initial=="all_inactive"
      list_sorted_active_index = []
      list_active = [false for i=1:length(list_rate)]
    else 
      throw(UndefVarError(:initial))
    end
    new(0, list_rate, sum_rates, threshold_active, list_active, list_sorted_active_index)
  end
end

function num_events(event_handler::ActiveEventListSorted)
  return length(event_handler.list_sorted_active_index)
end

function set!(event_handler::ActiveEventListSorted, index::Int, rate::Float64)
  old_rate = event_handler.list_rate[index]
  event_handler.sum_rates += rate - old_rate 
  event_handler.list_rate[index] = rate 
  if event_handler.list_active[index]
    if !(rate > event_handler.threshold_active)
      deactivate!(event_handler, index)
    end
  else
    if rate > event_handler.threshold_active
      activate!(event_handler, index)
    end
  end
end

#for internal use only
function activate!(event_handler::ActiveEventListSorted, index::Int)
  i = MonteCarloX.binary_search(event_handler.list_sorted_active_index, index)
  insert!(event_handler.list_sorted_active_index, i, index)
  event_handler.list_active[index] = true
end

function deactivate!(event_handler::ActiveEventListSorted, index::Int)
  i = MonteCarloX.binary_search(event_handler.list_sorted_active_index, index)
  deleteat!(event_handler.list_sorted_active_index, i)
  event_handler.list_active[index] = false
end

###############################################################################
###############################################################################
###############################################################################

"""
The simples event list that is completely static
"""
mutable struct SimpleEventList{T}<:AbstractEventHandlerRate
  list_event::Vector{T}           #static list of events
  list_rate::Vector{Float64}      #static list of rates
  sum_rates::Float64 
  threshold_min_rate::Float64
  noevent::Int

  #maybe at some point rewrite as general event list
  #maybe with preallocated array for empty list of rates....
  function SimpleEventList{T}(list_event::Vector{T}, list_rate::Vector{Float64}, threshold_min_rate::Float64, noevent::T) where T
    sum_rates = sum(list_rate)
    new(list_event, list_rate, sum_rates, threshold_min_rate, noevent)
  end
end

function num_events(event_handler::SimpleEventList{T}) where T
  if event_handler.sum_rates < event_handler.threshold_min_rate
    return 0
  else
    return length(event_handler.list_rate)
  end
end

function set!(event_handler::SimpleEventList{T}, index_event::Int, rate::Float64) where T
  event_handler.sum_rates -= event_handler.list_rate[index_event]
  event_handler.list_rate[index_event] = rate
  event_handler.sum_rates += rate 
end

###############################################################################
###############################################################################
###############################################################################


"""
This should not produce the identical long-term behavior as MaskedList and SortedList because in set! the event-rates are not remaining in a sorted order...
"""
mutable struct ActiveEventList<:AbstractEventHandlerRate
  internal_num_updates::Int
  #list_event::Vector{T} or Dict
  list_event_to_rate::Vector{Int} #static list pointing to active rates (zero if not active)
  list_rate_to_event::Vector{Int} #dynamic list pointing back to events 
  list_rate::Vector{Float64}      #dynamic list of active rates
  sum_rates::Float64 
  threshold_active::Float64
  noevent::Int

  #maybe at some point rewrite as general event list
  #maybe with preallocated array for empty list of rates....
  function ActiveEventList(list_event::Vector{T}, list_rate::Vector{Float64}, threshold_active::Float64) where T
    sum_rates = sum(list_rate)
    list_event_to_rate = [i for i=1:length(list_event)]
    list_rate_to_event = [i for i=1:length(list_event)]
    new(0, list_event_to_rate, list_rate_to_event, list_rate, sum_rates, threshold_active, 0)
  end
end

function num_events(event_handler::ActiveEventList)
  return length(event_handler.list_rate)
end

function set!(event_handler::ActiveEventList, index_event::Int, rate::Float64)
  index_rate = event_handler.list_event_to_rate[index_event]
  if index_rate > 0 # active
    event_handler.sum_rates -= event_handler.list_rate[index_rate]
    if (rate > event_handler.threshold_active)
      event_handler.list_rate[index_rate] = rate 
      event_handler.sum_rates += rate
    else
      #deactivate
      if index_rate == length(event_handler.list_rate)
        event_handler.list_event_to_rate[index_event] = 0
        pop!(event_handler.list_rate)
        pop!(event_handler.list_rate_to_event)
      else
        event_handler.list_event_to_rate[index_event] = 0
        rate_end = pop!(event_handler.list_rate)
        index_event_end = pop!(event_handler.list_rate_to_event)
        event_handler.list_rate[index_rate] = rate_end 
        event_handler.list_rate_to_event[index_rate] = index_event_end
        event_handler.list_event_to_rate[index_event_end] = index_rate
      end
    end
  else #inactive
    if rate > event_handler.threshold_active
      #activate
      push!(event_handler.list_rate, rate)
      push!(event_handler.list_rate_to_event, index_event)
      event_handler.list_event_to_rate[index_event] = length(event_handler.list_rate)
      event_handler.sum_rates += rate
    end
  end
end

###############################################################################
###############################################################################
###############################################################################

mutable struct MaskedEventList<:AbstractEventHandlerRate
  internal_size::Int
  internal_num_active::Int
  list_rate::Vector{Float64} 
  sum_rates::Float64 
  threshold_active::Float64
  list_active::Vector{Bool} 
  index_first_active::Int
  index_last_active::Int

  function MaskedEventList(list_rate::Vector{Float64}, threshold_active::Float64, initial::String="all_active")
    sum_rates = sum(list_rate)
    if initial=="all_active"
      num_active = length(list_rate)
      list_sorted_active_index = [i for i=1:length(list_rate)]
      list_active = [true for i=1:length(list_rate)]
      index_first_active = 1
      index_last_active = length(list_active)
    elseif initial=="all_inactive"
      num_active = 0
      list_sorted_active_index = []
      list_active = [false for i=1:length(list_rate)]
      index_first_active = length(list_active) + 1
      index_last_active = 0
    else 
      throw(UndefVarError(:initial))
    end
    new(length(list_rate), num_active, list_rate, sum_rates, threshold_active, list_active, index_first_active, index_last_active)
  end
end

function num_events(event_handler::MaskedEventList)
  return event_handler.internal_num_active
end

function first_active(event_handler::MaskedEventList)
  return event_handler.index_first_active
end

function last_active(event_handler::MaskedEventList)
  return event_handler.index_last_active
end

function next_active(event_handler::MaskedEventList, index::Int)::Int
  while index < event_handler.internal_size - 1
    index += 1
    if event_handler.list_active[index]
      return index
    end
  end
  return length(event_handler.list_active)+1
end

function previous_active(event_handler::MaskedEventList, index::Int)::Int
  while index > 2
    index -= 1
    if event_handler.list_active[index]
      return index
    end
  end
  return 0
end

function set!(event_handler::MaskedEventList, index::Int, rate::Float64)
  if event_handler.list_active[index]
    old_rate = event_handler.list_rate[index]
    event_handler.sum_rates -= old_rate 
    if rate > event_handler.threshold_active
      event_handler.list_rate[index] = rate 
      event_handler.sum_rates += rate
    else
      event_handler.list_rate[index] = 0 
      deactivate!(event_handler, index)
    end
  else #not active
    if rate > event_handler.threshold_active
      event_handler.list_rate[index] = rate 
      event_handler.sum_rates += rate
      activate!(event_handler, index)
    else
      event_handler.list_rate[index] = 0 
    end
  end
end

#for internal use only
function activate!(event_handler::MaskedEventList, index::Int)
  event_handler.list_active[index] = true
  event_handler.internal_num_active += 1
  if index < event_handler.index_first_active
    event_handler.index_first_active = index
  end
  if index > event_handler.index_last_active
    event_handler.index_last_active = index
  end
end

function deactivate!(event_handler::MaskedEventList, index::Int)
  event_handler.list_active[index] = false
  event_handler.internal_num_active -= 1
  if index == event_handler.index_first_active
    event_handler.index_first_active = next_active(event_handler, index)
  end
  if index == event_handler.index_last_active
    event_handler.index_last_active = previous_active(event_handler, index)
  end
end

###############################################################################
###############################################################################
###############################################################################
#mutable struct EventQueue <:AbstractEventHandlerTime
#TODO: needs proper implementation of pointer list that allows insertion; develop here...
#end
 
export
  AbstractEventHandler,
  EventDict,
  ActiveEventList,
  ActiveEventListSorted,
  SimpleEventList, 
  MaskedEventList,

  #methods
  set!,
  num_events,
  last_active,
  first_active,
  previous_active,
  next_active

end
export EventHandler

