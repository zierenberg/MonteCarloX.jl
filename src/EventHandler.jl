# EventHandler

abstract type AbstractEventHandler{T} end
abstract type AbstractEventHandlerTime{T} <: AbstractEventHandler{T} end
abstract type AbstractEventHandlerRate{T} <: AbstractEventHandler{T} end

###############################################################################
###############################################################################
###############################################################################

"""
The simples event list that is completely static
"""
mutable struct ListEventRateSimple{T}<:AbstractEventHandlerRate{T}
  list_event::Vector{T}          #static list of events
  list_rate::ProbabilityWeights  #static list of rates
  threshold_min_rate::Float64
  noevent::T

  function ListEventRateSimple{T}(list_event::Vector{T}, list_rate_::Vector{Float64}, threshold_min_rate::Float64, noevent::T) where T
    list_rate = ProbabilityWeights(list_rate_)
    new(list_event, list_rate, threshold_min_rate, noevent)
  end
end

function num_events(event_handler::ListEventRateSimple{T}) where T
  if sum(event_handler.list_rate) < event_handler.threshold_min_rate
    return 0
  else
    return length(event_handler.list_rate)
  end
end

function set!(event_handler::ListEventRateSimple{T}, index_event::Int, rate::Float64) where T
  #this is of type ProbabilityWeights that automatically updates the sum over the list of rates
  event_handler.list_rate[index_event] = rate
end


###############################################################################
###############################################################################
###############################################################################

mutable struct ListEventRateActiveMask{T}<:AbstractEventHandlerRate{T}
  list_event::Vector{T}           #static list of events
  list_rate::ProbabilityWeights   #static list of rates (includes updates sum)
  threshold_active::Float64
  list_active::Vector{Bool} 
  internal_num_active::Int
  index_first_active::Int
  index_last_active::Int
  noevent::T

  function ListEventRateActiveMask{T}(list_event::Vector{T}, list_rate_::Vector{Float64}, threshold_active::Float64, noevent::T; initial::String="all_active") where T
    list_rate = ProbabilityWeights(list_rate_)
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
    new(list_event, list_rate, threshold_active, list_active, num_active, index_first_active, index_last_active, noevent)
    #new(list_event, list_rate, threshold_active, list_active, num_active, index_first_active, index_last_active, noevent)
  end
end

function set!(event_handler::ListEventRateActiveMask, index::Int, rate::Float64)
  if event_handler.list_active[index]
    if rate > event_handler.threshold_active
      event_handler.list_rate[index] = rate 
    else
      event_handler.list_rate[index] = 0 
      deactivate!(event_handler, index)
    end
  else #not active
    if rate > event_handler.threshold_active
      event_handler.list_rate[index] = rate 
      activate!(event_handler, index)
    else
      event_handler.list_rate[index] = 0 
    end
  end
end

function num_events(event_handler::ListEventRateActiveMask)
  return event_handler.internal_num_active
end

####################################### for internal use only
#
function first_event(event_handler::ListEventRateActiveMask)
  return event_handler.index_first_active
end

function last_event(event_handler::ListEventRateActiveMask)
  return event_handler.index_last_active
end

function next_event(event_handler::ListEventRateActiveMask, index::Int)::Int
  while index < length(event_handler.list_rate) - 1
    index += 1
    if event_handler.list_active[index]
      return index
    end
  end
  return length(event_handler.list_active)+1
end

function previous_event(event_handler::ListEventRateActiveMask, index::Int)::Int
  while index > 2
    index -= 1
    if event_handler.list_active[index]
      return index
    end
  end
  return 0
end

function activate!(event_handler::ListEventRateActiveMask, index::Int)
  event_handler.list_active[index] = true
  event_handler.internal_num_active += 1
  if index < event_handler.index_first_active
    event_handler.index_first_active = index
  end
  if index > event_handler.index_last_active
    event_handler.index_last_active = index
  end
end

function deactivate!(event_handler::ListEventRateActiveMask, index::Int)
  event_handler.list_active[index] = false
  event_handler.internal_num_active -= 1
  if index == event_handler.index_first_active
    event_handler.index_first_active = next_event(event_handler, index)
  end
  if index == event_handler.index_last_active
    event_handler.index_last_active = previous_event(event_handler, index)
  end
end

###############################################################################
###############################################################################
###############################################################################
#mutable struct EventQueue <:AbstractEventHandlerTime
#TODO: needs proper implementation of pointer list that allows insertion; develop here...
#end
