# EventHandler
#
# TODO: explicit copy in constructor!

abstract type AbstractEventHandler{T} end
abstract type AbstractEventHandlerTime{T} <: AbstractEventHandler{T} end
abstract type AbstractEventHandlerRate{T} <: AbstractEventHandler{T} end

function Base.getindex(event_handler::AbstractEventHandlerRate, index::Int64)
    return event_handler.list_rate[index]
end


function reset_sum_rate(event_handler::AbstractEventHandlerRate)
    event_handler.list_rate.sum = 0.0
    for i = 1:length(event_handler.list_rate)
        @inbounds event_handler.list_rate.sum += event_handler.list_rate[i]
    end
end


###############################################################################
###############################################################################
###############################################################################
@doc """
    ListEventRateSimple{T}

Simplest event manager for a list of events of type T with a static list of rates

#API implemented:
- length(event_handler)
- getindex(event_handler, index)
- setindex!(event_handler, value, index)
"""
mutable struct ListEventRateSimple{T} <: AbstractEventHandlerRate{T}
    list_event::Vector{T}                                                                                        # static list of events
    list_rate::ProbabilityWeights{Float64,Float64,Vector{Float64}}    # static list of rates
    threshold_active::Float64
    num_active::Int
    check_sum::Int64

    function ListEventRateSimple{T}(list_event::Vector{T}, list_rate_::Vector{Float64}, threshold_active::Float64) where T
        list_rate = ProbabilityWeights(list_rate_)
        num_active = count(x->x > threshold_active, list_rate)
        new(list_event, list_rate, threshold_active, num_active, 0)
    end
end

function Base.length(event_handler::ListEventRateSimple)
    return event_handler.num_active
end

function Base.setindex!(event_handler::ListEventRateSimple, rate::Float64, index::Int64)
    if rate > event_handler.threshold_active 
        if !(event_handler.list_rate[index] > event_handler.threshold_active)
            event_handler.num_active += 1
        end
    else
        if event_handler.list_rate[index] > event_handler.threshold_active
            event_handler.num_active -= 1
        end
    end

    # this is of type ProbabilityWeights that automatically updates the sum over the list of rates
    event_handler.list_rate[index] = rate

    # update sum(rates) every 1e6 (heuristic) changes in rates
    event_handler.check_sum += 1
    if event_handler.check_sum > 1e6
        reset_sum_rate(event_handler)
        event_handler.check_sum = 0
    end
end

###############################################################################
###############################################################################
###############################################################################
@doc """
    ListEventRateActiveMask{T}

"""
mutable struct ListEventRateActiveMask{T} <: AbstractEventHandlerRate{T}
    list_event::Vector{T}                                            # static list of events
    list_rate::ProbabilityWeights{Float64,Float64,Vector{Float64}}  # static list of rates
    threshold_active::Float64
    list_active::Vector{Bool} 
    num_active::Int
    index_first_active::Int
    index_last_active::Int
    check_sum::Int64

    function ListEventRateActiveMask{T}(list_event::Vector{T}, list_rate_::Vector{Float64}, threshold_active::Float64; initial::String = "all_active") where T
        list_rate = ProbabilityWeights(list_rate_)
        if initial == "all_active"
            num_active = length(list_rate)
            list_sorted_active_index = [i for i = 1:length(list_rate)]
            list_active = [true for i = 1:length(list_rate)]
            index_first_active = 1
            index_last_active = length(list_active)
        elseif initial == "all_inactive"
            num_active = 0
            list_sorted_active_index = []
            list_active = [false for i = 1:length(list_rate)]
            index_first_active = length(list_active) + 1
            index_last_active = 0
        else 
            throw(UndefVarError(:initial))
        end
        new(list_event, list_rate, threshold_active, list_active, num_active, index_first_active, index_last_active, 0)
    end
end

function Base.length(event_handler::ListEventRateActiveMask)
    return event_handler.num_active
end

"""
    setindex!(event_handler::ListEventRateActiveMask, rate::Float64, index_event::Int64)
"""
function Base.setindex!(event_handler::ListEventRateActiveMask, rate::Float64, index::Int64)
    if event_handler.list_active[index]
        if rate > event_handler.threshold_active
            event_handler.list_rate[index] = rate 
        else
            event_handler.list_rate[index] = 0 
            deactivate!(event_handler, index)
        end
    else # not active
        if rate > event_handler.threshold_active
            event_handler.list_rate[index] = rate 
            activate!(event_handler, index)
        else
            event_handler.list_rate[index] = 0 
        end
    end

    # update sum(rates) every 1e6 (heuristic) changes in rates
    event_handler.check_sum += 1
    if event_handler.check_sum > 1e6
        reset_sum_rate(event_handler)
        event_handler.check_sum = 0
    end
end


####################################### for internal use only
#
function first_index(event_handler::ListEventRateActiveMask)
    return event_handler.index_first_active
end

function last_index(event_handler::ListEventRateActiveMask)
    return event_handler.index_last_active
end

function next_index(event_handler::ListEventRateActiveMask, index::Int)::Int
    while index < length(event_handler.list_rate) - 1
        index += 1
        if event_handler.list_active[index]
            return index
        end
    end
    return length(event_handler.list_active) + 1
end

function previous_index(event_handler::ListEventRateActiveMask, index::Int)::Int
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
    event_handler.num_active += 1
    if index < event_handler.index_first_active
        event_handler.index_first_active = index
    end
    if index > event_handler.index_last_active
        event_handler.index_last_active = index
    end
end

function deactivate!(event_handler::ListEventRateActiveMask, index::Int)
    event_handler.list_active[index] = false
    event_handler.num_active -= 1
    if index == event_handler.index_first_active
        event_handler.index_first_active = next_index(event_handler, index)
    end
    if index == event_handler.index_last_active
        event_handler.index_last_active = previous_index(event_handler, index)
    end
end

###############################################################################
###############################################################################
###############################################################################
@doc """
    EventQueue{T}([start_time::FLoat64])

Flexible event queue that stores an ordered list of (time, event) tuples 

#API implemented:
- length(event_handler) 
- getindex(event_handler, index)
- popfirst!(event_handler)
- add!(event_handler, time, event)
"""
mutable struct EventQueue{T} <: AbstractEventHandlerTime{T}
    sorted_list::LinkedList{Tuple{Float64,T}}
    time::Float64

    function EventQueue{T}(start_time::Number) where T
        sorted_list = LinkedList{Tuple{Float64,T}}()
        new(sorted_list, Float64(start_time))
    end
end
EventQueue{T}() where T = EventQueue{T}(0.0)

function get_time(event_handler::EventQueue)
    return event_handler.time
end

function set_time!(event_handler::EventQueue, time::Number)
    event_handler.time = time
end

function Base.length(event_handler::EventQueue)
    return length(event_handler.sorted_list)
end

function Base.getindex(event_handler::EventQueue, index::Int64)
    if 0 < index <= length(event_handler.sorted_list)
        return event_handler.sorted_list[positiontoindex(index, event_handler.sorted_list)]
    else
        return nothing
    end
end

# TODO: would require to resort list obviously
#function Base.setindex!(event_handler::EventQueue, tuple_time_event::Tuple{Float64,T}, index::Int64) where T
#    setindex!(event_handler.sorted_list, positiontoindex(index, event_handler.sorted_list), tuple_time_event)
#end
#
#function Base.setindex!(event_handler::EventQueue, time::Float64, index::Int64) where T
#    pos = positiontoindex(index, event_handler.sorted_list)
#    event = event_handler.sorted_list[pos][2]
#    setindex!(event_handler.sorted_list, pos, (time,event))
#end

function popfirst!(event_handler::EventQueue)
    return popfirst!(event_handler.sorted_list)
end


function Base.getindex(list::LinkedLists.LinkedList{Tuple{Float64,T}}, index::Int) where T
    if 0 < index <= length(list)
        return first(list[positiontoindex(index, list)])
    else
        return nothing
    end
end

function add!(event_handler::EventQueue, tuple_time_event::Tuple{Float64, T}) where T
    time = first(tuple_time_event)
    if time < event_handler.time
        throw(ErrorException("added event to queue with time in the past"))
    end
    if length(event_handler) == 0
        push!(event_handler.sorted_list, tuple_time_event)
    elseif time <= first(event_handler[1])
        pushfirst!(event_handler.sorted_list, tuple_time_event)
    elseif time >= first(event_handler[length(event_handler)])
        push!(event_handler.sorted_list, tuple_time_event)
    else
        # find position in sorted list (if multiple same times, insert before first)
        position = binary_search(event_handler.sorted_list, time)
        insert!(event_handler.sorted_list, positiontoindex(position, event_handler.sorted_list), tuple_time_event)
    end
end
