@doc """
    EventQueue{T}([start_time::Float64])

Ordered time-event queue storing `(time, event)` tuples.

# API implemented
- `length(event_handler)`
- `getindex(event_handler, index)`
- `popfirst!(event_handler)`
- `add!(event_handler, tuple_time_event)`
- `get_time(event_handler)`
- `set_time!(event_handler, time)`
"""
mutable struct EventQueue{T} <: AbstractEventHandlerTime{T}
    sorted_list::Vector{Tuple{Float64,T}}
    time::Float64

    function EventQueue{T}(start_time::Number) where T
        sorted_list = Vector{Tuple{Float64,T}}()
        new(sorted_list, Float64(start_time))
    end
end

EventQueue{T}() where T = EventQueue{T}(0.0)

function get_time(event_handler::EventQueue)
    return event_handler.time
end

function set_time!(event_handler::EventQueue, time::Number)
    event_handler.time = Float64(time)
end

function Base.length(event_handler::EventQueue)
    return length(event_handler.sorted_list)
end

function Base.getindex(event_handler::EventQueue, index::Int64)
    if 0 < index <= length(event_handler.sorted_list)
        return event_handler.sorted_list[index]
    else
        return nothing
    end
end

function Base.popfirst!(event_handler::EventQueue)
    first_event = first(event_handler.sorted_list)
    deleteat!(event_handler.sorted_list, 1)
    return first_event
end

function binary_search_insert_position(list::Vector{Tuple{Float64,T}}, time::Float64) where T
    left = 1
    right = length(list)

    while left <= right
        mid = (left + right) >>> 1
        mid_time = first(list[mid])
        if mid_time < time
            left = mid + 1
        else
            right = mid - 1
        end
    end

    return left
end

function add!(event_handler::EventQueue, tuple_time_event::Tuple{Float64,T}) where T
    time = first(tuple_time_event)
    if time < event_handler.time
        throw(ErrorException("added event to queue with time in the past"))
    end

    if isempty(event_handler.sorted_list)
        push!(event_handler.sorted_list, tuple_time_event)
        return event_handler
    end

    if time <= first(event_handler.sorted_list[1])
        pushfirst!(event_handler.sorted_list, tuple_time_event)
    elseif time >= first(event_handler.sorted_list[end])
        push!(event_handler.sorted_list, tuple_time_event)
    else
        position = binary_search_insert_position(event_handler.sorted_list, time)
        insert!(event_handler.sorted_list, position, tuple_time_event)
    end

    return event_handler
end
