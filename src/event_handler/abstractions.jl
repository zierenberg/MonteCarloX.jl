# Event handler abstractions
# Shared by all event-handler implementations used in continuous-time algorithms.

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
