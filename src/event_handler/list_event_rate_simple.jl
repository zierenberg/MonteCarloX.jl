@doc """
    ListEventRateSimple{T}

Simplest event manager for a list of events of type T with a static list of rates.

# API implemented
- length(event_handler)
- getindex(event_handler, index)
- setindex!(event_handler, value, index)
"""
mutable struct ListEventRateSimple{T} <: AbstractEventHandlerRate{T}
    list_event::Vector{T}
    list_rate::ProbabilityWeights{Float64,Float64,Vector{Float64}}
    threshold_active::Float64
    num_active::Int
    noevent::T
    check_sum::Int64

    function ListEventRateSimple{T}(
        list_event::Vector{T},
        list_rate_::Vector{Float64},
        threshold_active::Float64,
        noevent::T,
    ) where T
        list_rate = ProbabilityWeights(list_rate_)
        num_active = count(x -> x > threshold_active, list_rate)
        new(list_event, list_rate, threshold_active, num_active, noevent, 0)
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

    event_handler.list_rate[index] = rate

    event_handler.check_sum += 1
    if event_handler.check_sum > 1e6
        reset_sum_rate(event_handler)
        event_handler.check_sum = 0
    end
end
