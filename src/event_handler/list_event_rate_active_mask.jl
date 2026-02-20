@doc """
    ListEventRateActiveMask{T}

Event manager with explicit active-mask tracking for sparse active rates.
"""
mutable struct ListEventRateActiveMask{T} <: AbstractEventHandlerRate{T}
    list_event::Vector{T}
    list_rate::ProbabilityWeights{Float64,Float64,Vector{Float64}}
    threshold_active::Float64
    list_active::Vector{Bool}
    num_active::Int
    index_first_active::Int
    index_last_active::Int
    noevent::T
    check_sum::Int64

    function ListEventRateActiveMask{T}(
        list_event::Vector{T},
        list_rate_::Vector{Float64},
        threshold_active::Float64,
        noevent::T;
        initial::String = "all_active",
    ) where T
        list_rate = ProbabilityWeights(list_rate_)
        if initial == "all_active"
            num_active = length(list_rate)
            list_active = [true for _ = 1:length(list_rate)]
            index_first_active = 1
            index_last_active = length(list_active)
        elseif initial == "all_inactive"
            num_active = 0
            list_active = [false for _ = 1:length(list_rate)]
            index_first_active = length(list_active) + 1
            index_last_active = 0
        else
            throw(UndefVarError(:initial))
        end
        new(
            list_event,
            list_rate,
            threshold_active,
            list_active,
            num_active,
            index_first_active,
            index_last_active,
            noevent,
            0,
        )
    end
end

function Base.length(event_handler::ListEventRateActiveMask)
    return event_handler.num_active
end

function Base.setindex!(event_handler::ListEventRateActiveMask, rate::Float64, index::Int64)
    if event_handler.list_active[index]
        if rate > event_handler.threshold_active
            event_handler.list_rate[index] = rate
        else
            event_handler.list_rate[index] = 0
            deactivate!(event_handler, index)
        end
    else
        if rate > event_handler.threshold_active
            event_handler.list_rate[index] = rate
            activate!(event_handler, index)
        else
            event_handler.list_rate[index] = 0
        end
    end

    event_handler.check_sum += 1
    if event_handler.check_sum > 1e6
        reset_sum_rate(event_handler)
        event_handler.check_sum = 0
    end
end

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
