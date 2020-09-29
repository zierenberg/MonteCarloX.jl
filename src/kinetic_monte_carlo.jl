# KineticMonteCarlo
struct KineticMonteCarlo end

"""
    next(alg::KineticMonteCarlo, rates::AbstractWeights, [rng::AbstractRNG])::Tuple{Float64,Int}

Next stochastic event (`\\Delta t`, index) drawn proportional to probability given in `rates`
"""
function next(alg::KineticMonteCarlo, rates::Union{AbstractWeights, AbstractVector}, rng::AbstractRNG = Random.GLOBAL_RNG)::Tuple{Float64,Int}
    sum_rates = sum(rates)
    if !(sum_rates > 0)
        return Inf, 0
    end
    dtime = next_time(sum_rates, rng)
    event = next_event(rates, rng)
    return dtime, event
end

"""
    next(alg::KineticMonteCarlo, event_handler::AbstractEventHandlerTime, [rng::AbstractRNG])::Tuple{Float64,Int}

Take next event from the event handler
"""
function next(alg::KineticMonteCarlo, event_handler::AbstractEventHandlerTime, rng::AbstractRNG = Random.GLOBAL_RNG)::Tuple{Float64,Int}
    if (length(event_handler) > 0)
        dtime, event = pop_first!(event_handler)
    else
        dtime, event = Inf, event_handler.noevent
    end
        
    return dtime, event
end

"""
    next(alg::KineticMonteCarlo, event_handler::AbstractEventHandlerRate, [rng::AbstractRNG])

Next stochastic event (`\\Delta t`, event type) organized by `event_handler`
fast(to be tested, depends on overhead of EventList) implementation of next_event_rate if defined by EventList object

TOOD: write this as wrapper around above..
"""
function next(alg::KineticMonteCarlo, event_handler::AbstractEventHandlerRate, rng::AbstractRNG = Random.GLOBAL_RNG)::Tuple{Float64,Int}
    if !(sum(event_handler.list_rate) > 0)
        return Inf, event_handler.noevent
    end
    dtime = next_time(sum(event_handler.list_rate), rng)
    event = next_event(event_handler, rng)
    return dtime, event
end

"""
    next_time(rate::Float64, [rng::AbstractRNG])::Float64

Next stochastic `\\Delta t` for Poisson process with `rate`
"""
function next_time(rate::Float64, rng::AbstractRNG = Random.GLOBAL_RNG)::Float64
    return randexp(rng) / rate
end

"""
    next_event(cumulated_rates::Vector{T}, rng::AbstractRNG)::Int where {T<:AbstractFloat}

Select a single random index in `1:length(cumulated_rates)` with cumulated probability given in `cumulated_rates`.

# Remarks
Deprecated unless we find a good data structure for (dynamic) cumulated rates
"""
function next_event_cumulative(cumulated_rates::Vector{T}, rng::AbstractRNG = Random.GLOBAL_RNG)::Int where {T <: AbstractFloat}
    theta = rand(rng) * cumulated_rates[end]
    index = MonteCarloX.binary_search(cumulated_rates, theta)
    return index
end

"""
    next_event(rates::AbstractWeights, [rng::AbstractRNG])::Int 

Select a single random index in `1:length(rates)` with probability proportional to the entry in `rates`.

# Remarks
This is on average twice as fast as StatsBase.sampling because it can iterate from either beginning or end of rates
"""
function next_event(rates::Union{AbstractWeights, AbstractVector}, rng::AbstractRNG = Random.GLOBAL_RNG)::Int 
    sum_rates = sum(rates)
    theta = rand(rng) * sum_rates

    # this is for extra performance in case of large lists (factor of 2)
    if theta < 0.5 * sum_rates
        index = 1
        cumulated_rates = rates[index]
        while cumulated_rates < theta && index < length(rates)
            index += 1
            @inbounds cumulated_rates += rates[index]
        end
        return index

    else
        index = length(rates)
        cumulated_rates_lower = sum_rates - rates[index]
        # cumulated_rates in this case belong to index-1
        while cumulated_rates_lower > theta && index > 1
            index -= 1
            @inbounds cumulated_rates_lower -= rates[index]
        end
        return index

    end
end

"""
    next_event(rates::MVector{2,T}, [rng::AbstractRNG])::Int where {T <: AbstractFloat}

Select next event for special case of two events only.
"""

function next_event(rates::MVector{2,T}, rng::AbstractRNG = Random.GLOBAL_RNG)::Int where {T <: AbstractFloat}
    if rand(rng) * (rates[1] + rates[2]) < rates[1]
        return 1
    else
        return 2
    end
end

"""
    next_event(event_handler::AbstractEventHandlerRate{T}, [rng::AbstractRNG])::T where T

Select a single random event with a given probability managed by `event_handler`.

The `event_handler` also manages the case that no valid events are left (e.g.
when all rates are equal to zero). This becomes relevant when using [`advance!`](@ref)
to advance for some time.

See also: [`advance!`](@ref)
"""
function next_event(event_handler::AbstractEventHandlerRate{T}, rng::AbstractRNG = Random.GLOBAL_RNG)::T where T
    ne = length(event_handler)
    if ne > 1 
        theta::Float64 = rand(rng) * sum(event_handler.list_rate)
        index_last = last_index(event_handler) 
        index_first = first_index(event_handler)

        if theta < 0.5 * sum(event_handler.list_rate)
            index = index_first 
            cumulated_rates = event_handler.list_rate[index]
            while cumulated_rates < theta && index < index_last
                index = next_index(event_handler, index)
                @inbounds cumulated_rates += event_handler.list_rate[index]
            end
        else
            index = index_last
            cumulated_rates_lower = sum(event_handler.list_rate) - event_handler.list_rate[index]
            # cumulated_rates in this case belong to id-1
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

function next_event(event_handler::ListEventRateSimple{T}, rng::AbstractRNG = Random.GLOBAL_RNG)::T where T
    if length(event_handler) > 0
        index = next_event(event_handler.list_rate, rng)
        return event_handler.list_event[index] 
    else
        return event_handler.noevent
    end
end

"""
    advance!(alg::KineticMonteCarlo, event_handler::AbstractEventHandler, update!::Function, total_time::T, [rng::AbstractRNG])::T where {T<:Real} 

Draw events from `event_handler` and update `event_handler` with `update!` until `total_time` has passed. Return time of last event.
"""
function advance!(alg::KineticMonteCarlo, event_handler::AbstractEventHandler, update!::Function, total_time::T, rng::AbstractRNG = Random.GLOBAL_RNG)::T where {T <: AbstractFloat}
    time::T = 0
    while time <= total_time
        dt, event = next(alg, rng, event_handler)
        time += dt
        (event == event_handler.noevent) || update!(event_handler, event)
    end
    return time 
end
