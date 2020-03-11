# KineticMonteCarlo
struct KineticMonteCarlo end

"""
    next(alg::KineticMonteCarlo, [rng::AbstractRNG,] rates::AbstractWeights)::Tuple{Float64,Int}

Next stochastic event (`\\Delta t`, index) drawn proportional to probability given in `rates`
"""
function next(alg::KineticMonteCarlo, rng::AbstractRNG, rates::AbstractWeights)::Tuple{Float64,Int}
    dtime = next_time(rng, sum(rates))
    index = next_event(rng, rates)
    return dtime, index
end

next(alg::KineticMonteCarlo, rates::AbstractWeights) = next(alg, Random.GLOBAL_RNG, rates)

"""
    next(alg::KineticMonteCarlo, [rng::AbstractRNG,] event_handler::AbstractEventHandlerRate)

Next stochastic event (`\\Delta t`, event type) organized by `event_handler`
fast(to be tested, depends on overhead of EventList) implementation of next_event_rate if defined by EventList object
"""
function next(alg::KineticMonteCarlo, rng::AbstractRNG, event_handler::AbstractEventHandlerRate)::Tuple{Float64,Int}
    dt = next_time(rng, sum(event_handler.list_rate))
    id = next_event(rng, event_handler)
    return dt, id
end

next(alg::KineticMonteCarlo, event_handler::AbstractEventHandlerRate) = next(alg, Random.GLOBAL_RNG, event_handler)

"""
    next_time([rng::AbstractRNG,] rate::Float64)::Float64

Next stochastic `\\Delta t` for Poisson process with `rate`
"""
function next_time(rng::AbstractRNG, rate::Float64)::Float64
    return randexp(rng) / rate
end

next_time(rate::Float64) = next_time(Random.GLOBAL_RNG, rate)

"""
    next_event(rng::AbstractRNG, cumulated_rates::Vector{T})::Int where {T<:AbstractFloat}

Select a single random index in `1:length(cumulated_rates)` with cumulated probability given in `cumulated_rates`.

# Remarks
Deprecated unless we find a good data structure for (dynamic) cumulated rates
"""
function next_event(rng::AbstractRNG, cumulated_rates::Vector{T})::Int where {T <: AbstractFloat}
    theta = rand(rng) * cumulated_rates[end]
    index = MonteCarloX.binary_search(cumulated_rates, theta)
    return index
end

"""
    next_event([rng::AbstractRNG,] rates::AbstractWeights)::Int 

Select a single random index in `1:length(rates)` with probability proportional to the entry in `rates`.

# Remarks
This is on average twice as fast as StatsBase.sampling because it can iterate from either beginning or end of rates
"""
function next_event(rng::AbstractRNG, rates::AbstractWeights)::Int 
    theta = rand(rng) * rates.sum
    N = length(rates)

    # this is for extra performance in case of large lists (factor of 2)
    if theta < 0.5 * rates.sum
        index = 1
        cumulated_rates = rates[index]
        while cumulated_rates < theta && index < N
            index += 1
            @inbounds cumulated_rates += rates[index]
        end
        return index

    else
        index = N
        cumulated_rates_lower = rates.sum - rates[index]
        # cumulated_rates in this case belong to index-1
        while cumulated_rates_lower > theta && index > 1
            index -= 1
            @inbounds cumulated_rates_lower -= rates[index]
        end
        return index

    end
end

# relevent to write this as extra function this for performance issues (compared to keyword argument with preset value on GLOBAL_RNG)
next_event(list_rates::AbstractWeights) = next_event(Random.GLOBAL_RNG, list_rates) 


"""
    next_event([rng::AbstractRNG,] event_handler::AbstractEventHandlerRate{T})::T where T

Select a single random event with a given probability managed by `event_handler`.

The `event_handler` also manages the case that no valid events are left (e.g.
when all rates are equal to zero). This becomes relevant when using [`advance!`](@ref)
to advance for some time.

See also: [`advance!`](@ref)
"""
function next_event(rng::AbstractRNG, event_handler::AbstractEventHandlerRate{T})::T where T
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

next_event(event_handler::AbstractEventHandlerRate) = next_event(Random.GLOBAL_RNG, event_handler) 

function next_event(rng::AbstractRNG, event_handler::ListEventRateSimple{T})::T where T
    if length(event_handler) > 0
        index = next_event(rng, event_handler.list_rate)
        return event_handler.list_event[index] 
    else
        return event_handler.noevent
    end
end

#Q: need to keep the type stability output?
next_event(event_handler::ListEventRateSimple) = next_event(Random.GLOBAL_RNG, event_handler) 

"""
    advance!(alg::KineticMonteCarlo, [rng::AbstractRNG], event_handler::AbstractEventHandlerRate, update!::Function, total_time::T)::T where {T<:Real} 

Draw events from `event_handler` and update `event_handler` with `update!` until `total_time` has passed. Return time of last event.
"""
function advance!(alg::KineticMonteCarlo, rng::AbstractRNG, event_handler::AbstractEventHandlerRate, update!::Function, total_time::T)::T where {T <: AbstractFloat}
    time::T = 0
    while time <= total_time
        if length(event_handler) == 0
            println("WARNING: no events left before total_time reached")
            return time
        end
        dt, event = next(alg, rng, event_handler)
        time += dt
        update!(event_handler, event)
    end
    return time 
end

function advance!(alg::KineticMonteCarlo, event_handler::AbstractEventHandlerRate, update!::Function, total_time::T)::T where {T <: AbstractFloat} 
  advance!(alg, Random.GLOBAL_RNG, event_handler, update!, total_time)
end
