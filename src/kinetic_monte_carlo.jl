# KineticMonteCarlo
struct KineticMonteCarlo end

#TODO:
# * find better name for event_handler that captures both event_handler and
#   vector of rates

"""
    SimulationKineticMonteCarlo

object that handles kinetic Monte Carlo simulation. Includes `rng` for
persistent and reproducible simulation and `event_handler` that determines
transition events and times.
Most simple terms, event_handler can just be an AbstractVector or
ProbabilityWeight (latter is more performant)
"""
struct SimulationKineticMonteCarlo{T}
   rng::AbstractRNG
   event_handler::T
end
function init(rng::AbstractRNG, alg::KineticMonteCarlo, event_handler)
    return SimulationKineticMonteCarlo(rng, event_handler)
end
init(alg::KineticMonteCarlo, event_handler) = init(Random.GLOBAL_RNG, alg, event_handler)


@doc raw"""
    next(sim::SimulationKineticMonteCarlo)

Next stochastic event (``\Delta t``, index) drawn according to event_handler in
simulation protocol. Simplest case is that event_handler is a list of rates.
"""
function next(sim::SimulationKineticMonteCarlo{T}) where T <: Union{AbstractVector, AbstractWeights}
    sum_rates = sum(sim.event_handler)
    if !(sum_rates > 0)
        return Inf, 0
    end
    dtime = next_time(sim.rng, sum_rates)
    event = next_event(sim.rng, sim.event_handler)
    return dtime, event
end
function next(sim::SimulationKineticMonteCarlo{T}) where T <: AbstractEventHandlerRate
    sum_rates = sum(sim.event_handler.list_rate)
    if !(sum_rates > 0)
        return Inf, event_handler.noevent
    end
    dtime = next_time(sim.rng, sum_rates)
    event = next_event(sim.rng, sim.event_handler)
    return dtime, event
end

#@doc raw"""
#    next([rng::AbstractRNG,] alg::KineticMonteCarlo, rates::AbstractWeights)::Tuple{Float64,Int}
#
#Next stochastic event (``\Delta t``, index) drawn proportional to probability given in `rates`
#"""
#function next(rng::AbstractRNG, alg::KineticMonteCarlo, rates::Union{AbstractWeights, AbstractVector})::Tuple{Float64,Int}
#    sum_rates = sum(rates)
#    if !(sum_rates > 0)
#        return Inf, 0
#    end
#    dtime = next_time(rng, sum_rates)
#    event = next_event(rng, rates)
#    return dtime, event
#end
#next(alg::KineticMonteCarlo, rates::AbstractWeights) = next(Random.GLOBAL_RNG, alg, rates)

#@doc raw"""
#    next([rng::AbstractRNG,] alg::KineticMonteCarlo, event_handler::AbstractEventHandlerRate)
#
#Next stochastic event (``\Delta t``, event type) organized by `event_handler`
#fast(to be tested, depends on overhead of EventList) implementation of
#next_event_rate if defined by EventList object
#"""
#function next(rng::AbstractRNG, alg::KineticMonteCarlo, event_handler::AbstractEventHandlerRate)::Tuple{Float64,Int}
#    sum_rates = sum(event_handler.list_rate)
#    if !(sum_rates > 0)
#        return Inf, event_handler.noevent
#    end
#    dtime = next_time(rng, sum_rates)
#    event = next_event(rng, event_handler)
#    return dtime, event
#end
#next(alg::KineticMonteCarlo, event_handler::AbstractEventHandlerRate) = next(Random.GLOBAL_RNG, alg, event_handler)

@doc raw"""
    next_time(rng::AbstractRNG, rate::Float64)::Float64

Next stochastic ``\Delta t`` for Poisson process constant with `rate`
"""
function next_time(rng::AbstractRNG, rate_generation::Number)::Float64
    dt = randexp(rng) / rate_generation
    return dt
end

@doc raw"""
    next_time(rng::AbstractRNG, rate::Function, rate_generation::Number)::Float64

Next stochastic ``\Delta t`` for inhomogeneous Poisson process constant with `rate` function. 
Needs to be supplemented by a generation rate for proposals that are selected by the rate(t), which needs to be always larger or equal to rate(t) for all t. 
Alternatively, one can implement maximum(rate) as rate_generation.
Uses Ogata's thinning algorithm.
"""
function next_time(rng::AbstractRNG, rate::Function, rate_generation::Float64)::Float64
    dt = 0.0
    while true
        dt += next_time(rng, rate_generation)
        if rand(rng) < rate(dt) / rate_generation
            return dt
        end
    end
end
# TODO: test if this could be useful
# function next_time(rng::AbstractRNG, rate::Function)::Float64
#     next_time(rng, rate, maximum(rate))
# end


"""
    next_event(rng::AbstractRNG, event_handler::AbstractWeights)::Int

random selection of next event from `event_handler`.

`event_handler` can have multiple levels of complexity.
* of type AbstractWeight or AbstractVector, the function selects a single
random index in `1:length(rates)` with probability proportional to the entry in
`rates`.
* of type AbstractEventHandlerRate, the function selects single random event of
predefined type with a given probability managed by `event_handler`. The
`event_handler` also manages the case that no valid events are left (e.g.  when
all rates are equal to zero). This becomes relevant when using
[`advance!`](@ref) to advance for some time.


# Remarks
For simple case of two events there is a fast specialization and when using
ProbabilityWeights, this is still on average twice as fast as
StatsBase.sampling because it can iterate from either beginning or end of rates

See also: [`advance!`](@ref)
"""
function next_event(rng::AbstractRNG, rates::Union{AbstractWeights, AbstractVector})::Int
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

# specialization for two events only
function next_event(rng::AbstractRNG, rates::MVector{2,T})::Int where {T <: AbstractFloat}
    if rand(rng) * (rates[1] + rates[2]) < rates[1]
        return 1
    else
        return 2
    end
end

# specialization for AbstractEventHandlerRate
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

# specialization for ListEventRateSimple
function next_event(rng::AbstractRNG, event_handler::ListEventRateSimple{T})::T where T
    if length(event_handler) > 0
        index = next_event(rng, event_handler.list_rate)
        return event_handler.list_event[index]
    else
        return event_handler.noevent
    end
end
next_event(event_handler::ListEventRateSimple) = next_event(Random.GLOBAL_RNG, event_handler)



"""
    advance!(sim, update!::Function, total_time::T)::T where {T<:Real}

advances the `sim` that internally draw events, by updatint `sim` with
`update!(sim, event)` until `total_time` has passed. Return time of last event.
"""
function advance!(
        sim::SimulationKineticMonteCarlo,
        update!::Function,
        total_time::T
    )::T where {T <: AbstractFloat}
    time::T = 0
    while time <= total_time
        dt, event = next(sim)
        time += dt
        update!(sim, event)
    end
    return time
end


#TODO: maybe move this as specialization when constructing event_handler with
#known fixed number of rates
#
#
#"""
#    next_event(rng::AbstractRNG, cumulated_rates::Vector{T})::Int where {T<:AbstractFloat}
#
#Select a single random index in `1:length(cumulated_rates)` with cumulated probability given in `cumulated_rates`.
#
## Remarks
#Deprecated unless we find a good data structure for (dynamic) cumulated rates
#"""
#function next_event_cumulative(rng::AbstractRNG, cumulated_rates::Vector{T})::Int where {T <: AbstractFloat}
#    theta = rand(rng) * cumulated_rates[end]
#    index = MonteCarloX.binary_search(cumulated_rates, theta)
#    return index
#end
