# Kinetic Monte Carlo - core functionality
# Shared by all continuous-time event-driven algorithms.

"""
    AbstractKineticMonteCarlo <: AbstractAlgorithm

Base type for continuous-time kinetic Monte Carlo algorithms.

Kinetic Monte Carlo algorithms:
- Draw waiting times from event rates
- Draw events proportional to rates
- Track simulation time and number of accepted events
"""
abstract type AbstractKineticMonteCarlo <: AbstractAlgorithm end

"""
    reset_statistics!(alg::AbstractKineticMonteCarlo)

Reset kinetic Monte Carlo statistics (`steps`, `time`) to zero.
"""
function reset_statistics!(alg::AbstractKineticMonteCarlo)
    alg.steps = 0
    alg.time = 0.0
end

"""
    next_time(rng::AbstractRNG, rate_generation::Number)::Float64

Draw next waiting time for a homogeneous Poisson event stream.
"""
function next_time(rng::AbstractRNG, rate_generation::Number)::Float64
    if !(rate_generation > 0)
        return Inf
    end
    return randexp(rng) / rate_generation
end

"""
    next_time(rng::AbstractRNG, rate::Function, rate_generation::Real)::Real

Draw next waiting time for an inhomogeneous Poisson stream using thinning.
"""
function next_time(rng::AbstractRNG, rate::Function, rate_generation::Real)::Real
    if !(rate_generation > 0)
        return Inf
    end
    dt = 0.0
    while true
        dt += next_time(rng, rate_generation)
        if rand(rng) < rate(dt) / rate_generation
            return dt
        end
    end
end

"""
    next_event(rng::AbstractRNG, rates::Union{AbstractWeights, AbstractVector})::Int

Draw one event index proportional to rates.
"""
function next_event(rng::AbstractRNG, rates::Union{AbstractWeights, AbstractVector})::Int
    sum_rates = sum(rates)
    theta = rand(rng) * sum_rates

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
        while cumulated_rates_lower > theta && index > 1
            index -= 1
            @inbounds cumulated_rates_lower -= rates[index]
        end
        return index
    end
end

next_event(::AbstractRNG, ::Number) = 1

function next_event(rng::AbstractRNG, rates::MVector{2,T})::Int where {T <: AbstractFloat}
    if rand(rng) * (rates[1] + rates[2]) < rates[1]
        return 1
    else
        return 2
    end
end

function next_event(rng::AbstractRNG, event_handler::AbstractEventHandlerRate{T})::T where T
    ne = length(event_handler)
    if ne > 1
        total_rate = sum(event_handler.list_rate)
        theta::Float64 = rand(rng) * total_rate
        index_last = last_index(event_handler)
        index_first = first_index(event_handler)

        if theta < 0.5 * total_rate
            index = index_first
            cumulated_rates = event_handler.list_rate[index]
            while cumulated_rates < theta && index < index_last
                index = next_index(event_handler, index)
                @inbounds cumulated_rates += event_handler.list_rate[index]
            end
        else
            index = index_last
            cumulated_rates_lower = total_rate - event_handler.list_rate[index]
            while cumulated_rates_lower > theta && index > index_first
                index = previous_index(event_handler, index)
                @inbounds cumulated_rates_lower -= event_handler.list_rate[index]
            end
        end
        return index
    elseif ne == 1
        return first_index(event_handler)
    else
        return event_handler.noevent
    end
end

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
    next(alg::AbstractKineticMonteCarlo, rates::AbstractVector)

Draw next event waiting time and event id from raw rates.

For zero total rate, returns `(Inf, 0)`.
"""
function next(alg::AbstractKineticMonteCarlo, rate::Number)
    dt = next_time(alg.rng, rate)
    if !isfinite(dt)
        return Inf, nothing
    end
    return dt, 1
end

"""
    next(alg::AbstractKineticMonteCarlo, rates::AbstractVector)

Draw next event waiting time and event id from raw rates.

For zero total rate, returns `(Inf, 0)`.
"""
function next(alg::AbstractKineticMonteCarlo, rates::AbstractVector)
    if length(rates) == 1
        return next(alg, rates[1])
    end
    sum_rates = sum(rates)
    dt = next_time(alg.rng, sum_rates)
    if !isfinite(dt)
        return Inf, nothing
    end
    event = next_event(alg.rng, rates)
    return dt, event
end

"""
    next(alg::AbstractKineticMonteCarlo, rates::AbstractWeights)

Draw next event waiting time and event id from weighted rates.

For zero total rate, returns `(Inf, 0)`.
"""
function next(alg::AbstractKineticMonteCarlo, rates::AbstractWeights)
    if length(rates) == 1
        return next(alg, rates[1])
    end
    sum_rates = sum(rates)
    dt = next_time(alg.rng, sum_rates)
    if !isfinite(dt)
        return Inf, nothing
    end
    event = next_event(alg.rng, rates)
    return dt, event
end

"""
    next(alg::AbstractKineticMonteCarlo, event_handler::AbstractEventHandlerRate)

Draw next event waiting time and event from an event handler.
"""
function next(alg::AbstractKineticMonteCarlo, event_handler::AbstractEventHandlerRate)
    sum_rates = sum(event_handler.list_rate)
    dt = next_time(alg.rng, sum_rates)
    if !isfinite(dt)
        return Inf, nothing
    end
    event = next_event(alg.rng, event_handler)
    return dt, event
end

"""
    next(alg::AbstractKineticMonteCarlo, event_handler::AbstractEventHandlerTime)

Draw next event from a time-ordered event queue.
"""
function next(alg::AbstractKineticMonteCarlo, event_handler::AbstractEventHandlerTime)
    if length(event_handler) > 0
        time, event = popfirst!(event_handler)
        dt = time - get_time(event_handler)
        set_time!(event_handler, time)
        return dt, event
    else
        return Inf, nothing
    end
end

"""
    next(alg::AbstractKineticMonteCarlo, rates_at_time::Function)

Draw next event waiting time and event from a time-dependent rates function.

`rates_at_time` is called as `rates_at_time(alg.time)` and should return either
an `AbstractVector`, `AbstractWeights`, or `AbstractEventHandlerRate`.
"""
function next(alg::AbstractKineticMonteCarlo, rates_at_time::Function)
    return next(alg, rates_at_time(alg.time))
end

"""
    step!(alg::AbstractKineticMonteCarlo, rates::AbstractVector)

Perform one kinetic Monte Carlo event from raw rates.

Updates `alg.time` and `alg.steps` internally and returns `(t_new, event)`.
"""
function step!(alg::AbstractKineticMonteCarlo, rates::AbstractVector)
    dt, event = next(alg, rates)
    t_new = alg.time + dt
    alg.steps += 1
    alg.time = t_new
    return t_new, event
end

"""
    step!(alg::AbstractKineticMonteCarlo, rates::AbstractWeights)

Perform one kinetic Monte Carlo event from weighted rates.

Updates `alg.time` and `alg.steps` internally and returns `(t_new, event)`.
"""
function step!(alg::AbstractKineticMonteCarlo, rates::AbstractWeights)
    dt, event = next(alg, rates)
    t_new = alg.time + dt
    alg.steps += 1
    alg.time = t_new
    return t_new, event
end

"""
    step!(alg::AbstractKineticMonteCarlo, event_handler::AbstractEventHandlerRate)

Perform one kinetic Monte Carlo event from an event-handler backend.

Updates `alg.time` and `alg.steps` internally and returns `(t_new, event)`.
"""
function step!(alg::AbstractKineticMonteCarlo, event_handler::AbstractEventHandlerRate)
    dt, event = next(alg, event_handler)
    t_new = alg.time + dt
    alg.steps += 1
    alg.time = t_new
    return t_new, event
end

"""
    step!(alg::AbstractKineticMonteCarlo, event_handler::AbstractEventHandlerTime)

Perform one kinetic Monte Carlo event from a time-event handler backend.

Updates `alg.time` and `alg.steps` internally and returns `(t_new, event)`.
"""
function step!(alg::AbstractKineticMonteCarlo, event_handler::AbstractEventHandlerTime)
    dt, event = next(alg, event_handler)
    if event === nothing || !isfinite(dt)
        return Inf, event
    end
    t_new = alg.time + dt
    alg.steps += 1
    alg.time = t_new
    return t_new, event
end

"""
    step!(alg::AbstractKineticMonteCarlo, rates_at_time::Function)

Perform one kinetic Monte Carlo event from a time-dependent rates callback.

Updates `alg.time` and `alg.steps` internally and returns `(t_new, event)`.
"""
function step!(alg::AbstractKineticMonteCarlo, rates_at_time::Function)
    dt, event = next(alg, rates_at_time)
    t_new = alg.time + dt
    alg.steps += 1
    alg.time = t_new
    return t_new, event
end

"""
    advance!(alg::AbstractKineticMonteCarlo, rates::AbstractVector, total_time; t0=0, update!=nothing)

Advance a continuous-time process with raw rates until `total_time`.

If `update!` is provided, it is called as `update!(rates, event, t_new)`
after each sampled event.
"""
function _advance_loop!(;
    alg::AbstractKineticMonteCarlo,
    total_time,
    t0::Real,
    setup!::Union{Nothing,Function}=nothing,
    step_fn::Function,
    on_event!::Union{Nothing,Function}=nothing,
)
    alg.time = float(t0)
    setup! !== nothing && setup!()

    while alg.time < total_time
        t_new, event = step_fn()
        if event === nothing || !isfinite(t_new)
            return t_new
        end
        on_event! !== nothing && on_event!(event, t_new)
    end

    return alg.time
end

function advance!(
    alg::AbstractKineticMonteCarlo,
    rates::AbstractVector,
    total_time;
    t0 = 0.0,
    update! = nothing,
)
    return _advance_loop!(;
        alg=alg,
        total_time=total_time,
        t0=t0,
        step_fn=() -> step!(alg, rates),
        on_event! = update! === nothing ? nothing : (event, t_new) -> update!(rates, event, t_new),
    )
end

"""
    advance!(alg::AbstractKineticMonteCarlo, rates::AbstractWeights, total_time; t0=0, update!=nothing)

Advance a continuous-time process with weighted rates until `total_time`.

If `update!` is provided, it is called as `update!(rates, event, t_new)`
after each sampled event.
"""
function advance!(
    alg::AbstractKineticMonteCarlo,
    rates::AbstractWeights,
    total_time;
    t0 = 0.0,
    update! = nothing,
)
    return _advance_loop!(;
        alg=alg,
        total_time=total_time,
        t0=t0,
        step_fn=() -> step!(alg, rates),
        on_event! = update! === nothing ? nothing : (event, t_new) -> update!(rates, event, t_new),
    )
end

"""
    advance!(alg::AbstractKineticMonteCarlo, event_handler::AbstractEventHandlerRate, total_time; t0=0, update!=nothing)

Advance a continuous-time process with an event-handler backend until `total_time`.

If `update!` is provided, it is called as
`update!(event_handler, event, t_new)` after each sampled event.
"""
function advance!(
    alg::AbstractKineticMonteCarlo,
    event_handler::AbstractEventHandlerRate,
    total_time;
    t0 = 0.0,
    update! = nothing,
)
    return _advance_loop!(;
        alg=alg,
        total_time=total_time,
        t0=t0,
        step_fn=() -> step!(alg, event_handler),
        on_event! = update! === nothing ? nothing : (event, t_new) -> update!(event_handler, event, t_new),
    )
end

"""
    advance!(alg::AbstractKineticMonteCarlo, event_handler::AbstractEventHandlerTime, total_time; t0=0, update!=nothing)

Advance a continuous-time process with a time-event handler until `total_time`.

If `update!` is provided, it is called as
`update!(event_handler, event, t_new)` after each sampled event.
"""
function advance!(
    alg::AbstractKineticMonteCarlo,
    event_handler::AbstractEventHandlerTime,
    total_time;
    t0 = 0.0,
    update! = nothing,
)
    setup! = () -> set_time!(event_handler, alg.time)
    return _advance_loop!(;
        alg=alg,
        total_time=total_time,
        t0=t0,
        setup! = setup!,
        step_fn=() -> step!(alg, event_handler),
        on_event! = update! === nothing ? nothing : (event, t_new) -> update!(event_handler, event, t_new),
    )
end

"""
    advance!(alg::AbstractKineticMonteCarlo, sys, total_time; rates, t0=0, measure!=nothing, update!=nothing)

Advance a model system where rates are provided explicitly via
`rates(sys, t)` callback.

If `measure!` is provided, it is called before update as `measure!(sys, t_new, event)`.
If `update!` is provided, it is called as `update!(sys, event, t_new)`.

This two-stage interface is useful for kinetic simulations where measurements are
defined at event times before the event modifies the state.
"""
function advance!(
    alg::AbstractKineticMonteCarlo,
    sys,
    total_time;
    rates::Function,
    t0 = 0.0,
    measure! = nothing,
    update! = nothing,
)
    on_event! = (event, t_new) -> begin
        measure! !== nothing && measure!(sys, t_new, event)
        update! !== nothing && update!(sys, event, t_new)
    end

    return _advance_loop!(;
        alg=alg,
        total_time=total_time,
        t0=t0,
        step_fn=() -> step!(alg, t -> rates(sys, t)),
        on_event! = on_event!,
    )
end
