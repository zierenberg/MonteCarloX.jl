# Kinetic Monte Carlo - core functionality
# Shared by all continuous-time event-driven algorithms.

"""
    reset!(alg::AbstractKineticMonteCarlo)

Reset kinetic Monte Carlo statistics (`steps`, `time`) to zero.
"""
function reset!(alg::AbstractKineticMonteCarlo)
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
@inline next_event(rng::AbstractRNG, rate::Number) = 1

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
    event = 1
    if !isfinite(dt)
        return Inf, nothing
    end
    return dt, event
end

"""
    next(alg::AbstractKineticMonteCarlo, rates::AbstractVector)

Draw next event waiting time and event id from raw rates (StatsBase.AbstractWeights is also an AbstractVector).

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
    step!(alg::AbstractKineticMonteCarlo, event_source)

Perform one kinetic Monte Carlo step from an event_source (rates or an event handler).

Updates `alg.time` and `alg.steps` internally and returns `(t_new, event)`.

If dt=Inf then this is reflected in the algorithm.time and returned correspondingly
If event is nothing, then this is returned as well.
"""
@inline function step!(alg::AbstractKineticMonteCarlo, event_source::Union{AbstractVector, AbstractEventHandler, Function}) 
    dt, event = next(alg, event_source)
    t_new = alg.time + dt
    alg.steps += 1
    alg.time = t_new
    if !isfinite(t_new) 
        event = nothing
    end
    return t_new, event
end

"""
    event_source(sys) -> Union{AbstractVector, AbstractEventHandler, Function}

Return the event source for system `sys`. Must be implemented by the user.
"""
function event_source end

event_source(sys::Union{AbstractVector, AbstractEventHandler, Function}) = sys

"""
    modify!(sys, event, t)

Apply `event` to system `sys` at time `t`. Default is a no-op.
"""
modify!(sys, event, t) = nothing

"""
    measure!(sys, event, t)

Observe system `sys` at time `t` before `modify!`. Default is a no-op.
"""
measure!(sys, event, t) = nothing

"""
    advance!(alg::AbstractKineticMonteCarlo, sys, total_time; t0=0, measure!=measure!, modify!=modify!, ckpt=nothing, checkpoint_interval=nothing)

Advance system `sys` using `alg` until `total_time`.

Loop order per step:
1. `step!`         — draw next event from `event_source(sys)`
2. `measure!`      — observe before modification: `measure!(sys, event, t)`
3. `modify!`       — apply event to state:        `modify!(sys, event, t)`
4. `checkpoint!`   — optional: if `ckpt::CheckpointSession` and
                     `checkpoint_interval` are provided, write a checkpoint
                     every `checkpoint_interval` steps.

Stops early if the event source is exhausted or total time is reached.

# Checkpoint usage
```julia
ckpt = init_checkpoint("sim/ckpt.mcx", (alg=alg, sys=sys); step=0)
advance!(alg, sys, 1e6; ckpt=ckpt, checkpoint_interval=1000)
```
"""
function advance!(
    alg::AbstractKineticMonteCarlo,
    sys,
    total_time;
    t0 = 0.0,
    measure! = measure!,
    modify! = modify!,
    ckpt = nothing,
    checkpoint_interval = nothing,
)
    alg.time = float(t0)
    src = event_source(sys)
    src isa AbstractEventHandlerTime && set_time!(src, alg.time)

    while alg.time < total_time
        t_new, event = step!(alg, src)
        isnothing(event) && return alg.time
        measure!(sys, event, t_new)
        modify!(sys, event, t_new)
        if !isnothing(ckpt) && !isnothing(checkpoint_interval) && alg.steps % checkpoint_interval == 0
            checkpoint!(ckpt; step=alg.steps)
        end
    end
    return alg.time
end
