# ── Kinetic Monte Carlo ───────────────────────────────────────────────────────
#
# Continuous-time event-driven sampling: draw a Poissonian waiting time from the total rate,
# draw an event proportional to its rate, apply it. The event source is anything that answers
# `total_rate` and `next_event` — a plain rate vector, one of the event handlers
# (src/event_handler/), or a time-dependent function.

"""
    reset!(alg::AbstractKineticMonteCarlo)

Reset kinetic Monte Carlo statistics (`steps`, `time`) to zero.
"""
function reset!(alg::AbstractKineticMonteCarlo)
    alg.steps = 0
    alg.time = 0.0
end

# ── Waiting times ─────────────────────────────────────────────────────────────

"""
    next_time(rng, rate::Number) -> Float64

Waiting time of a homogeneous Poisson event stream with total `rate` (`Inf` for rate ≤ 0).
"""
function next_time(rng::AbstractRNG, rate::Number)::Float64
    rate > 0 || return Inf
    return randexp(rng) / rate
end

"""
    next_time(rng, rate::Function, rate_upper::Real) -> Float64

Waiting time of an inhomogeneous Poisson stream with instantaneous rate `rate(dt)`, sampled by
thinning against the dominating constant `rate_upper ≥ rate(dt)`.
"""
function next_time(rng::AbstractRNG, rate::Function, rate_upper::Real)::Float64
    rate_upper > 0 || return Inf
    dt = 0.0
    while true
        dt += next_time(rng, rate_upper)
        if rand(rng) < rate(dt) / rate_upper
            return dt
        end
    end
end

# ── Event selection ───────────────────────────────────────────────────────────

"""
    total_rate(source) -> Float64

Total event rate ΣR of an event source: a plain rate vector or an `AbstractEventHandlerRate`
(list handlers sum their rate list; `EventRateTree` reads its Fenwick root).
"""
total_rate(rates::AbstractVector) = sum(rates)
total_rate(event_handler::AbstractEventHandlerRate) = sum(event_handler.list_rate)

"""
    next_event(rng, rates::AbstractVector) -> Int

Draw one event index proportional to `rates`. One or two rates short-circuit (a single
`rand` decides two rates — the fast path for two-state problems); longer vectors use a
two-sided cumulative scan starting from the cheaper end.
"""
function next_event(rng::AbstractRNG, rates::AbstractVector)::Int
    n = length(rates)
    n == 1 && return 1
    if n == 2
        @inbounds return rand(rng) * (rates[1] + rates[2]) < rates[1] ? 1 : 2
    end
    sum_rates = sum(rates)
    theta = rand(rng) * sum_rates

    if theta < 0.5 * sum_rates
        index = 1
        cumulated_rates = rates[index]
        while cumulated_rates < theta && index < n
            index += 1
            @inbounds cumulated_rates += rates[index]
        end
        return index
    else
        index = n
        cumulated_rates_lower = sum_rates - rates[index]
        while cumulated_rates_lower > theta && index > 1
            index -= 1
            @inbounds cumulated_rates_lower -= rates[index]
        end
        return index
    end
end
@inline next_event(rng::AbstractRNG, rate::Number) = 1

# Same two-sided scan over a rate handler whose live events need index indirection
# (ListEventRateActiveMask); ListEventRateSimple below maps back to its event labels.
function next_event(rng::AbstractRNG, event_handler::AbstractEventHandlerRate{T})::T where T
    ne = length(event_handler)
    ne == 0 && return event_handler.noevent
    ne == 1 && return first_index(event_handler)
    sum_rates = total_rate(event_handler)
    theta::Float64 = rand(rng) * sum_rates
    index_last = last_index(event_handler)
    index_first = first_index(event_handler)

    if theta < 0.5 * sum_rates
        index = index_first
        cumulated_rates = event_handler.list_rate[index]
        while cumulated_rates < theta && index < index_last
            index = next_index(event_handler, index)
            @inbounds cumulated_rates += event_handler.list_rate[index]
        end
    else
        index = index_last
        cumulated_rates_lower = sum_rates - event_handler.list_rate[index]
        while cumulated_rates_lower > theta && index > index_first
            index = previous_index(event_handler, index)
            @inbounds cumulated_rates_lower -= event_handler.list_rate[index]
        end
    end
    return index
end

function next_event(rng::AbstractRNG, event_handler::ListEventRateSimple{T})::T where T
    length(event_handler) > 0 || return event_handler.noevent
    index = next_event(rng, event_handler.list_rate)
    return event_handler.list_event[index]
end

next_event(event_handler::ListEventRateSimple) = next_event(Random.GLOBAL_RNG, event_handler)

# ── One step: waiting time + event ────────────────────────────────────────────

"""
    next(alg::AbstractKineticMonteCarlo, source) -> (dt, event)

Draw the next waiting time and event from an event source: a scalar rate, a rate vector, a
rate handler, a time-ordered `AbstractEventHandlerTime`, or a function of time returning any
of these. Exhausted sources (total rate ≤ 0, empty queue) return `(Inf, nothing)`.
"""
function next(alg::AbstractKineticMonteCarlo, rate::Number)
    dt = next_time(alg.rng, rate)
    isfinite(dt) || return (Inf, nothing)
    return dt, 1
end

function next(alg::AbstractKineticMonteCarlo,
              source::Union{AbstractVector, AbstractEventHandlerRate})
    dt = next_time(alg.rng, total_rate(source))
    isfinite(dt) || return (Inf, nothing)
    return dt, next_event(alg.rng, source)
end

function next(alg::AbstractKineticMonteCarlo, event_handler::AbstractEventHandlerTime)
    length(event_handler) > 0 || return (Inf, nothing)
    time, event = popfirst!(event_handler)
    dt = time - get_time(event_handler)
    set_time!(event_handler, time)
    return dt, event
end

# Time-dependent rates: `rates_at_time(alg.time)` returns any of the sources above.
next(alg::AbstractKineticMonteCarlo, rates_at_time::Function) =
    next(alg, rates_at_time(alg.time))

"""
    step!(alg::AbstractKineticMonteCarlo, event_source) -> (t_new, event)

Perform one kinetic Monte Carlo step: advance `alg.time` by the drawn waiting time and count
the step. An exhausted source yields `(Inf, nothing)`.
"""
@inline function step!(alg::AbstractKineticMonteCarlo,
                       event_source::Union{AbstractVector, AbstractEventHandler, Function})
    dt, event = next(alg, event_source)
    t_new = alg.time + dt
    alg.steps += 1
    alg.time = t_new
    isfinite(t_new) || (event = nothing)
    return t_new, event
end

# ── System protocol ───────────────────────────────────────────────────────────

"""
    event_source(sys) -> Union{AbstractVector, AbstractEventHandler, Function}

Return the event source for system `sys`. Must be implemented by the user; raw sources
(vectors, handlers, functions) pass through unchanged.
"""
function event_source end

event_source(sys::Union{AbstractVector, AbstractEventHandler, Function}) = sys

"""
    modify!(sys, event, t)

Apply `event` to system `sys` at time `t`. Default is a no-op.
"""
modify!(sys, event, t) = nothing

"""
    observe!(sys, event, t)

Observe system `sys` at time `t`, BEFORE `modify!` applies the event — weight observables
with the elapsed time to form continuous-time averages. Default is a no-op. (Distinct from
`measure!`, the Measurements-container API.)
"""
observe!(sys, event, t) = nothing

# Design note: advance! runs the simulation loop for the user, which sits in tension with the
# package's template-for-specialization philosophy (compose your own loop from step!/observe!/
# modify!). It stays because the KMC loop order (step! → observe! → modify!) is easy to get
# wrong; whether MCMC deserves a symmetric convenience — or this one should slim down — is an
# open design question.

"""
    advance!(alg::AbstractKineticMonteCarlo, sys, total_time;
             t0=0, observe!=observe!, modify!=modify!, ckpt=nothing, checkpoint_interval=nothing)

Advance system `sys` using `alg` until `total_time`. Loop order per step:

1. `step!`      — draw the next event from `event_source(sys)`
2. `observe!`   — observe before modification: `observe!(sys, event, t)`
3. `modify!`    — apply the event:             `modify!(sys, event, t)`
4. optional checkpoint every `checkpoint_interval` steps (`ckpt::CheckpointSession`)

Stops early if the event source is exhausted; returns the final time.
"""
function advance!(
    alg::AbstractKineticMonteCarlo,
    sys,
    total_time;
    t0 = 0.0,
    observe! = observe!,
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
        observe!(sys, event, t_new)
        modify!(sys, event, t_new)
        if !isnothing(ckpt) && !isnothing(checkpoint_interval) && alg.steps % checkpoint_interval == 0
            checkpoint!(ckpt; step=alg.steps)
        end
    end
    return alg.time
end

# ── Gillespie: the named continuous-time sampler ──────────────────────────────

"""
    Gillespie(rng=Random.GLOBAL_RNG) <: AbstractKineticMonteCarlo

Gillespie algorithm (stochastic simulation algorithm): the direct sampler of the KMC
protocol above — exponential waiting times from the total rate, events proportional to their
rates. Carries the `rng` and the `steps`/`time` counters.
"""
mutable struct Gillespie{RNG<:AbstractRNG} <: AbstractKineticMonteCarlo
    rng::RNG
    steps::Int
    time::Float64
end
Gillespie(rng::AbstractRNG) = Gillespie(rng, 0, 0.0)
Gillespie() = Gillespie(Random.GLOBAL_RNG)
