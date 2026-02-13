abstract type AbstractPoissonProcess end
"""
    PoissonProcess(rate; rng=Random.GLOBAL_RNG)

Homogeneous Poisson process with constant rate.

# Arguments
- `rate::Real` — single process
- `rate::AbstractVector{<:Real}` — multiple processes
"""
struct PoissonProcess{R<:AbstractRNG,T} <: AbstractPoissonProcess
    rng::R
    rate::T
end
PoissonProcess(rate; rng=Random.GLOBAL_RNG) = PoissonProcess(rng, float(rate))

"""
    InhomogeneousPoissonProcess(rate, generation_rate; rng=Random.GLOBAL_RNG)

Inhomogeneous Poisson process with time-dependent rate.

# Arguments
- `rate`: function that returns either a scalar or vector of instantaneous rates at time `t`
- `generation_rate`: upper bound on the instantaneous rate (for thinning algorithm)
"""
struct InhomogeneousPoissonProcess{R<:AbstractRNG,F,G} <: AbstractPoissonProcess
    rng::R
    rate::F
    generation_rate::G
end
InhomogeneousPoissonProcess(rate, generation_rate; rng=Random.GLOBAL_RNG) =
    InhomogeneousPoissonProcess(rng, rate, generation_rate)

"""
    advance!(pp::AbstractPoissonProcess, t_final; t0=0, update!=nothing)

Advance a Poisson process until time `t_final`.

# Arguments
- `pp`: the Poisson process (homogeneous or inhomogeneous)
- `t_final`: final time
- `t0`: initial time (default 0)
- `update!`: optional callback `update!(event, t)` for storing events or manipulating state

# Returns
Final time `t` (may exceed `t_final` for the last event).

# Implementation notes
For homogeneous processes, events are generated directly from the constant rate.
For inhomogeneous processes, the thinning algorithm is used with `generation_rate` as an upper bound.
"""
advance!(pp::AbstractPoissonProcess, t_final; t0=0, update! = nothing)

# Homogeneous Poisson process
function advance!(
    pp::PoissonProcess, 
    t_final; 
    t0=zero(typeof(t_final)), 
    update! = nothing
)   
    t = t0
    while t < t_final
        sum_rates = sum(pp.rate) # this important in case update! changes rates
        sum_rates <= 0 && return oftype(t, Inf)     
        dt = next_time(pp.rng, sum_rates)
        t += dt
        event = next_event(pp.rng, pp.rate)
        update! !== nothing && update!(pp, t, event)
    end

    return t
end

# Inhomogeneous Poisson process
function advance!(
    pp::InhomogeneousPoissonProcess,
    t_final;
    t0=zero(typeof(t_final)),
    update! = nothing
)
    t = t0
    while t < t_final
        # this is important in case update! changes rates
        gen_rate = sum(pp.generation_rate)
        gen_rate <= 0 && return oftype(t, Inf)
        dt = next_time(pp.rng, t->sum(pp.rate(t)), gen_rate)
        t += dt
        # get event based on the relative rates at time of event
        event = next_event(pp.rng, pp.rate(t))
        update! !== nothing && update!(pp, t, event)
    end

    return t
end