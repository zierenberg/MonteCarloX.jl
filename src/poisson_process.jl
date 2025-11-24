struct PoissonProcess end
"""
    SimulationPoissonProcess

object that handles Poisson process simulations. Includes `rng` for
persistent and reproducible simulation, the option for time-dependent rates, and the option for optimization schemes (e.g. when the rate function is known to monotonically decrease between events).
"""
struct SimulationPoissonProcess{T1,T2}
   rng::AbstractRNG
   rate::T1
   generation_rate::T2
   piecewise_rate_decrease::Bool
end
function init(rng::AbstractRNG, alg::PoissonProcess, rate; generation_rate=nothing, piecewise_rate_decrease::Bool=false)
    if isa(rate, Number)
        rate = float(rate)
        @info "Creating Poisson process with constant rate"
    elseif isa(rate, AbstractVector)
        rate = float.(rate)
        @info "Creating Poisson process with constant rates for multiple channels"
    elseif isa(rate, Function)
        @info "Creating Poisson process with time-dependent rate(s)"
        if generation_rate === nothing && piecewise_rate_decrease == false
            error("Either generation_rate has to be provided or `piecewise_rate_decrease` has to be true.")
        end
        if isa(generation_rate, Number)
            generation_rate = float(generation_rate)
        end
    else
        error("Rate has to be either Number, Vector{Number}, or Function.")
    end
    return SimulationPoissonProcess(rng, rate, generation_rate, piecewise_rate_decrease)
end

init(alg::PoissonProcess, rate; generation_rate=nothing, piecewise_rate_decrease::Bool=false) = init(Random.GLOBAL_RNG, alg, rate; generation_rate=generation_rate, piecewise_rate_decrease=piecewise_rate_decrease)

@doc raw"""
    next(sim::PoissonProcess; t=nothing)

Next stochastic event (``\Delta t``, index) for a collection of Poisson processes. 
If they are inhomogeneous then the current time needs to be provided. 
For most custom solutions, use `next_time` and `next_event` directly.
"""
# for a single Poisson process
function next(sim::SimulationPoissonProcess{T1,T2})::Tuple{Float64,Int} where T1 <: Number where T2 <: Nothing
    rate = sim.rate
    dt = next_time(sim.rng, rate)
    return dt, 0
end
# for multiple Poisson processes (should be identical to the kinetic Monte Carlo case for constant rates)
function next(sim::SimulationPoissonProcess{T1,T2})::Tuple{Float64,Int} where T1 <: AbstractVector where T2 <: Nothing
    sum_rates = sum(sim.rate)
    if !(sum_rates > 0)
        return Inf, 0
    end
    dt = next_time(sim.rng, sum_rates)
    id = next_event(sim.rng, sim.rate)
    return dt, id
end
# for inhomogeneous Poisson process with time-dependent rate function and fixed generation rate
function next(sim::SimulationPoissonProcess{T1,T2}, t::Number)::Tuple{Float64,Int} where T1 <: Function where T2 <: Union{Number, AbstractArray}
    global_rate(time) = sum(sim.rate(time))
    generation_rate = sum(sim.generation_rate)
    if !(generation_rate > 0)
        return Inf, 0
    end
    dt = next_time(sim.rng, global_rate, generation_rate)
    id = next_event(sim.rng, sim.rate(t+dt))
    return t+dt, id
end
# makes no sense without feedback that intermediately increases the rate
# # for inhomogeneous Poisson process with time-dependent rate function and local piecewise generation rate
# function next(sim::SimulationPoissonProcess{T1,T2}, t::Number)::Tuple{Float64,Int} where T1 <: Function where T2 <: Nothing
#     global_rate(time) = sum(sim.rate(time))
#     generation_rate = global_rate(t)
#     if !(generation_rate > 0)
#         return Inf, 0
#     end
#     dt = next_time(sim.rng, global_rate(t), generation_rate)
#     id = next_event(sim.rng, sim.rate(t+dt))
#     return t+dt, id
# end