# test for waiting time method to simulate stochastic SIR process
using MonteCarloX
using Random
using Distributions
using StatsBase
using LinearAlgebra
using StaticArrays

using Printf
using DelimitedFiles
using ProgressMeter

using Distributions

function test(;
             mu=1/8, lambda=1/8, 
             time_total = 14.0,
             N::Int=Int(1e5)
            )
    # define system
    I0 = 1 
    S0 = N - I0
    R0 = 0
    system = SIR{Exponential, Exponential}(Exponential(mu), Exponential(lambda), S0, I0, R0)
    rng = MersenneTwister(1000);
    reset!(system, rng)
    handler = EventQueue{Tuple{Int64,Float64}}(0.0)
    initialize!(handler, system)
    pass_update!(handler, event) = update!(handler, event, system)
    dT_sim = advance!(KineticMonteCarlo(), handler, pass_update!, Float64(time_total))
    return system.I
end

function run(path_out::String; 
             mu=1/8, lambda=1/8, 
             time_total = 14.0,
             N::Int=Int(1e5)
            )
    # define system
    I0 = 1 
    S0 = N - I0
    R0 = 0
    system = SIR{Exponential, Exponential}(Exponential(mu), Exponential(lambda), S0, I0, R0)

    simulation_queue!(system, time_total, 1000)
end

function simulation_queue!(system, t, num_trajectories)
    @showprogress 1 for i in 1:num_trajectories
        rng = MersenneTwister(seed+i);
        reset!(system, rng)
        handler = EventQueue{Tuple{Int64,Float64}}(0.0)
        initialize!(handler, system)
        pass_update!(handler, event) = update!(handler, event, system)
        dT_sim = advance!(KineticMonteCarlo(), handler, pass_update!, Float64(time_total))
    end
end

mutable struct SIR{P1,P2}
    rng::AbstractRNG
    P_dt_recovery::P1
    P_dt_infection::P2
    N::Int
    S::Int
    I::Int
    R::Int
    S0::Int
    I0::Int
    R0::Int
    
    function SIR{P1,P2}(P_dt_recovery::P1, P_dt_infection::P2, S0::Int, I0::Int, R0::Int, rng::AbstractRNG=Random.GLOBAL_RNG) where {P1,P2}
        N = S0 + I0 + R0
        new(rng, P_dt_recovery, P_dt_infection, N, S0, I0, R0, S0, I0, R0)
    end
end

function reset!(system::SIR, rng::AbstractRNG)
    system.rng = rng
    system.S = system.S0
    system.I = system.I0
    system.R = system.R0
end

# event = Tuple{event_id::int, recovery_time:Float64}

function update!(handler::AbstractEventHandlerTime{Tuple{Int64,Float64}}, 
                 event::Tuple{Int64,Float64}, 
                 system::SIR)
    if first(event) == 1 # recovery
        system.I -= 1
        system.R += 1
    elseif first(event) == 2 # infection
        # potential next infection of host
        time_recovery = last(event)
        next_infection(handler, system, time_recovery)
        # events from newly infected
        new_infected(handler, system)
        system.S -= 1
        system.I += 1
    else
        throw(UndefVarError(:event))
    end
    println(get_time(handler), " ", event, " ", system.I)
end

function initialize!(handler::AbstractEventHandlerTime{Tuple{Int64,Float64}}, system::SIR)
    for i in 1:system.I
        new_infected(handler, system)
    end
end

function new_infected(handler::AbstractEventHandlerTime{Tuple{Int64,Float64}}, system::SIR) 
    time_recovery = get_time(handler) + rand(system.rng, system.P_dt_recovery)
    add!(handler, (time_recovery, (1, time_recovery)))
    next_infection(handler, system, time_recovery)
end

function next_infection(handler::AbstractEventHandlerTime{Tuple{Int64,Float64}}, system::SIR, time_recovery) 
    time_next_infection = get_time(handler) + rand(system.rng, system.P_dt_infection)
    if time_next_infection < time_recovery
        add!(handler, (time_next_infection, (2, time_recovery)))
    end
end
