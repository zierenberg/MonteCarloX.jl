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

function run(path_out::String; mu=1/8, lambda=1/8, epsilon = 0.0, pop=1e6)
    # define system
    I0 = 1 
    S0 = Int(pop) - I0
    R0 = 0
    system = SIR(lambda, mu, epsilon, S0, I0, R0)

    simulation_naive!(system, t, 1000)
end

function simulation_naive!(system, t, num_trajectories)
    @showprogress 1 for i in 1:num_trajectories
        rng = MersenneTwister(seed+i);
        reset!(system)
        handler = EventQueue{Tuple{Int64,Float64}}
        pass_update!(handler, time, event) = update!(handler, time, event, system)
        dT_sim = advance!(KineticMonteCarlo(), handler, pass_update!, Float64(t), rng)
    end
end

mutable struct SIR
    lambda::Float64
    mu::Float64
    epsilon::Float64
    N::Int
    S::Int
    I::Int
    R::Int
    S0::Int
    I0::Int
    R0::Int
    measure_S::Int
    measure_I::Int
    measure_R::Int
    
    function SIR(lambda, mu, epsilon, S0::Int, I0::Int, R0::Int)
        N = S0 + I0 + R0
        new(lambda, mu, epsilon, N, S0, I0, R0, S0, I0, R0, S0, I0, R0)
    end
end

function reset!(system::SIR)
    system.S = system.measure_S = system.S0
    system.I = system.measure_I = system.I0
    system.R = system.measure_R = system.R0
end

# event = Tuple{event_id::int, recovery_time:Float64}

function update!(event_handler::AbstractEventHandlerTime{Tuple{Int64,Float64}}, time::Float64, event::Tuple{Int64,Float64}, system::SIR)
    system.measure_S = system.S
    system.measure_I = system.I
    system.measure_R = system.R
    
    if event == 1 # recovery
        system.I -= 1
        system.R += 1
    elseif event == 2 # infection
        new_event_time = time - (1/system.lambda)*log(fabs(rand(rng)))
        recovery_time = second(event)
        if new_event_time < recovery_time
            add!(event_handler, new_event_time, (2, recovery_time))
        end
        new_event_time = time - (1/system.lambda)*log(fabs(rand(rng)))
        recovery_time = time - (1/system.mu)*log(fabs(rand(rng)))
        add!(event_handler, recovery_time, (1, recovery_time))
        if new_event_time < recovery_time
            add!(event_handler, new_event_time, (2, recovery_time))
        end
        system.S -= 1
        system.I += 1
    elseif event == 0 # absorbing state of zero infected
        
    else
        throw(UndefVarError(:event))
    end
end
