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

using Plots

function run(path_out::String; 
             mu=1/8, lambda=1/8, 
             time_total = 14.0,
             N::Int=Int(1e5)
            )
    # define system
    I0 = 1 
    S0 = N - I0
    R0 = 0
    system_queue = SIR{Exponential, Exponential}(Exponential(1/mu), Exponential(1/lambda), S0, I0, R0)
    system_rates = SIR_rates(mu, lambda, S0, I0, R0)

    I_max = 100
    range_I = 0:I_max
    hist_queue = Histogram(0:I_max+1)
    hist_rates = Histogram(0:I_max+1)

    println("original method with global rates")
    @time simulation_rates!(hist_rates, system_rates, time_total, 1000)
    println(hist_rates.weights[1:10])
    println("new method with local event times")
    @time simulation_queue!(hist_queue, system_queue, time_total, 1000)
    println(hist_queue.weights[1:10])

    #display(plot(hist_queue.weights - hist_rates.weights))
    plot()
    display(plot!(hist_queue, yscale = :log10))
    display(plot!(hist_rates, yscale = :log10))
end

function simulation_queue!(hist_I, system, time_total, num_trajectories; seed=1000)
    hist_I.weights .= 0
    @showprogress 1 for i in 1:num_trajectories
        rng = MersenneTwister(seed+i);
        reset!(system, rng)
        handler = EventQueue{Tuple{Int64,Float64}}(0.0)
        initialize!(handler, system)
        pass_update!(handler, event) = update!(handler, event, system)
        measure() = system.I
        dT_sim, obs_I = advance!(KineticMonteCarlo(), handler, pass_update!, measure, Float64(time_total), rng)
        if obs_I in hist_I.edges[1]
            hist_I[obs_I] += 1
        end
    end
end

function simulation_rates!(hist_I, system, time_total, num_trajectories; seed=1000)
    hist_I.weights .= 0
    @showprogress 1 for i in 1:num_trajectories
        rng = MersenneTwister(seed+i);
        reset!(system)
        rates = current_rates(system)
        pass_update!(rates, event) = update!(rates, event, system)
        measure() = system.I
        dT_sim, obs_I = advance!(KineticMonteCarlo(), rates, pass_update!, measure, Float64(time_total), rng)
        if obs_I in hist_I.edges[1]
            hist_I[obs_I] += 1
        end
    end
end

mutable struct SIR_rates
    mu::Float64
    lambda::Float64
    N::Int
    S::Int
    I::Int
    R::Int
    S0::Int
    I0::Int
    R0::Int
    
    function SIR_rates(mu, lambda, S0::Int, I0::Int, R0::Int, rng::AbstractRNG=Random.GLOBAL_RNG)
        N = S0 + I0 + R0
        new(mu, lambda, N, S0, I0, R0, S0, I0, R0)
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

function reset!(system, rng::AbstractRNG)
    system.rng = rng
    system.S = system.S0
    system.I = system.I0
    system.R = system.R0
end

function reset!(system::SIR_rates)
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
        # encounter leads randomly to new infection
        if rand(rng) < system.S/system.N
            new_infection(handler, system)
            system.S -= 1
            system.I += 1
        end
        # potential next encounter of host
        time_recovery = last(event)
        next_encounter(handler, system, time_recovery)
    else
        throw(UndefVarError(:event))
    end
end

function initialize!(handler::AbstractEventHandlerTime{Tuple{Int64,Float64}}, system::SIR)
    for i in 1:system.I
        new_infection(handler, system)
    end
end

function new_infection(handler::AbstractEventHandlerTime{Tuple{Int64,Float64}}, system::SIR) 
    time_recovery = get_time(handler) + rand(system.rng, system.P_dt_recovery)
    add!(handler, (time_recovery, (1, time_recovery)))
    next_encounter(handler, system, time_recovery)
end

function next_encounter(handler::AbstractEventHandlerTime{Tuple{Int64,Float64}}, system::SIR, time_recovery) 
    # time next encounter
    time_next_encounter = get_time(handler) + rand(system.rng, system.P_dt_infection)
    # only infect if within recovery and if the contact is susceptible 
    if time_next_encounter < time_recovery
        add!(handler, (time_next_encounter, (2, time_recovery)))
    end
end


####### for rates 
 
function update!(rates::AbstractVector, event::Int, system::SIR_rates)
    if event == 1 # recovery
        system.I -= 1
        system.R += 1
    elseif event == 2 # infection
        system.S -= 1
        system.I += 1
    else
        throw(UndefVarError(:event))
    end
    
    rates .= current_rates(system)
end


"""
evaluate current rates of SIR system
"""
function current_rates(system::SIR_rates)
    rate_recovery = system.mu * system.I
    rate_infection = system.lambda* system.I* system.S/system.N
    return MVector{2,Float64}(rate_recovery, rate_infection)
end
