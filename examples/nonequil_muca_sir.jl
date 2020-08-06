using MonteCarloX
using Random
using Distributions
using StatsBase
using LinearAlgebra
using StaticArrays

using Printf
using DelimitedFiles
using ProgressMeter


function test_mutable_trajectories(num_updates;
                                   seed_mc = 1000, seed_dyn = 1001)
    rng_mc = MersenneTwister(seed_mc);
    rng_dyn = MutableRandomNumbers(MersenneTwister(seed_dyn), mode=:dynamic)

    I0 = 1
    S0 = 0
    R0 = 0
    epsilon = 0
    mu = 1/8.
    lambda = mu 
    k = 0.02
    P_lambda = Gamma(k,lambda/k)
    system = SIR{typeof(P_lambda)}(rng_dyn, P_lambda, epsilon, mu, S0, I0, R0)

    time_meas = 14
    dT = 1
    list_T = collect(range(0, time_meas, step=dT))
    array_I = zeros(length(list_T), num_updates)
    list_S = zeros(length(list_T));
    list_R = zeros(length(list_T));
    @showprogress 1 for i in 1:num_updates
        # update (only updates, need to accept at some point)
        index = rand(rng_mc, 1:length(rng_dyn))
        old_rng = rng_dyn[index]
        rng_dyn[index] = rand(rng_mc)
        
        #think about how to implement the trajectory function to only
        #"reevaluate the simulation from the index case on -> requires storing
        #ALL events! In system?)
        MonteCarloX.reset(rng_dyn)
        trajectory!(rng_dyn, list_T, list_S, array_I[:,i], list_R, system)
    end

    return list_T, array_I
end

function trajectory!(rng, list_T, list_S, list_I, list_R, system)
    rates = current_rates(system)
    pass_update!(rates, index) = update!(rates, index, system, rng)
    
    list_S[1] = system.measure_S
    list_I[1] = system.measure_I
    list_R[1] = system.measure_R
    time_simulation = Float64(list_T[1])
    for i in 2:length(list_T)
        if time_simulation < list_T[i]
            dT = list_T[i] - time_simulation
            dT_sim = advance!(KineticMonteCarlo(), rng, rates, pass_update!, dT)
            time_simulation += dT_sim
        end
        list_S[i] = system.measure_S
        list_I[i] = system.measure_I
        list_R[i] = system.measure_R
    end
end


mutable struct SIR{D}
    epsilon::Float64
    mu::Float64
    P_lambda::D
    current_lambda::Vector{Float64}
    sum_current_lambda::Float64
    update_current_lambda::Int
    N::Int
    S::Int
    I::Int
    R::Int
    measure_S::Int
    measure_I::Int
    measure_R::Int
    
    function SIR{D}(rng::AbstractRNG, P_lambda::D, epsilon, mu, S0::Int, I0::Int, R0::Int) where D
        N = S0 + I0 + R0
        current_lambda = rand(rng, P_lambda, I0)
        sum_current_lambda = sum(current_lambda)
        new(epsilon, mu, P_lambda, current_lambda, sum_current_lambda, 0, N, S0, I0, R0, S0, I0, R0)
    end
end

"""
update rates and system according to the last event that happend:
recovery:  1
infection: 2
"""
function update!(rates::AbstractVector, index::Int, system::SIR, rng::AbstractRNG)
    system.measure_S = system.S
    system.measure_I = system.I
    system.measure_R = system.R
    if index == 1 # recovery
        random_I = rand(rng, 1:system.I)
        delete_from_system!(system, random_I)
        system.I -= 1
        system.R += 1
    elseif index == 2 # infection
        add_to_system!(system, rand(rng, system.P_lambda))
        system.S -= 1
        system.I += 1
        @assert length(system.current_lambda) == system.I
    elseif index == 0 # absorbing state of zero infected
        system.measure_S = system.S
        system.measure_I = system.I
        system.measure_R = system.R
    else
        throw(UndefVarError(:index))
    end
    
    rates .= current_rates(system)
end

function delete_from_system!(system, index)
    update_sum!(system, -system.current_lambda[index])
    deleteat!(system.current_lambda, index)
end

function add_to_system!(system, lambda)
    push!(system.current_lambda, lambda)
    update_sum!(system, lambda)
end

function update_sum!(system, change) 
    system.sum_current_lambda += change 
    system.update_current_lambda += 1
    if system.update_current_lambda > 1e5
        system.sum_current_lambda = sum(system.current_lambda)
        system.update_current_lambda = 0
    end
end

"""
evaluate current rates of SIR system
"""
function current_rates(system::SIR)
    rate_recovery = system.mu * system.I
    rate_infection = system.sum_current_lambda* system.S/system.N + system.epsilon
    return MVector{2,Float64}(rate_recovery, rate_infection)
end
