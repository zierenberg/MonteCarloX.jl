using MonteCarloX
using Random
using Distributions
using StatsBase

function example_sir_hetero_rates()
    # define default values (some specific initial conditions, I0=100)
    
    # generate distribution of trajectories for different distributions but same mean_lambda
    # A: delta_peak
    # B: Gaussian
    # C: Gamma
     
    # plot different distributions over time
end

function sir_hetero_rates(P_lambda::D, S0, I0, R0, time_meas, seed,
                          mu=1.0, epsilon=1e-3, dT=1) where D
    rng = MersenneTwister(seed);
    system = SIR{D}(rng, P_lambda, epsilon, mu, S0, I0, R0)

    list_T = collect(range(0, time_meas, step=dT))
    list_S = zeros(length(list_T));
    list_I = zeros(length(list_T));
    list_R = zeros(length(list_T));
    trajectory!(rng, list_T, list_S, list_I, list_R, system)

    return list_T, list_S, list_I, list_R, system
end

"""
# generate a stochastic trajectory
SIR dynamics is goverened by differential equation
``
    \\frac{dI}{dt} = \\sum_{i=1}^{I}\\lambda_i\\frac{S}{N} - \\mu I
``

# Arguments
* distribution for the heterogeneous rates P(lambda)
"""
function trajectory!(rng, list_T, list_S, list_I, list_R, system)
    rates = current_rates(system)
    pass_update!(rates, index) = update!(rates, index, system, rng)
    
    list_S[1] = system.measure_S
    list_I[1] = system.measure_I
    list_R[1] = system.measure_R
    dT_correction = 0.0
    for i in 2:length(list_T)
      dT = list_T[i]-list_T[i-1] - dT_correction
      dT_true = advance!(KineticMonteCarlo(), rng, rates, pass_update!, dT)
      dT_correction = dT_true - dT
      # Gillespie advance goes until time > list_T[i]
      list_S[i] = system.measure_S
      list_I[i] = system.measure_I
      list_R[i] = system.measure_R
    end
end


# D hopefully works as distribution template ;)
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
        delete_from_system(system, random_I)
        system.I -= 1
        system.R += 1
    elseif index == 2 # infection
        add_to_system(system, rand(rng, system.P_lambda))
        system.S -= 1
        system.I += 1
        @assert length(system.current_lambda) == system.I
    else
        throw(UndefVarError(:index))
    end
    
    rates .= current_rates(system)
end

function delete_from_system(system, index)
    update_sum(system, -system.current_lambda[index])
    deleteat!(system.current_lambda, index)
end

function add_to_system(system, lambda)
    push!(system.current_lambda, lambda)
    update_sum(system, lambda)
end

function update_sum(system, change) 
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
    return [rate_recovery, rate_infection]
end
