using MonteCarloX
using Random
using Distributions
using StatsBase
using Printf
using ProgressMeter
using LinearAlgebra
using DelimitedFiles
using StaticArrays

function example_dist_sir_hetero_rates(path)
    # define default values (some specific initial conditions, I0=100)
    list_N  = Int[1e4]
    list_I0 = Int[10,100]
    
    epsilon=1e-3
    mu=1.0
    lambda_0 = 1.0
    sigma    = 0.2
    seeds = collect(Int, 1:1e4);
    time_meas = 200
    num_bins_max = 100
    lambda_1 = 0.1

    # generate distribution of trajectories for different distributions but same mean_lambda
    P_Delta = Normal(lambda_0, 0.0)
    P_Gaussian = Normal(lambda_0, 0.1)
    P_Gamma = Gamma(lambda_0, 1) #check that mean=a*theta is lambda_0
    P_Bimodal = MixtureModel(Normal[Normal(lambda_1,0.01), Normal(2*lambda_0-lambda_1,0.01)])
    list_P = [P_Delta, P_Gaussian, P_Gamma, P_Bimodal]
    list_P_name = ["Delta", "Gaussian", "Gamma", "Bimodal"]
    for N in list_N
        for I0 in list_I0
            R0 = 0
            S0 = N-I0
            for (P,P_name) in zip(list_P, list_P_name)
                println(N, " ", I0, " ", P_name)
                list_time, list_dist = distribution(num_bins_max, P, S0, I0, R0, time_meas, seeds, mu=mu, epsilon=epsilon); 
                filename_base =  @sprintf("%s/distribution_%s_lambda%.2d_mu%.2e_epsilon%.2e_N%.2e_I0%.2e",
                                     path, P_name, lambda_0, mu, epsilon, N, I0)
                write_distributions(list_time, list_dist, filename_base)
            end
        end
    end
end

function write_distributions(list_time, list_dist, filename_base)
    for (time,dist) in zip(list_time, list_dist)
        filename = @sprintf("%s_T%.2e.dat",filename_base,time)
        open(filename; write=true) do f 
            write(f, "#I\t P(I)\n")
            writedlm(f, zip(collect(dist.edges[1]),dist.weights)) 
        end
    end
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

function distribution(num_bins_max::Int, P_lambda::D, S0, I0, R0, time_meas, seeds;
                      mu=1.0, epsilon=1e-3, dT=1) where D

    list_T = collect(range(0, time_meas, step=dT))
    trajectories_S = [zeros(Int,length(list_T)) for i=1:length(seeds)]
    trajectories_I = [zeros(Int,length(list_T)) for i=1:length(seeds)]
    trajectories_R = [zeros(Int,length(list_T)) for i=1:length(seeds)]
    @showprogress 1 for (i,seed) in enumerate(seeds)
        rng = MersenneTwister(seed);
        system = SIR{D}(rng, P_lambda, epsilon, mu, S0, I0, R0)
        trajectory!(rng, list_T, trajectories_S[i], trajectories_I[i], trajectories_R[i], system)
    end
    #index [time,trajectory] -> a way to make this [trajectory,time] to use later trajectories[:,t]
    trajectories_S = hcat(trajectories_S...); 
    trajectories_I = hcat(trajectories_I...); 
    trajectories_R = hcat(trajectories_R...); 
    
    # then fit distributions (for now only I) 
    # TODO: generalize (function for this) to all cases
    list_dist = []
    for t=1:size(trajectories_I,1)
        min_I = minimum(trajectories_I[t,:])-10
        max_I = maximum(trajectories_I[t,:])+10
        step = ceil((max_I-min_I)/num_bins_max)
        hist = fit(Histogram, trajectories_I[t,:], min_I:step:max_I)
        dist = normalize!(float(hist))
        push!(list_dist, dist)
    end

    return list_T, list_dist
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
    return MVector{2,Float64}(rate_recovery, rate_infection)
end
