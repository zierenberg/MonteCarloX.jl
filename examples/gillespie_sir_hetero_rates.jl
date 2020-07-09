using MonteCarloX
using Random
using Distributions
using StatsBase
using Printf
using ProgressMeter
using LinearAlgebra
using DelimitedFiles
using StaticArrays

function example_dist_sir_hetero_rates(path; 
                                      ratio_lambda_mu=1.0)
    # define default values (some specific initial conditions, I0=100)
    N = Int(1e5)
    list_I0 = Int[1,10,100]
    seeds = collect(Int, 1:1e6);
    
    # motivated by COVID-19 parameters [cf. Dehning et al, Science (2020)]
    mu = 1/8. # avg. infectious time of 5+3 days  
    lambda_0 = ratio_lambda_mu*mu # focus on the "critical point" of sustained activity
    sigma = mu/10
    time_meas = 21 # 1 week observation of effect of initial perturbation
    list_k = [1,1e-1,1e-2]
    list_alpha = [1.5,2,3] # mean only defined for alpha > 1

    # Different distributions with same mean speading rate lambda_0
    list_P = []
    list_P_name = []
    # Compared to [Lloyd-Smith et al., Science (2020)] delta-distribution for
    # differential-equation model corresponds already to an exponential
    # distribution of R_0 i.e. a geometric offspring distribution (k=1)
    P_Delta = Normal(lambda_0, 0.0)
    push!(list_P, P_Delta)
    push!(list_P_name, "Delta")
    P_Gaussian = Normal(lambda_0, sigma)
    push!(list_P, P_Gaussian)
    push!(list_P_name, "Gaussian")
    # We cannot fully reproduce the negative-binomial offspring distribution by
    # a Gamma-Poisson mixture because we have no generation based model here.
    # Still, we should approximate the same behavior by choosing a Gamma
    # distribution of spreading rates with the same mean
    # https://en.wikipedia.org/wiki/Gamma_distribution
    #   mean(Gamma) = k*theta = lambda_0 i.e. theta = lambda_0/k
    for k in list_k
        P_Gamma = Gamma(k, lambda_0/k) 
        push!(list_P, P_Gamma)
        push!(list_P_name, @sprintf("Gamma_k%.2e",k))
    end
    # Pareto distribution for scale-free spreading rates (motivated from
    # scale-free-ness of social networks)
    # P_Pareto(x_m, alpha) = GeneralizedPareto(x_m, x_m/alpha, 1/alpha)
    # for Pareto to give mean=lambda_0 : x_m(lambda_0,alpha) = (alpha-1)lambda_0/alpha
    for alpha in list_alpha
        x_m = (alpha-1)*lambda_0/alpha
        P_Pareto = GeneralizedPareto(x_m, x_m/alpha, 1/alpha)
        push!(list_P, P_Pareto)
        push!(list_P_name, @sprintf("Pareto_alpha%.2e",alpha))
    end
    # mean(lambda) = lambda_1*prior_1 + lambda_2*prior_2 = lambda_0
    # prior_1 + prior_2 = 1
    lambda_1 =  0.0
    lambda_2 = 10.0
    prior_1 = (lambda_0 - lambda_2)/(lambda_1 - lambda_2)
    prior_2 = 1 - prior_1
    P_Bimodal = MixtureModel(Normal[Normal(lambda_1, 0), Normal(lambda_2, 0)], [prior_1, prior_2])
    push!(list_P, P_Bimodal)
    push!(list_P_name, @sprintf("Bimodal_left%.2e_right%.2e",lambda_1, lambda_2))
    # simulations
    for I0 in list_I0
        R0 = 0
        S0 = N-I0
        for (P,P_name) in zip(list_P, list_P_name)
            println(N, " ", I0, " ", P_name)
            list_time, list_dist = distribution(P, S0, I0, R0, time_meas, seeds, mu=mu); 
            filename_base =  @sprintf("%s/distribution_%s_lambda%.2e_mu%.2e_N%.2e_I0%.2e",
                                 path, P_name, lambda_0, mu, N, I0)
            write_distributions(list_time, list_dist, filename_base)
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

function sir_hetero_rates(P_lambda::D, S0, I0, R0, time_meas, seed;
                          mu=1.0, epsilon=0.0, dT=1) where D
    rng = MersenneTwister(seed);
    system = SIR{D}(rng, P_lambda, epsilon, mu, S0, I0, R0)

    list_T = collect(range(0, time_meas, step=dT))
    list_S = zeros(length(list_T));
    list_I = zeros(length(list_T));
    list_R = zeros(length(list_T));
    trajectory!(rng, list_T, list_S, list_I, list_R, system)

    return list_T, list_S, list_I, list_R, system
end

# memory-wise this is not the most efficient solution
# Alternative is to have one trajectory readout and directly add to
# distribution (which requires some a priori size though)
function distribution(P_lambda::D, S0, I0, R0, time_meas, seeds;
                      mu=1.0, epsilon=0, dT=1) where D

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
        min_I = 0
        max_I = maximum(trajectories_I[t,:])+2
        step = 1
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
    update_sum(system, lambda)
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
