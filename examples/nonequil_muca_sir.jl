# proof-of-principle implementation of non-equilibrium muca for standard SIR dynamics
using MonteCarloX
using Random
using Distributions
using StatsBase
using LinearAlgebra
using StaticArrays

using Printf
using DelimitedFiles
using ProgressMeter

function generate_data(path_out::String;
                       do_muca::Bool=true,
                       do_naive::Bool=false,
                       mu=1/8, t=14, epsilon = 0.0,
                       num_iterations::Int = 10,
                       num_updates_iteration::Int = Int(1e6),
                       num_updates_production::Int = Int(1e7)
                      )
    # define system
    I0 = 1 
    S0 = Int(1e5) - I0
    R0 = 0
    lambda = mu
    system = SIR(lambda, mu, epsilon, S0, I0, R0)

    # define muca bounds
    I_max = 300
    I_bound = 100
    range_I = 0:I_max
    hist = Histogram(0:I_max+1)
    logW = Histogram(0:I_max+1, Float64)

    # i-o stuff
    list_hist = zeros(length(hist.weights), num_iterations)
    list_logW = zeros(length(logW.weights), num_iterations)

    # total time of evaluation on my machine: 0:02:01
    if do_muca
        display(plot())
        # iteration with histograms & weights written to file
        for iteration in 1:num_iterations
            println("iteration ", iteration)
            #store weights
            list_logW[:,iteration] .= logW.weights

            muca_iteration!(hist, logW, system, t, num_updates_iteration)

            # store histogram and weights
            list_hist[:,iteration] .= hist.weights

            display(plot!(log.(hist.weights), xlims=(0,120)))
            # update weights
            if iteration != num_iterations
                muca_update!(logW, hist, I_bound)
            end
        end
        # write histogram and weights
        filename = @sprintf("%s/pop_muca_sir_critical_iterations_hist.dat",path_out)
        open(filename,"w") do fout
            write(fout, "#I\t H^1(I)\t ... \n")
            writedlm(fout, [collect(range_I) list_hist])
        end
        filename = @sprintf("%s/pop_muca_sir_critical_iterations_logW.dat",path_out)
        open(filename,"w") do fout
            write(fout, "#I\t ln(W^1(I))\t ...\n")
            writedlm(fout, [collect(range_I) list_logW])
        end
         
        # production run with final histogram and final reweighting
        println("production run  ")
        muca_iteration!(hist, logW, system, t, num_updates_production)
        log_dist = log.(hist.weights) .- logW.weights 
        norm = sum(exp.(log_dist))
        log_dist = log_dist .- log(norm)
        filename = @sprintf("%s/pop_muca_sir_critical_production.dat",path_out)
        open(filename,"w") do fout
            write(fout, "#I\t P(I)\t ln(W(I))\t H(I)\n")
            writedlm(fout, [collect(range_I) exp.(log_dist) logW.weights hist.weights])
        end
    end

    # analytic solution for comparisson
    dist = distribution_analytic_critical(range_I, mu, t)
    filename = @sprintf("%s/pop_muca_sir_critical_analytical.dat",path_out)
    open(filename,"w") do fout
        write(fout, "#I\t P(I)\n")
        writedlm(fout, [collect(range_I) dist])
    end
    
    # naive estimation with same "computing time" (skip for now because muca is not very efficient)
    # Time of evaluation on my machine: 0:03:35
    if do_naive
        num_trajectories = Int(1e7)
        simulation_naive!(hist, system, t, num_trajectories)
        filename = @sprintf("%s/pop_muca_sir_critical_naive.dat",path_out)
        open(filename,"w") do fout
            write(fout, "#I\t P(I)\n")
            writedlm(fout, [collect(range_I) hist.weights./num_trajectories])
        end
    end
end

"""
Analytical solution frrom Raúl Toral (online lecture course)
https://ifisc.uib-csic.es/raul/CURSOS/SP/Master_equations.pdf
Eq. (5.31)-(5.32)
"""
function distribution_analytic_critical(range_I, mu, t) 
    b_t = mu*t

    dist = zeros(length(range_I))
    for (i,I) in enumerate(range_I)
        if I==0
            dist[i] = b_t/(1+b_t)
        else
            dist[i] = b_t^(I - 1) / ((1+b_t)^(I + 1))
        end
    end
    return dist
end

function simulation_naive!(hist_I, system, t, num_trajectories;
                            dT::Float64 = 1.0,
                            seed = 1000)
    hist_I.weights .= 0
    @showprogress 1 for i in 1:num_trajectories
        rng = MersenneTwister(seed+i);
        reset!(system)
        # evolution
        rates = current_rates(system)
        pass_update!(rates, event) = update!(rates, event, system)
        dT_sim = advance!(KineticMonteCarlo(), rng, rates, pass_update!, Float64(t))
        if system.measure_I in hist_I.edges[1]
            hist_I[system.measure_I] += 1
        end
    end
end

function muca_iteration!(hist_I, logW, system, t, num_updates; 
                        dT::Float64 = 1.0,
                        epsilon::Float64 = 0.0,
                        updates_therm::Int = Int(1e5),
                        seed_mc = 1000, seed_dynamics = 2000)
    rng_mc = MersenneTwister(seed_mc);
    rng_dynamics = MutableRandomNumbers(MersenneTwister(seed_dynamics), mode=:dynamic)

    current_I = system.I
    hist_I.weights .= 0
    @showprogress 1 for i in 1:(updates_therm + num_updates)
        # update (todo: make incremental update of rng) 
        rand_index = rand(rng_mc, 1:length(rng_dynamics))
        old_rng = rng_dynamics[rand_index]
        rng_dynamics[rand_index] = rand(rng_mc)
        
        # reset dynamics and rerun (has a LOT OF POTENTIAL for optimization)
        MonteCarloX.reset(rng_dynamics)
        reset!(system)
        rates = current_rates(system)
        pass_update!(rates, event) = update!(rates, event, system)
        dT_sim = advance!(KineticMonteCarlo(), rng_dynamics, rates, pass_update!, Float64(t))
        new_I = system.measure_I

        # acceptance
        accept = false
        if new_I in logW.edges[1]
            if rand(rng_mc) < exp(logW[new_I]-logW[current_I])
                #accept
                current_I = new_I
                accept = true
            end
        end
        if !accept
            rng_dynamics[rand_index] = old_rng
        end
        if i > updates_therm
            hist_I[current_I] += 1
        end
    end
end

"""
Naive muca update: 
W^{n_1}(I) = W^{n}(I)/H(I) for all I in range [0,I_bound)
if H(I) > 10 [arbitrary choice] 
"""
function muca_update!(logW, hist_I, I_bound)
    for I in logW.edges[1]
        if I < I_bound && hist_I[I] > 1
            logW[I] = logW[I] - log(hist_I[I])
        elseif I >= I_bound
            logW[I] = logW[I-1]
        end
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

function update!(rates::AbstractVector, event::Int, system::SIR)
    system.measure_S = system.S
    system.measure_I = system.I
    system.measure_R = system.R
    if event == 1 # recovery
        system.I -= 1
        system.R += 1
    elseif event == 2 # infection
        system.S -= 1
        system.I += 1
    elseif event == 0 # absorbing state of zero infected
        
    else
        throw(UndefVarError(:event))
    end
    
    rates .= current_rates(system)
end


"""
evaluate current rates of SIR system
"""
function current_rates(system::SIR)
    rate_recovery = system.mu * system.I
    rate_infection = system.lambda* system.I* system.S/system.N + system.epsilon
    return MVector{2,Float64}(rate_recovery, rate_infection)
end
