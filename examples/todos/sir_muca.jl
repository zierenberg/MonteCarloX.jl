# %% #src
import Pkg                                          #src
Pkg.activate(joinpath(@__FILE__, "../../../../"))   #src
Pkg.instantiate()                                   #src
include(joinpath(@__DIR__, "..", "defaults.jl"))    #src

# # SIR Model with Multicanonical Sampling
#
# Multicanonical sampling for the SIR model at criticality to efficiently
# sample rare extinction events and the full infected distribution P(I).

using Random, Plots, Statistics
using MonteCarloX

# ## System definition

mutable struct SIR{D<:AbstractFloat} <: AbstractSystem
    lambda::D
    mu::D
    epsilon::D
    N::Int
    S::Int
    I::Int
    R::Int
end

function SIR(lambda, mu, epsilon, S0::Int, I0::Int, R0::Int)
    N = S0 + I0 + R0
    return SIR(lambda, mu, epsilon, N, S0, I0, R0)
end

MonteCarloX.event_source(sys::SIR) = [sys.mu * sys.I, sys.lambda * sys.I * sys.S/sys.N + sys.epsilon]

function MonteCarloX.modify!(sys::SIR, event::Int, t)
    if event == 1
        sys.I -= 1
        sys.R += 1
    elseif event == 2
        sys.S -= 1
        sys.I += 1
    end
    return nothing
end

# ## Multicanonical update

function muca_update!(logW, hist, I_bound)
    for I in eachindex(hist)
        if I < I_bound && hist[I] > 1
            logW[I] -= log(hist[I])
        elseif I >= I_bound
            logW[I] = logW[I-1]
        end
    end
end

# ## One multicanonical iteration

function muca_iteration!(hist, logW, sys, t, num_updates; updates_therm=1000, seed_mc=1000, seed_dynamics=2000)
    rng_mc = MersenneTwister(seed_mc)
    current_I = sys.I
    hist .= 0
    
    for i in 1:(updates_therm + num_updates)
        sys.S = sys.N - sys.I - sys.R
        sys.I = sys.I
        sys.R = 0
        
        alg = Gillespie(MersenneTwister(seed_dynamics + i))
        advance!(alg, sys, t)
        
        new_I = sys.I
        
        if new_I in eachindex(logW)
            if rand(rng_mc) < exp(logW[new_I] - logW[current_I])
                current_I = new_I
            end
        end
        
        if i > updates_therm
            hist[current_I] += 1
        end
    end
    
    return current_I
end

# ## Parameters

const CI_MODE = get(ENV, "MCX_SMOKE", get(ENV, "MCX_CI", "false")) == "true"

I0 = 1
S0 = CI_MODE ? 100 : Int(1e4)
N = S0 + I0
R0 = 0
lambda = 1.0
mu = 1.0
t = 10.0
I_max = CI_MODE ? 30 : 200

range_I = 0:I_max
hist = zeros(Int, length(range_I))
logW = zeros(Float64, length(range_I))

# ## Multicanonical iterations

num_iterations = CI_MODE ? 3 : 10
num_updates_iteration = CI_MODE ? 1000 : Int(1e5)

println("Running $num_iterations multicanonical iterations...")
println("  I_max      = $I_max")
println("  updates    = $num_updates_iteration per iteration")

for iteration in 1:num_iterations
    println("  Iteration $iteration / $num_iterations")
    
    iteration == 1 && println("    logW(I=10): $(logW[10+1])")
    
    current_I = muca_iteration!(hist, logW, SIR(lambda, mu, 0.0, S0, I0, R0),
                                t, num_updates_iteration;
                                updates_therm=1000,
                                seed_mc=1000+iteration,
                                seed_dynamics=2000+iteration)
    
    iteration < num_iterations && muca_update!(logW, hist, I_max÷2)
end

# ## Production run

num_updates_production = CI_MODE ? 5000 : Int(1e6)
println("\nProduction run with $num_updates_production updates...")
muca_iteration!(hist, logW, SIR(lambda, mu, 0.0, S0, I0, R0),
                t, num_updates_production;
                updates_therm=0, seed_mc=3000, seed_dynamics=4000)

# ## Reweight distribution

log_dist = log.(hist) .- logW
log_dist .-= log(sum(exp.(log_dist)))

# ## Analytical solution

function distribution_analytic_critical(range_I, mu, t)
    b_t = mu * t
    dist = zeros(length(range_I))
    for (i, I) in enumerate(range_I)
        if I == 0
            dist[i] = b_t / (1 + b_t)
        else
            dist[i] = b_t^(I - 1) / ((1 + b_t)^(I + 1))
        end
    end
    return dist
end

dist_analytical = distribution_analytic_critical(range_I, mu, t)

# ## Plot

p = plot(xlabel="Infected I", ylabel="Probability P(I)",
         title="SIR at criticality (λ=μ=$mu, t=$t)", legend=:best, grid=true,
         size=(700, 280), margin=5Plots.mm)

plot!(p, range_I, exp.(log_dist); label="Multicanonical", color=:crimson, lw=3)
plot!(p, range_I, dist_analytical; label="Analytical", color=:black, lw=2, ls="--")

xlims!(p, -1, min(30, I_max + 1))

println("\nResults:")
println("  P(extinction) = $(exp(log_dist[1]))")
println("  Mean I        = $(sum(range_I .* exp.(log_dist)))")

p
