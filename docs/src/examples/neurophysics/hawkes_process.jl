# %% #src
import Pkg                                          #src
Pkg.activate(joinpath(@__FILE__, "../../../../"))   #src
Pkg.instantiate()                                   #src
include(joinpath(@__DIR__, "..", "defaults.jl"))    #src

# # Hawkes Process (Self-Exciting Point Process)
#
# The Hawkes process is a self-exciting point process where each event
# increases the instantaneous rate, which then decays exponentially:
#
# ```math
# \lambda(t) = \lambda_0 + \sum_{t_i < t} \alpha \exp\!\left(-\frac{t - t_i}{\tau}\right)
# ```
#
# The process is stationary iff the branching ratio ``n = \alpha/\tau < 1``,
# giving mean intensity ``\langle\lambda\rangle = \lambda_0\,/\,(1 - \alpha/\tau)``.
#
# We simulate using the inhomogeneous Poisson **thinning algorithm**
# (Lewis & Shedler 1979) via MonteCarloX's `next_time`.

using Random, Plots, Statistics
using MonteCarloX: next_time

# ## Simulation

function hawkes_thinning(τ, λ0, α; T_max=100.0, seed=42)
    rng     = MersenneTwister(seed)
    samples = Float64[]
    while true
        t_now = isempty(samples) ? 0.0 : samples[end]

        # Excitation at t_now⁺ from all past events — this is the peak of λ
        # going forward, since all terms decay with elapsed time s > 0.
        # next_time(rng, λ_elapsed, λ_max) uses elapsed time s as argument.
        excitation = sum(α * exp(-(t_now - t) / τ) for t in samples; init=0.0)
        λ_max      = λ0 + excitation

        # Rate as a function of elapsed time s since t_now
        λ_elapsed(s) = λ0 + excitation * exp(-s / τ)

        t_next = t_now + next_time(rng, λ_elapsed, λ_max)
        t_next > T_max && break
        push!(samples, t_next)
    end
    return samples
end

# ## Parameters and analytical benchmarks

τ, λ0, α = 1.0, 0.2, 0.5          # branching ratio α/τ = 0.5 < 1 (stationary regime)


λ_ana  = λ0 / (1 - α/τ)           # mean intensity (h/(1-m))
Δt_ana = 1 / λ_ana                # mean inter-event time

T_max     = 10_000 * Δt_ana        # simulate for long enough to get good statistics

samples = hawkes_thinning(τ, λ0, α; T_max)

λ_emp  = length(samples) / T_max
Δt_emp = mean(diff(samples))

println("          analytical   empirical")
println("  λ     :  $(round(λ_ana;  digits=3))        $(round(λ_emp;  digits=3))")
println("  ⟨Δt⟩  :  $(round(Δt_ana; digits=3))        $(round(Δt_emp; digits=3))")

# ## Plots

# Intensity trace with events
t_grid = range(max(0,T_max-100), T_max, length=1000)
print(t_grid)
λ_grid = [λ0 + sum(α * exp(-(t - t0) / τ) for t0 in samples if t0 < t; init=0.0) for t in t_grid]

p1 = plot(t_grid, λ_grid; color=:steelblue, lw=1.5, label="λ(t)",
          xlabel="Time", ylabel="λ(t)", legend=:topright,
          size=(700, 250), margin=4Plots.mm)
# set xrange
xlims!(p1, minimum(t_grid), maximum(t_grid))
hline!(p1, [λ_ana]; color=:green, ls=:dash, lw=1.5, label="⟨λ⟩ analytical")
vline!(p1, samples; color=:crimson, alpha=0.3, lw=0.8, label=nothing)

# Inter-event time histogram
# (Hawkes Δt is not exponential due to clustering; exponential shown for reference only)
dt     = diff(samples)
t_plot = range(0, maximum(dt) * 1.5, length=200)

p2 = histogram(dt; bins=30, normalize=:pdf, alpha=0.6, color=:crimson,
               label="Simulated Δt",
               xlabel="Inter-event time Δt", ylabel="Density",
               size=(700, 250), margin=4Plots.mm)
plot!(p2, t_plot, λ_ana .* exp.(-λ_ana .* t_plot);
      color=:green, ls=:dash, lw=2, label="Exp(λ_ana) — reference")

plot(p1, p2; layout=(2,1), size=(700, 520))