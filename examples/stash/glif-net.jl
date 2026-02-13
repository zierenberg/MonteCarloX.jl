using MonteCarloX
using Random

""" Network of generalized LIF-neurons, connected via matrix W and global input.
For plotting 'Plots' has to be imported."""
function glif_net(W::Array{Float64,2}; 
    tau::Float64 = 1.0, sigma::Float64 = 1.0, threshold::Float64 = 1.0,
    plotflag::Bool = false, nSamples::Int = 10, rng::AbstractRNG = MersenneTwister(1000))
    
    alg = InhomogeneousPoisson()

    # LIF potential decreases exponentially from last
    # potential u0 at t0=0.0
    decay(dt, u0) = u0 * exp(-dt / tau)
    lambda(dt, u0) = exp((decay(dt, u0) - threshold) / sigma)

    spikes = zeros(nSamples)
    spike_indices = zeros(Int, nSamples)

    t0 = 0.0
    n = size(W, 1)
    u0s = zeros(n) # neuron potentials
    min_rate = exp(-threshold)
    for i in 1:nSamples
        t_previous = t0
        lambdas = dt->[lambda(dt, u0) for u0 in u0s]

        # potentials will always decrease once bigger than the bias input
        # we can therefore choose max rate from sum of current single rates
        max_rate = sum(map(l->max(l, min_rate), lambdas(0.0)))
        
        (dt, ind) = next(alg, rng, lambdas, max_rate)
        
        t0 += dt
        # get prespike potentials
        u0s = map(u0->decay(t0 - t_previous, u0), u0s)
        # increase voltages by spike input
        u0s += W[:,ind]

        spikes[i] = t0
        spike_indices[i] = ind
    end

    if plotflag
        plots = []
        maxT = spikes[end] + 5 * tau
        n_plots = min(5, n)

        for i in 1:n_plots
            # For plotting we are inefficient; (t > t0) means only causal influence
            # spike contribution from spiking neuron ind to receiving neuron i
            ratefun(t) = sum([(t > t0) * decay(t - t0, W[i,ind]) for 
                                                (t0, ind) in zip(spikes, spike_indices)])
            plotrange = range(0.0, stop = maxT, length = 2000)

            p = plot(plotrange, map(ratefun, plotrange), grid = false, legend = false)
            # plot lines when events happen
            if sum(spike_indices .== i) > 0
                vline!(spikes[spike_indices .== i], linealpha = 0.5, linewidth = 0.1) 
            end

            push!(plots, p)
        end

        display(plot(plots..., layout = grid(n_plots, 1)))
    end

    return spikes, spike_indices
end


using Plots
using LinearAlgebra
W = randn(10,10) - Matrix{Float64}(I,10,10)*3
glif_net(W,plotflag=true)