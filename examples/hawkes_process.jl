using MonteCarloX
using Random
using Plots


""" Self-exciting process. Rate jumps up by x_increase on event and subsequently
decays to the baserate with timeconstant tau."""
function hawkes_process(tau::Float64 = 1.0, baserate::Float64 = 0.2, x_increase::Float64 = 0.001;
    plotflag::Bool = false, nSamples::Int = 30, rng::AbstractRNG = MersenneTwister(1000))
    
    next_sample = InhomogeneousPoissonProcess.next_event_time_for_piece_wise_decreasing_rate

    decay(t, t0, x0) = x0 * exp(-(t - t0) / tau)
    Lambda(t, t0, x0) = baserate + decay(t, t0, x0)

    samples = zeros(nSamples)

    t0 = 0.0
    x0 = 0.0
    for i in 1:nSamples
        t_previous = t0
        t0 += next_sample(dt->Lambda(dt, 0.0, x0), rng)
        # The current exponential decays starting value x0 is 
        # the decay up to now (t0) plus the increase 
        x0 = decay(t0, t_previous, x0) + x_increase
        samples[i] = t0
    end

    if plotflag
        # For plotting we are inefficient; (t > t0) means only causal influence
        ratefun(t) = sum([(t > t0) * decay(t, t0, x_increase) for t0 in samples])
        maxT = samples[end] + 4 * tau
        plotrange = range(0.0, stop = maxT, length = 2000)
        plot(plotrange, map(ratefun, plotrange), grid = false)
        display(vline!(samples)) # plot lines when events happen
    end

    return samples
end


