# tests for kinetic_monte_carlo.jl
using MonteCarloX
using StatsBase
using Random
using LinearAlgebra

include("includes.jl")

function test_kmc_next()
    pass = true

    list_rates = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    sum_rates = sum(list_rates)
    weights = ProbabilityWeights(list_rates)

    pass &= check(abs(sum_rates - sum(weights)) < 1e-10,
                  @sprintf("... check sum(weights)[%f] == sum_rates[%f]\n",
                           sum(weights), sum_rates)
                 )

    # compare cumulative distribution of times and events to the expected
    # versions for Poisson process
    tbin = 0.5 / sum_rates
    pdf_time(t) = sum_rates * exp(-(t + tbin) * sum_rates)
    cdf_time(t) = 1.0 - exp(-(t + tbin) * sum_rates)
    pdf_event(e) = list_rates[e] / sum_rates
    samples = 10^5
    times = zeros(samples)
    events = zeros(samples)
    alg = KineticMonteCarlo()
    rng = MersenneTwister(1000)
    for i = 1:samples
        time, event = next(alg, rng, weights)
        times[i] = time
        events[i] = event
    end

    t_max = maximum(times)
    hist_time = fit(Histogram, times, 0:tbin:t_max + tbin)
    cdf_time_meas = normalize(float(hist_time), mode = :probability)
    pdf_sum = 0.0
    for i in 1:length(cdf_time_meas.weights)
        pdf_sum += cdf_time_meas.weights[i]
        cdf_time_meas.weights[i] = pdf_sum
    end
    # since measurements only performed until time_max, we have to rescale
    # pdf_time(t) to integrate to 1 in the bounds (-infty, t_max). This is
    # achieved by deviding by cdf(t_max). This then percolates to cdf(t)
    cdf_time_renorm(t) = cdf_time(t) / cdf_time(cdf_time_meas.edges[1][end])
    kld_time = kldivergence(cdf_time_meas, cdf_time_renorm)
    pass &= check(abs(kld_time) < 0.01,
                  @sprintf("... kldivergence cdf_time to cdf_true = %f\n",
                           kld_time)
                 )

    hist_event = fit(Histogram, events, 1:(length(list_rates) + 1))
    pdf_event_meas = normalize(float(hist_event))
    kld_event = kldivergence(pdf_event_meas, pdf_event)
    pass &= check(abs(kld_event) < 0.01,
                  @sprintf("... kldivergence pdf_event to pdf_true = %f\n",
                           kld_event)
                 )

    # check zero rates give infinite time
    pass &= check(next(alg, rng, [0.0])==(Inf,0), "... if no rates left, return (Inf,0)\n")

    return pass
end

# TODO
function test_kmc_event_handler()
    pass = true

    return pass
end
