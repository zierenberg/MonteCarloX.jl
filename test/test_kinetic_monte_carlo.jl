# tests for kinetic_monte_carlo.jl
using MonteCarloX
using StatsBase
using Random
using LinearAlgebra

function test_kmc_next(;verbose = false)
    pass = true

    list_rates = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    sum_rates = sum(list_rates)
    weights = ProbabilityWeights(list_rates)
    tbin = 0.5 / sum_rates
    pdf_time(t) = sum_rates * exp(-(t + tbin) * sum_rates)
    cdf_time(t) = 1.0 - exp(-(t + tbin) * sum_rates)
    pdf_event(e) = list_rates[e] / sum_rates

    if verbose
        println("... check sum(weights)[$(sum(weights))] == sum_rates [$(sum_rates)]")
    end
    pass &= abs(sum_rates - sum(weights)) < 1e-10

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
    pass &= abs(kld_time) < 0.01
    if verbose
        println("... kldivergence cdf_time to cdf_true = $(kld_time)")
    end

    hist_event = fit(Histogram, events, 1:(length(list_rates) + 1))
    pdf_event_meas = normalize(float(hist_event))
    kld_event = kldivergence(pdf_event_meas, pdf_event)
    if verbose
        println("... kldivergence pdf_event to pdf_true = $(kld_event)")
    end
    pass &= abs(kld_event) < 0.01

    return pass
end

# TODO
function test_kmc_event_handler(;verbose = false)
    pass = true

    return pass
end
