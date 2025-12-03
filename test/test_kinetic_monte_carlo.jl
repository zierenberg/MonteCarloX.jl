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
    sim = init(MersenneTwister(1000), KineticMonteCarlo(), weights)
    for i = 1:samples
        time, event = next(sim)
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
    sim = init(MersenneTwister(1000), KineticMonteCarlo(), [0.0])
    pass &= check(next(sim)==(Inf,0), "... if no rates left, return (Inf,0)\n")

    ###############################################################################
    ###### event handler
    list_rates = [0.1,0.2,0.3]
    weights = ProbabilityWeights(list_rates)
    event_handler_rate_simple = ListEventRateSimple{Int}(collect(1:length(list_rates)), list_rates, 0.0, 0)
    event_handler_rate_mask = ListEventRateActiveMask{Int}(collect(1:length(list_rates)), list_rates, 0.0, 0)

    samples = 10
    sim_base = init(MersenneTwister(1000), KineticMonteCarlo(), weights)
    sim_simple = init(MersenneTwister(1000), KineticMonteCarlo(), event_handler_rate_simple)
    sim_mask = init(MersenneTwister(1000), KineticMonteCarlo(), event_handler_rate_mask)
    for i = 1:samples
        t_base, e_base = next(sim_base)
        t_rate_simple, e_rate_simple = next(sim_simple)
        t_rate_mask, e_rate_mask = next(sim_mask)
        pass &= check(t_base==t_rate_simple, @sprintf("... %d, base == simple (time) \n", i))
        pass &= check(e_base==e_rate_simple, @sprintf("... %d, base == simple (event) \n", i))
        pass &= check(t_base==t_rate_mask, @sprintf("... %d, base == mask (time) \n", i))
        pass &= check(e_base==e_rate_mask, @sprintf("... %d, base == mask (event) \n", i))
    end

    return pass
end

function test_kmc_advance()
    pass = true

    list_rates = [0.1,0.2,0.3]
    weights = ProbabilityWeights(list_rates)
    event_handler_rate_simple = ListEventRateSimple{Int}(collect(1:length(list_rates)), list_rates, 0.0, 0)
    event_handler_rate_mask = ListEventRateActiveMask{Int}(collect(1:length(list_rates)), list_rates, 0.0, 0)

    sim_base = init(MersenneTwister(1000), KineticMonteCarlo(), weights)
    sim_simple = init(MersenneTwister(1000), KineticMonteCarlo(), event_handler_rate_simple)
    sim_mask = init(MersenneTwister(1000), KineticMonteCarlo(), event_handler_rate_mask)

    T = 10.
    update!(sim,event) = missing
    time_base = advance!(sim_base, T, update! = update!)
    pass &= check(time_base > T, @sprintf("... final time base > target \n"))
    time_simple = advance!(sim_simple, T, update! = update!)
    pass &= check(time_simple > T, @sprintf("... final time simple > target \n"))
    pass &= check(time_simple == time_base, @sprintf("... final time simple == base  \n"))
    time_mask = advance!(sim_mask, T, update! = update!)
    pass &= check(time_mask > T, @sprintf("... final time mask > target \n"))
    pass &= check(time_mask == time_base, @sprintf("... final time mask == base  \n"))

    return pass
end

