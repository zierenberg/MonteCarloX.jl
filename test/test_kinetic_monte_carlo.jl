using MonteCarloX
using Random
using StatsBase
using Test

function test_gillespie_constructors(; verbose=false)
    rng = MersenneTwister(42)
    alg = Gillespie(rng)
    alg_default = Gillespie()

    pass = true
    pass &= alg.rng === rng
    pass &= alg.steps == 0
    pass &= alg.time == 0.0
    pass &= alg_default.rng isa AbstractRNG

    if verbose
        println("Gillespie constructor test pass: $(pass)")
    end

    return pass
end

function test_kmc_next_distribution(; verbose=false)
    rng = MersenneTwister(1234)
    alg = Gillespie(rng)

    rates = [0.1, 0.2, 0.7]
    total_rate = sum(rates)
    n_samples = 50_000

    times = Vector{Float64}(undef, n_samples)
    counts = zeros(Int, length(rates))

    for i in 1:n_samples
        dt, event = next(alg, rates)
        times[i] = dt
        counts[event] += 1
    end

    mean_dt = mean(times)
    event_probs = counts ./ n_samples

    pass = true
    pass &= abs(mean_dt - 1 / total_rate) < 0.03
    pass &= abs(event_probs[1] - rates[1] / total_rate) < 0.02
    pass &= abs(event_probs[2] - rates[2] / total_rate) < 0.02
    pass &= abs(event_probs[3] - rates[3] / total_rate) < 0.02

    if verbose
        println("KMC next distribution test pass: $(pass)")
    end

    return pass
end

function test_kmc_next_event_handler_consistency(; verbose=false)
    rates = [0.1, 0.2, 0.3, 0.4]
    weights = ProbabilityWeights(rates)
    simple = ListEventRateSimple{Int}(collect(1:length(rates)), rates, 0.0, 0)
    mask = ListEventRateActiveMask{Int}(collect(1:length(rates)), rates, 0.0, 0)

    alg_vec = Gillespie(MersenneTwister(2024))
    alg_weights = Gillespie(MersenneTwister(2024))
    alg_simple = Gillespie(MersenneTwister(2024))
    alg_mask = Gillespie(MersenneTwister(2024))

    pass = true
    for _ in 1:100
        dt_vec, ev_vec = next(alg_vec, rates)
        dt_w, ev_w = next(alg_weights, weights)
        dt_simple, ev_simple = next(alg_simple, simple)
        dt_mask, ev_mask = next(alg_mask, mask)

        pass &= dt_vec == dt_w == dt_simple == dt_mask
        pass &= ev_vec == ev_w == ev_simple == ev_mask
    end

    if verbose
        println("KMC event-handler consistency test pass: $(pass)")
    end

    return pass
end

function test_kmc_single_rate_matches_randexp(; verbose=false)
    seed = 2026
    rate = 1.2
    n_samples = 1_000

    alg = Gillespie(MersenneTwister(seed))
    rng_naive = MersenneTwister(seed)

    pass = true
    for _ in 1:n_samples
        dt, event = next(alg, [rate])
        dt_ref = randexp(rng_naive) / rate
        pass &= dt == dt_ref
        pass &= event == 1
    end

    if verbose
        println("KMC single-rate matches randexp test pass: $(pass)")
    end

    return pass
end

function test_kmc_advance_and_statistics(; verbose=false)
    rates = [0.1, 0.2, 0.3]
    alg = Gillespie(MersenneTwister(11))

    callback_count = Ref(0)
    function update_rates!(r, event, t)
        callback_count[] += 1
        return nothing
    end

    t_final = advance!(alg, rates, 10.0; update! = update_rates!)

    pass = true
    pass &= t_final > 10.0
    pass &= callback_count[] == alg.steps
    pass &= alg.time == t_final

    reset_statistics!(alg)
    pass &= alg.steps == 0
    pass &= alg.time == 0.0

    dt, event = next(alg, [0.0])
    pass &= dt == Inf
    pass &= event === nothing

    if verbose
        println("KMC advance/statistics test pass: $(pass)")
    end

    return pass
end

function test_kmc_advance_with_explicit_rates_callback(; verbose=false)
    sys = Dict(:N => 10)
    local_rates(s, t) = [0.42, 0.40]

    alg = Gillespie(MersenneTwister(99))
    measured_N = Int[]
    updated_N_before = Int[]

    measure_cb = (state, t, event) -> push!(measured_N, state[:N])
    function update_cb(state, event, t)
        push!(updated_N_before, state[:N])
        if event == 1
            state[:N] += 1
        elseif event == 2
            state[:N] = max(0, state[:N] - 1)
        end
        return nothing
    end

    t_final = advance!(
        alg,
        sys,
        30.0;
        rates = local_rates,
        measure! = measure_cb,
        update! = update_cb,
    )

    pass = true
    pass &= t_final > 30.0
    pass &= length(measured_N) == length(updated_N_before)
    pass &= measured_N == updated_N_before
    pass &= alg.steps == length(measured_N)
    pass &= alg.time == t_final

    if verbose
        println("KMC callback-based advance ordering test pass: $(pass)")
    end

    return pass
end


function test_step_helper_rates_callback(; verbose=false)
    alg = Gillespie(MersenneTwister(12))
    local_rates(t) = [0.42, 0.40]

    t_new, event = step!(alg, local_rates)

    pass = true
    pass &= t_new > 0.0
    pass &= event in (1, 2)
    pass &= alg.steps == 1
    pass &= alg.time == t_new

    if verbose
        println("step! rates callback test pass: $(pass)")
    end

    return pass
end


function run_kinetic_monte_carlo_testsets(; verbose=false)
    @testset "Kinetic Monte Carlo" begin
        @testset "Gillespie constructors" begin
            @test test_gillespie_constructors(verbose=verbose)
        end
        @testset "next distribution" begin
            @test test_kmc_next_distribution(verbose=verbose)
        end
        @testset "event handler consistency" begin
            @test test_kmc_next_event_handler_consistency(verbose=verbose)
        end
        @testset "single-rate matches randexp" begin
            @test test_kmc_single_rate_matches_randexp(verbose=verbose)
        end
        @testset "advance and statistics" begin
            @test test_kmc_advance_and_statistics(verbose=verbose)
        end
        @testset "advance with rate callback" begin
            @test test_kmc_advance_with_explicit_rates_callback(verbose=verbose)
        end
        @testset "step! rates callback" begin
            @test test_step_helper_rates_callback(verbose=verbose)
        end
    end
    return true
end
