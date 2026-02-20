using MonteCarloX
using Random
using StatsBase
using StaticArrays
using Test

function test_gillespie_constructors(; verbose=false)
    rng = MersenneTwister(42)
    alg = Gillespie(rng)
    alg_default = Gillespie()

    pass = true
    pass &= alg.rng === rng
    pass &= alg.steps == 0
    pass &= alg.time == 0.0
    pass &= alg_default.rng === Random.GLOBAL_RNG

    if verbose
        println("Gillespie constructor test pass: $(pass)")
    end

    return pass
end

function test_next_time_thinning(; verbose=false)
    rng = MersenneTwister(77)
    rate_generation = 2.0
    rate = t -> 0.5 * rate_generation  # always accepts with probability 0.5

    samples = [next_time(rng, rate, rate_generation) for _ in 1:1_000]
    pass = all(isfinite, samples) && all(>(0), samples)

    if verbose
        println("Next time thinning mean: $(mean(samples))")
    end

    return pass
end

function test_next_event_scalar_rates(; verbose=false)
    rng = MersenneTwister(78)
    pass = next_event(rng, 5.0) == 1

    if verbose
        println("Next event scalar rates pass: $(pass)")
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

function test_advance_time_event_queue_empty(; verbose=false)
    rng = MersenneTwister(79)
    alg = Gillespie(rng)
    q = EventQueue{Int}()

    t_final = advance!(alg, q, 5.0)

    pass = t_final == Inf && alg.time == 0.0 && alg.steps == 0

    if verbose
        println("Advance empty time-queue pass: $(pass)")
    end

    return pass
end

function test_advance_time_event_queue_progress(; verbose=false)
    rng = MersenneTwister(80)
    alg = Gillespie(rng)
    q = EventQueue{Int}()
    add!(q, (1.0, 1))
    add!(q, (2.5, 2))

    seen = Int[]
    t_final = advance!(alg, q, 3.0; update! = (handler, event, t) -> push!(seen, event))

    pass = t_final == Inf && isapprox(alg.time, 2.5; atol = 1e-12) && alg.steps == 2 && seen == [1, 2]

    if verbose
        println("Advance populated time-queue pass: $(pass)")
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

function test_kmc_next_time_special_cases(; verbose=false)
    rng = MersenneTwister(314)

    pass = true
    pass &= next_time(rng, t -> 1.0, 0.0) == Inf
    pass &= next_time(rng, t -> 1.0, -1.0) == Inf

    if verbose
        println("KMC next_time special cases test pass: $(pass)")
    end

    return pass
end

function test_kmc_next_event_special_cases(; verbose=false)
    pass = true

    rng_num = MersenneTwister(9)
    pass &= next_event(rng_num, 0.7) == 1

    rates = MVector{2, Float64}(0.3, 0.7)
    rng_a = MersenneTwister(19)
    rng_b = MersenneTwister(19)
    draw_ref = rand(rng_b) * (rates[1] + rates[2]) < rates[1] ? 1 : 2
    draw_mvec = next_event(rng_a, rates)
    pass &= draw_mvec == draw_ref
    pass &= draw_mvec in (1, 2)

    if verbose
        println("KMC next_event special cases test pass: $(pass)")
    end

    return pass
end

function test_kmc_next_single_rate_inf_paths(; verbose=false)
    alg = Gillespie(MersenneTwister(5050))

    dt_num, ev_num = next(alg, 0.0)
    dt_vec, ev_vec = next(alg, [0.0])
    dt_w, ev_w = next(alg, ProbabilityWeights([0.0]))

    handler = ListEventRateSimple{Int}([1], [0.0], 0.0, -1)
    dt_handler, ev_handler = next(alg, handler)

    pass = true
    pass &= dt_num == Inf && ev_num === nothing
    pass &= dt_vec == Inf && ev_vec === nothing
    pass &= dt_w == Inf && ev_w === nothing
    pass &= dt_handler == Inf && ev_handler === nothing

    if verbose
        println("KMC next single-rate Inf paths pass: $(pass)")
    end

    return pass
end

function test_kmc_event_handler_edge_cases(; verbose=false)
    pass = true

    rng = MersenneTwister(77)

    handler_many = ListEventRateActiveMask{Int}([10, 20, 30], [0.2, 0.3, 0.5], 0.0, -1)
    event_many = next_event(rng, handler_many)
    pass &= event_many in (1, 2, 3)

    handler_one = ListEventRateActiveMask{Int}([10, 20, 30], [0.0, 0.0, 0.0], 0.0, -1; initial = "all_inactive")
    handler_one[2] = 0.4
    pass &= next_event(rng, handler_one) == 2

    handler_none = ListEventRateActiveMask{Int}([10, 20, 30], [0.0, 0.0, 0.0], 0.0, -1; initial = "all_inactive")
    pass &= next_event(rng, handler_none) == -1

    simple = ListEventRateSimple{Int}([10, 20], [1.0, 0.0], 0.0, -1)
    Random.seed!(2026)
    pass &= next_event(simple) == 10
    pass &= next_event(MersenneTwister(2026), simple) == 10

    if verbose
        println("KMC event-handler edge cases test pass: $(pass)")
    end

    return pass
end

function test_kmc_step_and_advance_zero_rate_edges(; verbose=false)
    pass = true

    alg_step = Gillespie(MersenneTwister(2027))
    t_step, event_step = step!(alg_step, [0.0])
    pass &= t_step == Inf
    pass &= event_step === nothing
    pass &= alg_step.time == Inf
    pass &= alg_step.steps == 1

    alg_step_cb = Gillespie(MersenneTwister(2028))
    t_step_cb, event_step_cb = step!(alg_step_cb, t -> [0.0])
    pass &= t_step_cb == Inf
    pass &= event_step_cb === nothing
    pass &= alg_step_cb.time == Inf
    pass &= alg_step_cb.steps == 1

    alg_adv = Gillespie(MersenneTwister(2029))
    seen_events = Any[]
    seen_times = Float64[]
    t_adv = advance!(alg_adv, [0.0], 10.0; update! = (r, e, t) -> begin
        push!(seen_events, e)
        push!(seen_times, t)
    end)

    pass &= t_adv == Inf
    pass &= alg_adv.time == Inf
    pass &= alg_adv.steps == 1
    pass &= isempty(seen_events)
    pass &= isempty(seen_times)

    if verbose
        println("KMC step/advance zero-rate edge cases test pass: $(pass)")
    end

    return pass
end

function test_kmc_two_rate_vector_and_weights_paths(; verbose=false)
    pass = true

    rates_vec = [0.2, 0.8]
    rates_w = ProbabilityWeights(copy(rates_vec))

    alg_vec = Gillespie(MersenneTwister(3030))
    alg_w = Gillespie(MersenneTwister(3030))

    dt_vec, ev_vec = next(alg_vec, rates_vec)
    dt_w, ev_w = next(alg_w, rates_w)

    pass &= dt_vec == dt_w
    pass &= ev_vec == ev_w
    pass &= ev_vec in (1, 2)

    t_vec, ev_step_vec = step!(alg_vec, rates_vec)
    t_w, ev_step_w = step!(alg_w, rates_w)

    pass &= t_vec == t_w
    pass &= ev_step_vec == ev_step_w
    pass &= ev_step_vec in (1, 2)
    pass &= alg_vec.steps == 1
    pass &= alg_w.steps == 1

    alg_w_zero = Gillespie(MersenneTwister(3031))
    rates_w_zero = ProbabilityWeights([0.0, 0.0])
    seen = Any[]
    t_adv = advance!(alg_w_zero, rates_w_zero, 10.0; update! = (r, e, t) -> push!(seen, e))
    pass &= t_adv == Inf
    pass &= alg_w_zero.time == Inf
    pass &= alg_w_zero.steps == 1
    pass &= isempty(seen)

    if verbose
        println("KMC two-rate vector/weights paths test pass: $(pass)")
    end

    return pass
end

function test_kmc_time_event_handler_paths(; verbose=false)
    pass = true

    queue = EventQueue{Int}(0.0)
    add!(queue, (0.3, 11))
    add!(queue, (0.9, 22))

    alg = Gillespie(MersenneTwister(4040))
    t1, e1 = step!(alg, queue)
    pass &= isapprox(t1, 0.3; atol = 1e-12)
    pass &= e1 == 11
    pass &= isapprox(alg.time, 0.3; atol = 1e-12)
    pass &= alg.steps == 1

    t2, e2 = step!(alg, queue)
    pass &= isapprox(t2, 0.9; atol = 1e-12)
    pass &= e2 == 22
    pass &= isapprox(alg.time, 0.9; atol = 1e-12)
    pass &= alg.steps == 2

    t3, e3 = step!(alg, queue)
    pass &= t3 == Inf
    pass &= e3 === nothing
    pass &= isapprox(alg.time, 0.9; atol = 1e-12)
    pass &= alg.steps == 2

    queue2 = EventQueue{Int}(0.0)
    add!(queue2, (1.2, 7))
    alg2 = Gillespie(MersenneTwister(4041))
    observed = Tuple{Int, Float64}[]

    t_final = advance!(alg2, queue2, 10.0; t0 = 1.0, update! = (q, e, t) -> push!(observed, (e, t)))
    pass &= t_final == Inf
    pass &= alg2.steps == 1
    pass &= length(observed) == 1
    pass &= observed[1][1] == 7
    pass &= isapprox(observed[1][2], 1.2; atol = 1e-12)
    pass &= isapprox(get_time(queue2), 1.2; atol = 1e-12)

    if verbose
        println("KMC time-event-handler paths test pass: $(pass)")
    end

    return pass
end


function run_kinetic_monte_carlo_testsets(; verbose=false)
    @testset "Kinetic Monte Carlo" begin
        @testset "Gillespie constructors" begin
            @test test_gillespie_constructors(verbose=verbose)
        end
        @testset "Core helpers" begin
            @test test_next_time_thinning(verbose=verbose)
            @test test_next_event_scalar_rates(verbose=verbose)
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
        @testset "next_time special cases" begin
            @test test_kmc_next_time_special_cases(verbose=verbose)
        end
        @testset "next_event special cases" begin
            @test test_kmc_next_event_special_cases(verbose=verbose)
        end
        @testset "next single-rate Inf paths" begin
            @test test_kmc_next_single_rate_inf_paths(verbose=verbose)
        end
        @testset "event-handler edge cases" begin
            @test test_kmc_event_handler_edge_cases(verbose=verbose)
        end
        @testset "step!/advance! zero-rate edges" begin
            @test test_kmc_step_and_advance_zero_rate_edges(verbose=verbose)
        end
        @testset "two-rate vector/weights paths" begin
            @test test_kmc_two_rate_vector_and_weights_paths(verbose=verbose)
        end
        @testset "time-event-handler paths" begin
            @test test_kmc_time_event_handler_paths(verbose=verbose)
        end
        @testset "time-event advance" begin
            @test test_advance_time_event_queue_empty(verbose=verbose)
            @test test_advance_time_event_queue_progress(verbose=verbose)
        end
    end
    return true
end
