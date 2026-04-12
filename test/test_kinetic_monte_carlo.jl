using MonteCarloX
using Random
using StatsBase
using StaticArrays
using Test

mutable struct ExplicitSourceTestSystem
    N::Int
end

MonteCarloX.event_source(sys::ExplicitSourceTestSystem) = [0.42, 0.40]

function test_gillespie_constructors()
    rng = MersenneTwister(42)
    alg = Gillespie(rng)
    alg_default = Gillespie()

    pass = true
    pass &= check(alg.rng === rng, "rng stored\n")
    pass &= check(alg.steps == 0, "steps == 0\n")
    pass &= check(alg.time == 0.0, "time == 0.0\n")
    pass &= check(alg_default.rng === Random.GLOBAL_RNG, "default RNG\n")

    return pass
end

function test_next_time_thinning()
    rng = MersenneTwister(77)
    rate_generation = 2.0
    effective_rate = 0.5 * rate_generation
    rate = t -> effective_rate

    n_samples = 50_000
    samples = [next_time(rng, rate, rate_generation) for _ in 1:n_samples]
    mean_dt = sum(samples) / n_samples

    pass = check(abs(mean_dt - 1 / effective_rate) < 0.03, "thinning mean dt matches effective rate\n")

    return pass
end

function test_next_event_scalar_rates()
    rng = MersenneTwister(78)
    pass = check(next_event(rng, 5.0) == 1, "scalar rate returns 1\n")

    return pass
end

function test_kmc_next_distribution()
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
    pass &= check(abs(mean_dt - 1 / total_rate) < 0.03, "mean dt correct\n")
    for k in 1:3
        pass &= check(abs(event_probs[k] - rates[k] / total_rate) < 0.02, "event prob $k correct\n")
    end

    return pass
end

function test_kmc_next_event_handler_consistency()
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
    pass = check(pass, "all event handler types produce identical sequences\n")

    return pass
end

function test_kmc_single_rate()
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
    pass = check(pass, "single-rate matches naive randexp/rate\n")

    return pass
end

function test_kmc_advance_and_statistics()
    rates = [0.1, 0.2, 0.3]
    alg = Gillespie(MersenneTwister(11))

    callback_count = Ref(0)
    function modify_rates!(r, event, t)
        callback_count[] += 1
        return nothing
    end

    t_final = advance!(alg, rates, 10.0; modify! = modify_rates!)

    pass = true
    pass &= check(t_final > 10.0, "t_final > 10.0\n")
    pass &= check(callback_count[] == alg.steps, "callback count == steps\n")
    pass &= check(alg.time == t_final, "alg.time == t_final\n")

    reset!(alg)
    pass &= check(alg.steps == 0, "steps reset\n")
    pass &= check(alg.time == 0.0, "time reset\n")

    dt, event = next(alg, [0.0])
    pass &= check(dt == Inf, "zero rate dt == Inf\n")
    pass &= check(event === nothing, "zero rate event == nothing\n")

    return pass
end

function test_advance_time_event_queue_empty()
    rng = MersenneTwister(79)
    alg = Gillespie(rng)
    q = EventQueue{Int}()

    t_final = advance!(alg, q, 5.0)

    pass = true
    pass &= check(t_final == Inf, "empty queue t_final == Inf\n")
    pass &= check(alg.time == Inf, "empty queue alg.time == Inf\n")
    pass &= check(alg.steps == 1, "empty queue steps == 1\n")

    return pass
end

function test_advance_time_event_queue_progress()
    rng = MersenneTwister(80)
    alg = Gillespie(rng)
    q = EventQueue{Int}()
    add!(q, (1.0, 1))
    add!(q, (2.5, 2))

    seen = Int[]
    t_final = advance!(alg, q, 3.0; measure! = (handler, event, t) -> push!(seen, event))

    pass = true
    pass &= check(t_final == Inf, "queue t_final == Inf\n")
    pass &= check(alg.time == Inf, "queue alg.time == Inf\n")
    pass &= check(alg.steps == 3, "queue steps == 3\n")
    pass &= check(seen == [1, 2], "events seen in order\n")

    return pass
end

function test_kmc_advance_with_explicit_rates_callback()
    sys = ExplicitSourceTestSystem(10)

    alg = Gillespie(MersenneTwister(99))
    measured_N = Int[]
    modified_N_before = Int[]

    measure_cb = (state, event, t) -> push!(measured_N, state.N)
    function modify_cb(state, event, t)
        push!(modified_N_before, state.N)
        if event == 1
            state.N += 1
        elseif event == 2
            state.N = max(0, state.N - 1)
        end
        return nothing
    end

    t_final = advance!(
        alg,
        sys,
        30.0;
        measure! = measure_cb,
        modify! = modify_cb,
    )

    pass = true
    pass &= check(t_final > 30.0, "t_final > 30.0\n")
    pass &= check(length(measured_N) == length(modified_N_before), "measure and modify counts match\n")
    pass &= check(measured_N == modified_N_before, "measure before modify ordering\n")
    pass &= check(alg.steps == length(measured_N), "steps == measure count\n")
    pass &= check(alg.time == t_final, "alg.time == t_final\n")

    return pass
end

function test_step_helper_rates_callback()
    alg = Gillespie(MersenneTwister(12))
    local_rates(t) = [0.42, 0.40]

    t_new, event = step!(alg, local_rates)

    pass = true
    pass &= check(t_new > 0.0, "t_new > 0.0\n")
    pass &= check(event in (1, 2), "event in (1, 2)\n")
    pass &= check(alg.steps == 1, "steps == 1\n")
    pass &= check(alg.time == t_new, "alg.time == t_new\n")

    return pass
end

function test_kmc_next_time_special_cases()
    rng = MersenneTwister(314)

    pass = true
    pass &= check(next_time(rng, t -> 1.0, 0.0) == Inf, "zero generation rate -> Inf\n")
    pass &= check(next_time(rng, t -> 1.0, -1.0) == Inf, "negative generation rate -> Inf\n")

    return pass
end

function test_kmc_next_event_special_cases()
    pass = true

    # MVector path
    rates = MVector{2, Float64}(0.3, 0.7)
    rng_a = MersenneTwister(19)
    rng_b = MersenneTwister(19)
    draw_ref = rand(rng_b) * (rates[1] + rates[2]) < rates[1] ? 1 : 2
    draw_mvec = next_event(rng_a, rates)
    pass &= check(draw_mvec == draw_ref, "MVector matches manual draw\n")

    pass &= check(next_event(MersenneTwister(20), MVector{2, Float64}(1.0, 0.0)) == 1, "MVector deterministic event 1\n")
    pass &= check(next_event(MersenneTwister(21), MVector{2, Float64}(0.0, 1.0)) == 2, "MVector deterministic event 2\n")

    # empty handler
    simple_empty = ListEventRateSimple{Int}(Int[], Float64[], 0.0, -1)
    pass &= check(next_event(MersenneTwister(23), simple_empty) == -1, "empty handler returns default\n")
    pass &= check(next_event(simple_empty) == -1, "empty handler no-rng returns default\n")

    return pass
end

function test_kmc_event_handler_edge_cases()
    pass = true

    rng = MersenneTwister(77)

    handler_many = ListEventRateActiveMask{Int}([10, 20, 30], [0.2, 0.3, 0.5], 0.0, -1)
    event_many = next_event(rng, handler_many)
    pass &= check(event_many in (1, 2, 3), "active mask selects valid event\n")

    handler_one = ListEventRateActiveMask{Int}([10, 20, 30], [0.0, 0.0, 0.0], 0.0, -1; initial = "all_inactive")
    handler_one[2] = 0.4
    pass &= check(next_event(rng, handler_one) == 2, "single active event selected\n")

    handler_none = ListEventRateActiveMask{Int}([10, 20, 30], [0.0, 0.0, 0.0], 0.0, -1; initial = "all_inactive")
    pass &= check(next_event(rng, handler_none) == -1, "no active events returns default\n")

    simple = ListEventRateSimple{Int}([10, 20], [1.0, 0.0], 0.0, -1)
    Random.seed!(2026)
    pass &= check(next_event(simple) == 10, "simple no-rng selects active\n")
    pass &= check(next_event(MersenneTwister(2026), simple) == 10, "simple with-rng selects active\n")

    return pass
end

function test_kmc_zero_rate_advance()
    alg = Gillespie(MersenneTwister(2029))
    seen = Any[]
    t_adv = advance!(alg, [0.0], 10.0; measure! = (r, e, t) -> push!(seen, e))

    pass = true
    pass &= check(t_adv == Inf, "advance! zero rate t == Inf\n")
    pass &= check(alg.steps == 1, "advance! zero rate steps == 1\n")
    pass &= check(isempty(seen), "advance! zero rate no events measured\n")

    return pass
end

function test_kmc_time_event_handler_paths()
    pass = true

    queue = EventQueue{Int}(0.0)
    add!(queue, (0.3, 11))
    add!(queue, (0.9, 22))

    alg = Gillespie(MersenneTwister(4040))
    t1, e1 = step!(alg, queue)
    pass &= check(isapprox(t1, 0.3; atol = 1e-12), "first step t == 0.3\n")
    pass &= check(e1 == 11, "first step event == 11\n")
    pass &= check(isapprox(alg.time, 0.3; atol = 1e-12), "first step alg.time\n")
    pass &= check(alg.steps == 1, "first step steps == 1\n")

    t2, e2 = step!(alg, queue)
    pass &= check(isapprox(t2, 0.9; atol = 1e-12), "second step t == 0.9\n")
    pass &= check(e2 == 22, "second step event == 22\n")
    pass &= check(isapprox(alg.time, 0.9; atol = 1e-12), "second step alg.time\n")
    pass &= check(alg.steps == 2, "second step steps == 2\n")

    t3, e3 = step!(alg, queue)
    pass &= check(t3 == Inf, "third step (empty) t == Inf\n")
    pass &= check(e3 === nothing, "third step (empty) event == nothing\n")
    pass &= check(isapprox(alg.time, Inf; atol = 1e-12), "third step alg.time == Inf\n")
    pass &= check(alg.steps == 3, "third step steps == 3\n")

    queue2 = EventQueue{Int}(0.0)
    add!(queue2, (1.2, 7))
    alg2 = Gillespie(MersenneTwister(4041))
    observed = Tuple{Int, Float64}[]

    t_final = advance!(alg2, queue2, 10.0; t0 = 1.0, measure! = (q, e, t) -> push!(observed, (e, t)))
    pass &= check(t_final == Inf, "advance queue t_final == Inf\n")
    pass &= check(alg2.steps == 2, "advance queue steps == 2\n")
    pass &= check(length(observed) == 1, "advance queue one event observed\n")
    pass &= check(observed[1][1] == 7, "advance queue event == 7\n")
    pass &= check(isapprox(observed[1][2], 1.2; atol = 1e-12), "advance queue event time\n")
    pass &= check(isapprox(get_time(queue2), 1.2; atol = 1e-12), "advance queue final time\n")

    return pass
end

@testset "Kinetic Monte Carlo" begin
    @testset "Gillespie constructors" begin
        @test test_gillespie_constructors()
    end
    @testset "Core helpers" begin
        @test test_next_time_thinning()
        @test test_next_event_scalar_rates()
    end
    @testset "next distribution" begin
        @test test_kmc_next_distribution()
    end
    @testset "event handler consistency" begin
        @test test_kmc_next_event_handler_consistency()
    end
    @testset "single-rate" begin
        @test test_kmc_single_rate()
    end
    @testset "advance and statistics" begin
        @test test_kmc_advance_and_statistics()
    end
    @testset "advance with rate callback" begin
        @test test_kmc_advance_with_explicit_rates_callback()
    end
    @testset "step! rates callback" begin
        @test test_step_helper_rates_callback()
    end
    @testset "next_time special cases" begin
        @test test_kmc_next_time_special_cases()
    end
    @testset "next_event special cases" begin
        @test test_kmc_next_event_special_cases()
    end
    @testset "event-handler edge cases" begin
        @test test_kmc_event_handler_edge_cases()
    end
    @testset "zero-rate advance" begin
        @test test_kmc_zero_rate_advance()
    end
    @testset "time-event-handler paths" begin
        @test test_kmc_time_event_handler_paths()
    end
    @testset "time-event advance" begin
        @test test_advance_time_event_queue_empty()
        @test test_advance_time_event_queue_progress()
    end
end
