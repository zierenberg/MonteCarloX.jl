using MonteCarloX
using Test
using StatsBase
using Random

struct UnsupportedContainer end

function test_preallocated_schedule_constructor()
    times = [3.0, 1.0, 2.0]
    schedule = PreallocatedSchedule(times)

    pass = true
    pass &= check(schedule.times == [1.0, 2.0, 3.0], "times sorted\n")
    pass &= check(schedule.checkpoint_idx == 1, "checkpoint_idx == 1\n")

    return pass
end

function test_measurement_pair_constructor()
    m = Measurement((s -> s^2) => Float64[])

    pass = true
    pass &= check(m isa Measurement, "isa Measurement\n")
    measure!(m, 3.0)
    pass &= check(m.data == [9.0], "measure! applies observable\n")

    return pass
end

function test_measurements_interval_constructors()
    dict_measurements = Dict{Symbol, Measurement}(
        :x => Measurement((s -> s), Float64[]),
        :y => Measurement((s -> 2s), Float64[]),
    )
    m_dict = Measurements(dict_measurements; interval = 0.5)

    pairs = [
        :x => ((s -> s) => Float64[]),
        :y => ((s -> 2s) => Float64[]),
    ]
    m_pairs = Measurements(pairs; interval = 0.25)

    pass = true
    pass &= check(m_dict.schedule isa IntervalSchedule, "dict: IntervalSchedule\n")
    pass &= check(m_dict.schedule.interval == 0.5, "dict: interval == 0.5\n")
    pass &= check(m_dict.schedule._checkpoint == 0.0, "dict: checkpoint == 0.0\n")

    pass &= check(m_pairs.schedule isa IntervalSchedule, "pairs: IntervalSchedule\n")
    pass &= check(m_pairs.schedule.interval == 0.25, "pairs: interval == 0.25\n")
    pass &= check(haskey(m_pairs.measurements, :x), "pairs: has :x\n")
    pass &= check(haskey(m_pairs.measurements, :y), "pairs: has :y\n")

    return pass
end

function test_measurements_dict_preallocated_constructor()
    measurements = Dict{Symbol, Measurement}(
        :x => Measurement((s -> s), Float64[]),
        :y => Measurement((s -> 2s), Float64[]),
    )
    times = [3, 1, 2]

    m = Measurements(measurements, times)

    pass = true
    pass &= check(m.schedule isa PreallocatedSchedule, "PreallocatedSchedule\n")
    pass &= check(m.schedule.times == [1.0, 2.0, 3.0], "times sorted\n")
    pass &= check(m.schedule.checkpoint_idx == 1, "checkpoint_idx == 1\n")

    return pass
end

function test_measurements_pairs_preallocated_constructor()
    pairs = [
        :x => ((s -> s) => Float64[]),
        :y => ((s -> 2s) => Float64[]),
    ]
    times = [2, 1, 3]

    m = Measurements(pairs, times)

    pass = true
    pass &= check(m.schedule isa PreallocatedSchedule, "PreallocatedSchedule\n")
    pass &= check(m.schedule.times == [1.0, 2.0, 3.0], "times sorted\n")
    pass &= check(haskey(m.measurements, :x), "has :x\n")
    pass &= check(haskey(m.measurements, :y), "has :y\n")
    pass &= check(m[:x] isa Measurement, "m[:x] isa Measurement\n")
    pass &= check(m[:y] isa Measurement, "m[:y] isa Measurement\n")

    return pass
end

function test_measurements_setindex()
    m = Measurements(
        Dict{Symbol, Measurement}(:x => Measurement((s -> s), Float64[])),
        [1.0, 2.0]
    )

    replacement = Measurement((s -> 3s), Float64[])
    m[:x] = replacement

    pass = true
    pass &= check(m[:x] === replacement, "setindex! replaces measurement\n")
    pass &= check(m.measurements[:x] === replacement, "internal dict updated\n")

    return pass
end

function test_measurements_accessors()
    m = Measurements(
        [:x => ((s -> s) => Float64[])],
        [1.0, 2.0, 3.0],
    )

    measure!(m, 7.0, 1.1)
    measure!(m, 9.0, 3.2)

    pass = true
    pass &= check(times(m) == [1.0, 2.0, 3.0], "times accessor\n")
    pass &= check(data(m, :x) == [7.0, 9.0, 9.0], "data accessor\n")

    return pass
end

function test_measurements_interval_cadence()
    m = Measurements(
        [:x => ((s -> s) => Float64[])],
        interval = 1.0,
    )

    pass = true

    measure!(m, 5.0, -0.1)
    pass &= check(isempty(m[:x].data), "below first checkpoint: no measurement\n")
    pass &= check(m.schedule._checkpoint == 0.0, "checkpoint unchanged\n")

    measure!(m, 7.0, 0.0)
    pass &= check(m[:x].data == [7.0], "at checkpoint: one measurement\n")
    pass &= check(m.schedule._checkpoint == 1.0, "checkpoint advanced\n")

    measure!(m, 9.0, 3.5)
    pass &= check(m[:x].data == [7.0, 9.0], "jump: still one measurement per call\n")
    pass &= check(m.schedule._checkpoint == 2.0, "checkpoint advanced after jump\n")

    return pass
end

function test_measurements_preallocated_measure_event_skipping()
    m = Measurements(
        [
            :x => ((s -> s) => Float64[]),
            :y => ((s -> 2s) => Float64[]),
        ],
        [1.0, 2.0, 3.0]
    )

    measure!(m, 7.0, 2.5)

    pass = true
    pass &= check(m.schedule.checkpoint_idx == 3, "skipped to checkpoint 3\n")
    pass &= check(m[:x].data == [7.0, 7.0], "x: two measurements from skip\n")
    pass &= check(m[:y].data == [14.0, 14.0], "y: two measurements from skip\n")

    measure!(m, 9.0, 10.0)
    pass &= check(m.schedule.checkpoint_idx == 4, "final checkpoint processed\n")
    pass &= check(m[:x].data == [7.0, 7.0, 9.0], "x: three measurements\n")
    pass &= check(m[:y].data == [14.0, 14.0, 18.0], "y: three measurements\n")

    measure!(m, 11.0, 20.0)
    pass &= check(m[:x].data == [7.0, 7.0, 9.0], "x: no append after completion\n")
    pass &= check(m[:y].data == [14.0, 14.0, 18.0], "y: no append after completion\n")

    return pass
end

function test_measurements_is_complete()
    pass = true

    m_interval = Measurements(
        [:x => ((s -> s) => Float64[])],
        interval=1.0
    )
    pass &= check(is_complete(m_interval) == false, "interval: not complete initially\n")
    measure!(m_interval, 1.0, 100.0)
    pass &= check(is_complete(m_interval) == false, "interval: never complete\n")

    m_preallocated = Measurements(
        [:x => ((s -> s) => Float64[])],
        [1.0, 2.0]
    )
    pass &= check(is_complete(m_preallocated) == false, "preallocated: not complete initially\n")
    measure!(m_preallocated, 5.0, 1.5)
    pass &= check(is_complete(m_preallocated) == false, "preallocated: not complete mid-way\n")
    measure!(m_preallocated, 6.0, 2.5)
    pass &= check(is_complete(m_preallocated) == true, "preallocated: complete\n")

    return pass
end

function test_measurement_reset_single()
    m_vec = Measurement((s -> s), Float64[])
    measure!(m_vec, 1.0)
    measure!(m_vec, 2.0)
    reset!(m_vec)

    bins = 0.0:1.0:4.0
    hist = fit(Histogram, [0.2, 1.4, 2.6], bins)
    m_hist = Measurement((s -> s), hist)
    reset!(m_hist)

    pass = true
    pass &= check(isempty(m_vec.data), "vector data cleared\n")
    pass &= check(all(iszero, m_hist.data.weights), "histogram weights zeroed\n")

    return pass
end

function test_measurement_reset_special_cases()
    pass = true

    interval_schedule = IntervalSchedule(0.3)
    interval_schedule._checkpoint = 1.2
    preallocated_schedule = PreallocatedSchedule([1.0, 2.0])
    preallocated_schedule.checkpoint_idx = 3

    pass &= check(reset!(interval_schedule) === interval_schedule, "interval reset! returns self\n")
    pass &= check(interval_schedule._checkpoint == 0.0, "interval checkpoint reset\n")

    pass &= check(reset!(preallocated_schedule) === preallocated_schedule, "preallocated reset! returns self\n")
    pass &= check(preallocated_schedule.checkpoint_idx == 1, "preallocated checkpoint_idx reset\n")

    custom_container = [1, 2, 3]
    m_custom = Measurement((s -> s), custom_container)
    emptied = try; reset!(m_custom); isempty(custom_container); catch; false; end
    pass &= check(emptied, "vector container emptied by reset!\n")

    m_bad = Measurement((s -> s), UnsupportedContainer())
    threw = try; reset!(m_bad); false; catch err; err isa ArgumentError; end
    pass &= check(threw, "unsupported container throws ArgumentError\n")

    return pass
end

function test_measurements_reset_container()
    m_interval = Measurements(
        [
            :x => ((s -> s) => Float64[]),
            :h => ((s -> Int(s)) => fit(Histogram, Int[], 0:1:4)),
        ],
        interval=1.0,
    )
    measure!(m_interval, 1.0, 0.0)
    measure!(m_interval, 2.0, 1.0)

    pass = true
    pass &= check(m_interval.schedule._checkpoint > 0.0, "checkpoint advanced\n")
    pass &= check(!isempty(m_interval[:x].data), "data collected\n")

    reset!(m_interval)
    pass &= check(m_interval.schedule._checkpoint == 0.0, "interval checkpoint reset\n")
    pass &= check(isempty(m_interval[:x].data), "x data cleared\n")
    pass &= check(all(iszero, m_interval[:h].data.weights), "histogram weights zeroed\n")

    m_preallocated = Measurements(
        [:x => ((s -> s) => Float64[])],
        [1.0, 2.0],
    )
    measure!(m_preallocated, 7.0, 5.0)
    pass &= check(m_preallocated.schedule.checkpoint_idx == 3, "checkpoint advanced\n")

    reset!(m_preallocated)
    pass &= check(m_preallocated.schedule.checkpoint_idx == 1, "preallocated checkpoint_idx reset\n")
    pass &= check(isempty(m_preallocated[:x].data), "x data cleared\n")

    return pass
end

function test_tau_int_estimator()
    pass = true

    pass &= check(integrated_autocorrelation_time(fill(1.0, 64)) == 0.5, "constant signal tau_int == 0.5\n")
    pass &= check(tau_int(fill(1.0, 64)) == 0.5, "tau_int alias works\n")

    rng = MersenneTwister(123)

    white = randn(rng, 4_000)
    tau_white = integrated_autocorrelation_time(white)
    pass &= check(0.5 <= tau_white <= 2.0, "white noise tau_int near 0.5\n")

    phi = 0.8
    n = 6_000
    ar = Vector{Float64}(undef, n)
    ar[1] = randn(rng)
    noise_scale = sqrt(1 - phi^2)
    @inbounds for t in 2:n
        ar[t] = phi * ar[t - 1] + noise_scale * randn(rng)
    end

    tau_ar = integrated_autocorrelation_time(ar)
    pass &= check(2.5 <= tau_ar <= 7.0, "AR(1) tau_int in expected range\n")

    tau_capped = integrated_autocorrelation_time(ar; max_lag=50)
    pass &= check(isfinite(tau_capped), "capped tau_int is finite\n")
    pass &= check(tau_capped >= 0.5, "capped tau_int >= 0.5\n")

    threw = try; integrated_autocorrelation_time(randn(rng, 20); max_lag=11); false
    catch err; err isa ArgumentError; end
    pass &= check(threw, "excessive max_lag throws ArgumentError\n")

    tau_batch = integrated_autocorrelation_times([randn(rng, 20)]; min_points=2, max_lag=0)
    pass &= check(length(tau_batch) == 1, "batch returns one element\n")
    pass &= check(isnan(tau_batch[1]), "skipped batch entry is NaN\n")

    return pass
end

@testset "Measurements" begin
    @testset "PreallocatedSchedule constructor" begin
        @test test_preallocated_schedule_constructor()
    end
    @testset "Measurement(pair) constructor" begin
        @test test_measurement_pair_constructor()
    end
    @testset "Measurements interval constructors" begin
        @test test_measurements_interval_constructors()
    end
    @testset "Measurements(dict, times)" begin
        @test test_measurements_dict_preallocated_constructor()
    end
    @testset "Measurements(pairs, times)" begin
        @test test_measurements_pairs_preallocated_constructor()
    end
    @testset "Measurements setindex!" begin
        @test test_measurements_setindex()
    end
    @testset "accessor API" begin
        @test test_measurements_accessors()
    end
    @testset "interval cadence" begin
        @test test_measurements_interval_cadence()
    end
    @testset "measure! event skipping" begin
        @test test_measurements_preallocated_measure_event_skipping()
    end
    @testset "is_complete" begin
        @test test_measurements_is_complete()
    end
    @testset "reset! single measurement" begin
        @test test_measurement_reset_single()
    end
    @testset "reset! special cases" begin
        @test test_measurement_reset_special_cases()
    end
    @testset "reset! measurements container" begin
        @test test_measurements_reset_container()
    end
    @testset "tau_int estimator" begin
        @test test_tau_int_estimator()
    end
end
