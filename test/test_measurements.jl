using MonteCarloX
using Test
using StatsBase
using Random

struct UnsupportedContainer end

function test_measurement_constructors()
    pass = true

    # Measurement from pair
    m = Measurement((s -> s^2) => Float64[])
    pass &= check(m isa Measurement, "Measurement from pair\n")
    measure!(m, 3.0)
    pass &= check(m.data == [9.0], "measure! applies observable\n")

    # Measurements from pairs + interval
    m_interval = Measurements(
        [:x => ((s -> s) => Float64[]), :y => ((s -> 2s) => Float64[])];
        interval = 0.5,
    )
    pass &= check(m_interval.schedule isa IntervalSchedule, "IntervalSchedule\n")
    pass &= check(m_interval.schedule.interval == 0.5, "interval == 0.5\n")
    pass &= check(m_interval.schedule._checkpoint == 0.0, "checkpoint == 0.0\n")
    pass &= check(haskey(m_interval.measurements, :x), "has :x\n")
    pass &= check(haskey(m_interval.measurements, :y), "has :y\n")

    # Measurements from pairs + preallocated times (verifies sorting)
    m_pre = Measurements([:x => ((s -> s) => Float64[])], [3, 1, 2])
    pass &= check(m_pre.schedule isa PreallocatedSchedule, "PreallocatedSchedule\n")
    pass &= check(m_pre.schedule.times == [1.0, 2.0, 3.0], "times sorted\n")
    pass &= check(m_pre.schedule.checkpoint_idx == 1, "checkpoint_idx == 1\n")

    return pass
end

function test_measurements_interval_behavior()
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

    pass &= check(is_complete(m) == false, "interval schedule never complete\n")

    return pass
end

function test_measurements_preallocated_behavior()
    m = Measurements(
        [
            :x => ((s -> s) => Float64[]),
            :y => ((s -> 2s) => Float64[]),
        ],
        [1.0, 2.0, 3.0]
    )

    pass = true

    # accessors
    pass &= check(times(m) == [1.0, 2.0, 3.0], "times accessor\n")
    pass &= check(is_complete(m) == false, "not complete initially\n")

    # event skipping: t=2.5 covers checkpoints 1 and 2
    measure!(m, 7.0, 2.5)
    pass &= check(m.schedule.checkpoint_idx == 3, "skipped to checkpoint 3\n")
    pass &= check(m[:x].data == [7.0, 7.0], "x: two measurements from skip\n")
    pass &= check(m[:y].data == [14.0, 14.0], "y: two measurements from skip\n")
    pass &= check(data(m, :x) == [7.0, 7.0], "data accessor\n")

    # final checkpoint
    measure!(m, 9.0, 10.0)
    pass &= check(m[:x].data == [7.0, 7.0, 9.0], "x: three measurements\n")
    pass &= check(is_complete(m) == true, "complete after all checkpoints\n")

    # no append after completion
    measure!(m, 11.0, 20.0)
    pass &= check(m[:x].data == [7.0, 7.0, 9.0], "no append after completion\n")

    # setindex!
    replacement = Measurement((s -> 3s), Float64[])
    m[:x] = replacement
    pass &= check(m[:x] === replacement, "setindex! replaces measurement\n")

    return pass
end

function test_measurements_reset()
    pass = true

    # single Measurement: vector
    m_vec = Measurement((s -> s), Float64[])
    measure!(m_vec, 1.0)
    measure!(m_vec, 2.0)
    reset!(m_vec)
    pass &= check(isempty(m_vec.data), "vector data cleared\n")

    # single Measurement: histogram
    bins = 0.0:1.0:4.0
    hist = fit(Histogram, [0.2, 1.4, 2.6], bins)
    m_hist = Measurement((s -> s), hist)
    reset!(m_hist)
    pass &= check(all(iszero, m_hist.data.weights), "histogram weights zeroed\n")

    # Measurements with interval schedule
    m_interval = Measurements(
        [
            :x => ((s -> s) => Float64[]),
            :h => ((s -> Int(s)) => fit(Histogram, Int[], 0:1:4)),
        ],
        interval=1.0,
    )
    measure!(m_interval, 1.0, 0.0)
    measure!(m_interval, 2.0, 1.0)
    reset!(m_interval)
    pass &= check(m_interval.schedule._checkpoint == 0.0, "interval checkpoint reset\n")
    pass &= check(isempty(m_interval[:x].data), "x data cleared\n")
    pass &= check(all(iszero, m_interval[:h].data.weights), "histogram weights zeroed\n")

    # Measurements with preallocated schedule
    m_pre = Measurements(
        [:x => ((s -> s) => Float64[])],
        [1.0, 2.0],
    )
    measure!(m_pre, 7.0, 5.0)
    reset!(m_pre)
    pass &= check(m_pre.schedule.checkpoint_idx == 1, "preallocated checkpoint_idx reset\n")
    pass &= check(isempty(m_pre[:x].data), "x data cleared\n")

    # unsupported container type
    m_bad = Measurement((s -> s), UnsupportedContainer())
    threw = try; reset!(m_bad); false; catch err; err isa ArgumentError; end
    pass &= check(threw, "unsupported container throws ArgumentError\n")

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
    @testset "constructors" begin
        @test test_measurement_constructors()
    end
    @testset "interval behavior" begin
        @test test_measurements_interval_behavior()
    end
    @testset "preallocated behavior" begin
        @test test_measurements_preallocated_behavior()
    end
    @testset "reset!" begin
        @test test_measurements_reset()
    end
    @testset "tau_int estimator" begin
        @test test_tau_int_estimator()
    end
end
