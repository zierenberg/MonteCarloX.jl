using MonteCarloX
using Test
using StatsBase

struct UnsupportedContainer end

function test_preallocated_schedule_constructor(; verbose=false)
    times = [3.0, 1.0, 2.0]
    schedule = PreallocatedSchedule(times)

    pass = true
    pass &= schedule.times == [1.0, 2.0, 3.0]
    pass &= schedule.checkpoint_idx == 1

    if verbose
        println("PreallocatedSchedule constructor test pass: $(pass)")
    end

    return pass
end

function test_measurement_pair_constructor(; verbose=false)
    m = Measurement((s -> s^2) => Float64[])

    pass = true
    pass &= m isa Measurement
    measure!(m, 3.0)
    pass &= m.data == [9.0]

    if verbose
        println("Measurement(pair) constructor test pass: $(pass)")
    end

    return pass
end

function test_measurements_interval_constructors(; verbose=false)
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
    pass &= m_dict.schedule isa IntervalSchedule
    pass &= m_dict.schedule.interval == 0.5
    pass &= m_dict.schedule._checkpoint == 0.0

    pass &= m_pairs.schedule isa IntervalSchedule
    pass &= m_pairs.schedule.interval == 0.25
    pass &= haskey(m_pairs.measurements, :x)
    pass &= haskey(m_pairs.measurements, :y)

    if verbose
        println("Measurements interval constructors test pass: $(pass)")
    end

    return pass
end

function test_measurements_dict_preallocated_constructor(; verbose=false)
    measurements = Dict{Symbol, Measurement}(
        :x => Measurement((s -> s), Float64[]),
        :y => Measurement((s -> 2s), Float64[]),
    )
    times = [3, 1, 2]

    m = Measurements(measurements, times)

    pass = true
    pass &= m.schedule isa PreallocatedSchedule
    pass &= m.schedule.times == [1.0, 2.0, 3.0]
    pass &= m.schedule.checkpoint_idx == 1

    if verbose
        println("Measurements(dict, times) constructor test pass: $(pass)")
    end

    return pass
end

function test_measurements_pairs_preallocated_constructor(; verbose=false)
    pairs = [
        :x => ((s -> s) => Float64[]),
        :y => ((s -> 2s) => Float64[]),
    ]
    times = [2, 1, 3]

    m = Measurements(pairs, times)

    pass = true
    pass &= m.schedule isa PreallocatedSchedule
    pass &= m.schedule.times == [1.0, 2.0, 3.0]
    pass &= haskey(m.measurements, :x)
    pass &= haskey(m.measurements, :y)
    pass &= m[:x] isa Measurement
    pass &= m[:y] isa Measurement

    if verbose
        println("Measurements(pairs, times) constructor test pass: $(pass)")
    end

    return pass
end

function test_measurements_setindex(; verbose=false)
    m = Measurements(
        Dict{Symbol, Measurement}(:x => Measurement((s -> s), Float64[])),
        [1.0, 2.0]
    )

    replacement = Measurement((s -> 3s), Float64[])
    m[:x] = replacement

    pass = true
    pass &= m[:x] === replacement
    pass &= m.measurements[:x] === replacement

    if verbose
        println("Measurements setindex! test pass: $(pass)")
    end

    return pass
end

function test_measurements_accessors(; verbose=false)
    m = Measurements(
        [
            :x => ((s -> s) => Float64[]),
        ],
        [1.0, 2.0, 3.0],
    )

    measure!(m, 7.0, 1.1)
    measure!(m, 9.0, 3.2)

    pass = true
    pass &= times(m) == [1.0, 2.0, 3.0]
    pass &= data(m, :x) == [7.0, 9.0, 9.0]

    if verbose
        println("Measurements accessor API test pass: $(pass)")
    end

    return pass
end

function test_measurements_interval_cadence(; verbose=false)
    m = Measurements(
        [
            :x => ((s -> s) => Float64[]),
        ],
        interval = 1.0,
    )

    pass = true

    # below first checkpoint (0.0) does not measure
    measure!(m, 5.0, -0.1)
    pass &= isempty(m[:x].data)
    pass &= m.schedule._checkpoint == 0.0

    # exactly at checkpoint measures once
    measure!(m, 7.0, 0.0)
    pass &= m[:x].data == [7.0]
    pass &= m.schedule._checkpoint == 1.0

    # if t jumps, interval schedule still measures at most once per call
    measure!(m, 9.0, 3.5)
    pass &= m[:x].data == [7.0, 9.0]
    pass &= m.schedule._checkpoint == 2.0

    if verbose
        println("Measurements interval cadence test pass: $(pass)")
    end

    return pass
end

function test_measurements_preallocated_measure_event_skipping(; verbose=false)
    m = Measurements(
        [
            :x => ((s -> s) => Float64[]),
            :y => ((s -> 2s) => Float64[]),
        ],
        [1.0, 2.0, 3.0]
    )

    # Skip over first two checkpoints in one call
    measure!(m, 7.0, 2.5)

    pass = true
    pass &= m.schedule.checkpoint_idx == 3
    pass &= m[:x].data == [7.0, 7.0]
    pass &= m[:y].data == [14.0, 14.0]

    # Process final checkpoint
    measure!(m, 9.0, 10.0)
    pass &= m.schedule.checkpoint_idx == 4
    pass &= m[:x].data == [7.0, 7.0, 9.0]
    pass &= m[:y].data == [14.0, 14.0, 18.0]

    # Further calls after completion should not append data
    measure!(m, 11.0, 20.0)
    pass &= m[:x].data == [7.0, 7.0, 9.0]
    pass &= m[:y].data == [14.0, 14.0, 18.0]

    if verbose
        println("Measurements preallocated event-skipping test pass: $(pass)")
    end

    return pass
end

function test_measurements_is_complete(; verbose=false)
    pass = true

    # Interval schedule is indefinite by definition
    m_interval = Measurements(
        [
            :x => ((s -> s) => Float64[]),
        ],
        interval=1.0
    )
    pass &= is_complete(m_interval) == false
    measure!(m_interval, 1.0, 100.0)
    pass &= is_complete(m_interval) == false

    # Preallocated schedule is complete once checkpoint index passes times length
    m_preallocated = Measurements(
        [
            :x => ((s -> s) => Float64[]),
        ],
        [1.0, 2.0]
    )
    pass &= is_complete(m_preallocated) == false
    measure!(m_preallocated, 5.0, 1.5)
    pass &= is_complete(m_preallocated) == false
    measure!(m_preallocated, 6.0, 2.5)
    pass &= is_complete(m_preallocated) == true

    if verbose
        println("Measurements is_complete test pass: $(pass)")
    end

    return pass
end

function test_measurement_reset_single(; verbose=false)
    m_vec = Measurement((s -> s), Float64[])
    measure!(m_vec, 1.0)
    measure!(m_vec, 2.0)
    reset!(m_vec)

    bins = 0.0:1.0:4.0
    hist = fit(Histogram, [0.2, 1.4, 2.6], bins)
    m_hist = Measurement((s -> s), hist)
    reset!(m_hist)

    pass = true
    pass &= isempty(m_vec.data)
    pass &= all(iszero, m_hist.data.weights)

    if verbose
        println("Measurement reset! (single) test pass: $(pass)")
    end

    return pass
end

function test_measurement_reset_special_cases(; verbose=false)
    # schedule reset! returns schedule and restores initial state
    interval_schedule = IntervalSchedule(0.3)
    interval_schedule._checkpoint = 1.2
    preallocated_schedule = PreallocatedSchedule([1.0, 2.0])
    preallocated_schedule.checkpoint_idx = 3

    pass = true
    pass &= reset!(interval_schedule) === interval_schedule
    pass &= interval_schedule._checkpoint == 0.0

    pass &= reset!(preallocated_schedule) === preallocated_schedule
    pass &= preallocated_schedule.checkpoint_idx == 1

    # supported but non-explicitly-handled container (e.g. empty! method exists but it is not an AbstractVector or Histogram) throws no error and empties container 
    custom_container = [1, 2, 3]
    m_custom = Measurement((s -> s), custom_container)
    pass &= try
        reset!(m_custom)
        isempty(custom_container)
    catch err
        false
    end

    # unsupported container throws informative ArgumentError
    m_bad = Measurement((s -> s), UnsupportedContainer())
    pass &= try
        reset!(m_bad)
        false
    catch err
        err isa ArgumentError
    end

    if verbose
        println("Measurement reset! special-cases test pass: $(pass)")
    end

    return pass
end

function test_measurements_reset_container(; verbose=false)
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
    pass &= m_interval.schedule._checkpoint > 0.0
    pass &= !isempty(m_interval[:x].data)

    reset!(m_interval)
    pass &= m_interval.schedule._checkpoint == 0.0
    pass &= isempty(m_interval[:x].data)
    pass &= all(iszero, m_interval[:h].data.weights)

    m_preallocated = Measurements(
        [
            :x => ((s -> s) => Float64[]),
        ],
        [1.0, 2.0],
    )
    measure!(m_preallocated, 7.0, 5.0)
    pass &= m_preallocated.schedule.checkpoint_idx == 3

    reset!(m_preallocated)
    pass &= m_preallocated.schedule.checkpoint_idx == 1
    pass &= isempty(m_preallocated[:x].data)

    if verbose
        println("Measurements reset! (container) test pass: $(pass)")
    end

    return pass
end

function run_measurements_testsets(; verbose=false)
    @testset "Measurements" begin
        @testset "PreallocatedSchedule constructor" begin
            @test test_preallocated_schedule_constructor(verbose=verbose)
        end
        @testset "Measurement(pair) constructor" begin
            @test test_measurement_pair_constructor(verbose=verbose)
        end
        @testset "Measurements interval constructors" begin
            @test test_measurements_interval_constructors(verbose=verbose)
        end
        @testset "Measurements(dict, times)" begin
            @test test_measurements_dict_preallocated_constructor(verbose=verbose)
        end
        @testset "Measurements(pairs, times)" begin
            @test test_measurements_pairs_preallocated_constructor(verbose=verbose)
        end
        @testset "Measurements setindex!" begin
            @test test_measurements_setindex(verbose=verbose)
        end
        @testset "accessor API" begin
            @test test_measurements_accessors(verbose=verbose)
        end
        @testset "interval cadence" begin
            @test test_measurements_interval_cadence(verbose=verbose)
        end
        @testset "measure! event skipping" begin
            @test test_measurements_preallocated_measure_event_skipping(verbose=verbose)
        end
        @testset "is_complete" begin
            @test test_measurements_is_complete(verbose=verbose)
        end
        @testset "reset! single measurement" begin
            @test test_measurement_reset_single(verbose=verbose)
        end
        @testset "reset! special cases" begin
            @test test_measurement_reset_special_cases(verbose=verbose)
        end
        @testset "reset! measurements container" begin
            @test test_measurements_reset_container(verbose=verbose)
        end
    end
    return true
end
