using MonteCarloX
using Test
using StatsBase

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
        @testset "Measurements(dict, times)" begin
            @test test_measurements_dict_preallocated_constructor(verbose=verbose)
        end
        @testset "Measurements(pairs, times)" begin
            @test test_measurements_pairs_preallocated_constructor(verbose=verbose)
        end
        @testset "Measurements setindex!" begin
            @test test_measurements_setindex(verbose=verbose)
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
        @testset "reset! measurements container" begin
            @test test_measurements_reset_container(verbose=verbose)
        end
    end
    return true
end
