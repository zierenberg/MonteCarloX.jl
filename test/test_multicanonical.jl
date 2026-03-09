using MonteCarloX
using Random
using StatsBase
using Test

function test_multicanonical_accept(; verbose=false)
    rng = MersenneTwister(42)
    pass = true

    bins = 0.0:1.0:4.0
    lw = BinnedObject(bins, 0.0)
    alg = Multicanonical(rng, lw)

    step = 0.1
    function update!(x::Float64, alg::AbstractImportanceSampling)::Float64
        x_new = x + randn(alg.rng) * step
        if accept!(alg, x_new, x)
            return x_new
        else
            return x
        end
    end

    # updates
    x = 2.0  # start in bin 2
    for _ in 1:10
        x = update!(x, alg)
    end
    
    # test num_accepts
    pass &= alg.accepted >= 0 && alg.accepted <= 10
    # test acceptance rate (log_weight is zero, so acceptance should be 1)
    pass &= acceptance_rate(alg) == 1.0

    # reset
    reset!(alg)
    pass &= alg.accepted == 0
    pass &= all(iszero, ensemble(alg).histogram.weights)
    
    return pass
end


function test_multicanonical_weight_update_inplace(; verbose=false)
    rng = MersenneTwister(901)
    pass = true

    bins = 0.0:1.0:4.0
    lw = BinnedObject(bins, 0.0)
    alg = Multicanonical(rng, lw)

    w_before = copy(ensemble(alg).logweight.weights)
    ensemble(alg).histogram.weights .= [0.2, 0.8, 1.1, 2.5]

    if verbose
        # print the indices of the bins that are being updated
        println("logweight weights:", ensemble(alg).logweight.weights)
        println("histogram weights:", ensemble(alg).histogram.weights)
    end

    pass &= update!(alg) === nothing

    expected = copy(w_before)
    for i in eachindex(expected)
        h = ensemble(alg).histogram.weights[i]
        if h > 0
            expected[i] -= log(h)
        end
    end

    pass &= all(isapprox.(ensemble(alg).logweight.weights, expected))

    if verbose
        println("Multicanonical in-place update:")
        println("  before: $(w_before)")
        println("  after:  $(ensemble(alg).logweight.weights)")
    end

    return pass
end

function test_multicanonical_mode(; verbose=false)
    rng = MersenneTwister(902)
    bins_lw = 0.0:1.0:4.0

    lw = BinnedObject(bins_lw, 0.0)
    alg = Multicanonical(rng, lw)

    pass = true    
    pass &= try
        update!(alg; mode=:notavail)  # unsupported mode should throw
        false
    catch err
        err isa ArgumentError
    end

    if verbose
        println("Multicanonical mode compatibility: $(pass)")
    end

    return pass
end

function test_multicanonical_default_rng(; verbose=false)
    bins = 0.0:1.0:4.0
    lw = BinnedObject(bins, 0.0)
    alg = Multicanonical(lw)

    pass = alg.rng === Random.GLOBAL_RNG

    if verbose
        println("Multicanonical default RNG: $(pass)")
    end

    return pass
end

function test_multicanonical_accept_out_of_bounds(; verbose=false)
    rng = MersenneTwister(905)
    bins = 0.0:1.0:4.0
    lw = BinnedObject(bins, 0.0)
    alg = Multicanonical(rng, lw)

    x_old = 2.0
    x_new = 10.0  # outside right edge

    steps_before = alg.steps
    accepted_before = alg.accepted
    hist_before = copy(ensemble(alg).histogram.weights)

    pass = try
        accept!(alg, x_new, x_old)
        false
    catch err
        err isa BoundsError
    end
    pass &= alg.steps == steps_before
    pass &= alg.accepted == accepted_before
    pass &= all(ensemble(alg).histogram.weights .== hist_before)

    if verbose
        println("Multicanonical out-of-bounds delegated to BinnedObject: $(pass)")
    end

    return pass
end

function test_multicanonical_set_logweight_range_function(; verbose=false)
    rng = MersenneTwister(903)
    bins = 0.0:1.0:6.0
    lw = BinnedObject(bins, 0.0)
    alg = Multicanonical(rng, lw)

    pass = true

    fill!(ensemble(alg).logweight.weights, 0.0)

    pass &= set_logweight!(alg, (1.0, 4.0), x -> 10.0 + x) === nothing
    expected = [0.0, 11.5, 12.5, 13.5, 0.0, 0.0]
    pass &= all(isapprox.(ensemble(alg).logweight.weights, expected))

    pass &= set_logweight!(alg, 0.0:1.0:6.0, x -> -x^2) === nothing
    centers = collect(ensemble(alg).histogram.bins[1])
    pass &= all(isapprox.(ensemble(alg).logweight.weights, -centers.^2))

    if verbose
        println("Multicanonical set-logweight range/function API: $(pass)")
    end

    return pass
end

function test_multicanonical_set_logweight_range_errors(; verbose=false)
    rng = MersenneTwister(904)
    bins = 0.0:1.0:5.0
    lw = BinnedObject(bins, 0.0)
    alg = Multicanonical(rng, lw)

    pass = true

    pass &= try
        set_logweight!(alg, (100.0, 200.0), x -> x)
        false
    catch err
        err isa ArgumentError
    end

    if verbose
        println("Multicanonical set-logweight range errors: $(pass)")
    end

    return pass
end

function run_multicanonical_testsets(; verbose=false)
    @testset "Multicanonical" begin
        @testset "Accept/reject" begin
            @test test_multicanonical_accept(verbose=verbose)
        end
        @testset "Accept out-of-bounds" begin
            @test test_multicanonical_accept_out_of_bounds(verbose=verbose)
        end
        @testset "In-place update" begin
            @test test_multicanonical_weight_update_inplace(verbose=verbose)
        end
        @testset "Mode compatibility" begin
            @test test_multicanonical_mode(verbose=verbose)
        end
        @testset "Default RNG" begin
            @test test_multicanonical_default_rng(verbose=verbose)
        end
        @testset "Set logweight range/function" begin
            @test test_multicanonical_set_logweight_range_function(verbose=verbose)
        end
        @testset "Set logweight range errors" begin
            @test test_multicanonical_set_logweight_range_errors(verbose=verbose)
        end
    end
    return true
end
