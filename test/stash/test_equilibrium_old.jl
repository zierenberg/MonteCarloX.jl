using MonteCarloX
using Random
using LinearAlgebra
using StatsBase

# TODO: use this also for PT and adaptive overlap stories etc
""" Testing metropolis on unimodaL probability distribution"""
function test_unimodal_metropolis(;verbose = false)
    log_weight_unimodal(x, x0, var) = -(x - x0)^2 / 2 / var - log(sqrt(2 * pi * var))
    var = 1.0
    dx = 1.0
    list_x0 = [0.,1.,2.,3.]
    pass = true

    rng = MersenneTwister(42)
    samples = Int(1e6)
    list_x = zeros(samples)
    bound_kld = 0.1
    bound_x = 0.1

    for x0 in list_x0
        log_weight_sampling(x) = log_weight_unimodal(x, x0, var)
        
        x = x0
        for update in 1:samples 
            # Metropolis:
            x_new = step(x, dx, rng)
            if accept(rng, log_weight_sampling, x_new, x)
                x = x_new
            end
            list_x[update] = x
        end
        # sort into binned histogram
        # bin size/ 2*sigma ~    #elements per bin (~100)/ samples*0.6/
        bin_x =    1e4 * sqrt(var) * 2 / 0.6 / samples
        P_meas = fit(Histogram, list_x, minimum(list_x):bin_x:maximum(list_x) + bin_x, closed = :left)
        P_meas = normalize(P_meas)
        kld = abs(kldivergence(P_meas, x->exp(log_weight_sampling(x))))
        if verbose
            println("result kldivergence P_meas to P_true = $(kld)")
        end
        pass &= kld < bound_kld

        ###########################################################################
        est_x0 = sum(list_x) / length(list_x)
        diff_est_x0 = abs(est_x0 - x0)
        if verbose
            println("result estimate x0 = $(est_x0) vs x0 = $(x0) [diff = $(diff_est_x0)]")
        end
        pass &= diff_est_x0 < bound_x
    end
    return pass
end

""" Testing metropolis on unimodaL probability distribution"""
function test_2D_unimodal_metropolis(;verbose = false)
    log_weight_unimodal(x::Float64, y::Float64, x0::Float64, y0::Float64)::Float64 = -(x - x0)^2 / 2.0 - (y - y0)^2 / 2.0
    dx = dy = 0.1
    list_x0 = [0.0,1.0,2.0,3.0]
    y0 = 1.0
    pass = true

    rng = MersenneTwister(1000)

    for x0 in list_x0
        log_weight_sampling(x::Float64, y::Float64)::Float64 = log_weight_unimodal(x, y, x0, y0)
        samples = Int(1e6)
        list_x = zeros(samples)
        list_y = zeros(samples)
        x = x0
        y = y0
        # thermalization 100 updates
        for update in 1:samples + 100
            # Metropolis:
            x_new = x - dx + 2.0 * dx * rand(rng)
            y_new = y - dy + 2.0 * dy * rand(rng)
            if accept(rng, log_weight_sampling, (x_new, y_new), (x, y))
                x = x_new
                y = y_new
            end
            if update > 100
                list_x[update - 100] = x
                list_y[update - 100] = y
            end
        end
        # sort into binned histogram
        # bin size/ 2*sigma ~    #elements per bin (~100)/ samples*0.6/
        ###########################################################################
        est_x0 = sum(list_x) / length(list_x)
        diff_est_x0 = abs(est_x0 - x0)
        if verbose
            println("result estimate x0 = $(est_x0) vs x0 = $(x0) [diff = $(diff_est_x0)]")
        end
        pass &= diff_est_x0 < 0.1
    end
    return pass
end

# TODO: test speed difference
""" Testing sweep on unimodaL probability distribution"""
function test_unimodal_sweep(;verbose = false)
    log_weight_unimodal(x, x0, var) = -(x - x0)^2 / 2 / var - log(sqrt(2 * pi * var))
    var = 1.0

    rng = MersenneTwister(1000)

    pass = true
    list_x0 = [0.0,1.,2.,3.]
    for x0 in list_x0
        log_weight_sampling(x) = log_weight_unimodal(x, x0, var)
        samples = Int(1e5)
        list_x = zeros(samples)

        if verbose
            println("\nTesting sweep with multiple functions")
        end
        system = System(x0)
        stats1 = Stats(0, 0)
        stats2 = Stats(0, 0)
        # TODO: optimize this
        list_updates = [()->update(log_weight_sampling, system, 0.1, rng, stats1),
                        ()->update(log_weight_sampling, system, 0.2, rng, stats2),
                        ()->update(log_weight_sampling, system, 0.1, rng)]
        list_probabilities = ProbabilityWeights([0.2,0.4,0.4])

        # perfrom 100 updates as thermalization
        sweep(list_updates, list_probabilities, rng, number_updates = 100)

        stats2 = Stats(0, 0)
        for updates in 1:samples
            sweep(list_updates, list_probabilities, rng, number_updates = 10)
            list_x[updates] = system.x
        end


        ###########################################################################
        num_update1 = stats1.accept + stats1.reject
        num_update1_target = 0.2 * samples * 10 + 100
        rel_diff_num_update1 = abs(num_update1 - num_update1_target) / num_update1_target
        if verbose
            println("number calls to update1= $(num_update1) vs target = $(num_update1_target) [rel_diff = $(rel_diff_num_update1)]")
        end
        pass &= rel_diff_num_update1 < 0.01
        ###########################################################################
        num_update2 = stats2.accept + stats2.reject
        num_update2_target = 0.4 * samples * 10
        rel_diff_num_update2 = abs(num_update2 - num_update2_target) / num_update2_target
        if verbose
            println("number calls to update2= $(num_update2) vs target = $(num_update2_target) [rel_diff = $(rel_diff_num_update2)]")
        end
        pass &= rel_diff_num_update2 < 0.01
        ##########################################################################
        # sort into binned histogram
        # bin size/ 2*sigma ~    #elements per bin (~100)/ samples*0.6/
        bin_x =    1e4 * sqrt(var) * 2 / 0.6 / samples
        P_meas = fit(Histogram, list_x, minimum(list_x):bin_x:maximum(list_x) + bin_x, closed = :left)
        P_meas = normalize(P_meas)
        kld = abs(kldivergence(P_meas, x->exp(log_weight_sampling(x))))
        if verbose
            println("result kldivergence P_meas to P_true = $(kld)")
        end
        pass &= kld < 0.1
        ###########################################################################
        est_x0 = sum(list_x) / length(list_x)
        diff_est_x0 = abs(est_x0 - x0)
        if verbose
            println("result estimate x0 = $(est_x0) vs x0 = $(x0) [diff = $(diff_est_x0)]")
        end
        pass &= diff_est_x0 < 0.1


        if verbose
            println("\nTesting sweep with single function")
        end
        update1() = update(log_weight_sampling, system, 0.1, rng)
        for updates in 1:samples
            sweep(update1, rng, number_updates = 10)
            list_x[updates] = system.x
        end
        ##########################################################################
        # sort into binned histogram
        # bin size/ 2*sigma ~    #elements per bin (~100)/ samples*0.6/
        bin_x =    1e4 * sqrt(var) * 2 / 0.6 / samples
        P_meas = fit(Histogram, list_x, minimum(list_x):bin_x:maximum(list_x) + bin_x, closed = :left)
        P_meas = normalize(P_meas)
        kld = abs(kldivergence(P_meas, x->exp(log_weight_sampling(x))))
        if verbose
            println("result kldivergence P_meas to P_true = $(kld)")
        end
        pass &= kld < 0.1
        ###########################################################################
        est_x0 = sum(list_x) / length(list_x)
        diff_est_x0 = abs(est_x0 - x0)
        if verbose
            println("result estimate x0 = $(est_x0) vs x0 = $(x0) [diff = $(diff_est_x0)]")
        end
        pass &= diff_est_x0 < 0.1
    end
    return pass
end


""" Testing reweighting on unimodaL probability distribution"""
# TODO: write
function test_unimodal_reweighting()
    return pass
end


###############################################################################
###############################################################################
###############################################################################

mutable struct System
    x::Float64
end

mutable struct Stats
    accept::Float64
    reject::Float64
end

"""Metropolis accept/reject for symmetric proposals"""
function accept(rng::AbstractRNG, log_weight_sampling, new_state, old_state)
    log_new = new_state isa Tuple ? log_weight_sampling(new_state...) : log_weight_sampling(new_state)
    log_old = old_state isa Tuple ? log_weight_sampling(old_state...) : log_weight_sampling(old_state)
    log_ratio = log_new - log_old
    return log_ratio > 0 || rand(rng) < exp(log_ratio)
end

"""Apply an update function a fixed number of times"""
function sweep(update_fn::Function, rng::AbstractRNG; number_updates::Int=1)
    for _ in 1:number_updates
        update_fn()
    end
end

"""Sweep over a weighted set of update functions"""
function sweep(list_updates::Vector{Function}, list_probabilities::ProbabilityWeights, rng::AbstractRNG; number_updates::Int=1)
    for _ in 1:number_updates
        update_fn = StatsBase.sample(rng, list_updates, list_probabilities)
        update_fn()
    end
end


function update(log_weight_sampling, system::System, dx, rng::AbstractRNG)
    x_new = step(system.x, dx, rng)
    if accept(rng, log_weight_sampling, x_new, system.x)
        system.x = x_new
    end
end

# TODO: test second version with statistics could this be implemented in update w/o speed loss? -> could write a macro like @assert
function update(log_weight_sampling, system::System, dx, rng::AbstractRNG, stats::Stats)
    x_new = step(system.x, dx, rng)
    if accept(rng, log_weight_sampling, x_new, system.x)
        system.x = x_new
        stats.accept += 1
    else
        stats.reject += 1
    end
end

function step(x, dx, rng)
    return x - dx + 2 * dx * rand(rng)
end
