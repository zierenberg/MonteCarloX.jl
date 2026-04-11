using MonteCarloX
using Random
using StatsBase
using StatsBase: normalize, kldivergence
using Distributions
using Test

function test_importance_sampling_basics()
    rng = MersenneTwister(42)
    logdensity(x) = -0.5 * x^2
    alg = ImportanceSampling(rng, logdensity)

    pass = true
    pass &= check(alg.rng === rng, "rng stored\n")
    pass &= check(alg.ensemble === FunctionEnsemble(logdensity), "ensemble stored\n")
    pass &= check(hasmethod(accept!, Tuple{typeof(alg), Float64, Float64}), "accept! defined\n")
    pass &= check(hasmethod(reset!, Tuple{typeof(alg)}), "reset! defined\n")

    pass &= check(ensemble(alg) isa FunctionEnsemble, "ensemble accessor\n")
    pass &= check(logweight(alg) === logdensity, "logweight accessor\n")

    pass &= check(MonteCarloX.should_record_visit(ensemble(alg)) == false, "no visit recording\n")
    pass &= check(MonteCarloX.record_visit!(ensemble(alg), 1.0) === nothing, "record_visit! is no-op\n")

    return pass
end

function test_metropolis_1d_gaussian()
    rng = MersenneTwister(42)

    mu = 1.0; sigma = 1.0
    logweight(x) = -0.5 * ((x - mu) / sigma)^2

    alg = Metropolis(rng, logweight)

    bins = -10.0:0.1:10.0
    measurements = Measurements([
        :timeseries => (x->x) => Float64[],
        :histogram => (x->x) => fit(Histogram, Float64[], bins)
    ], interval=10)

    step = sigma
    function update(x::Float64, alg::Metropolis)::Float64
        x_new = x + randn(alg.rng) * step
        if accept!(alg, x_new, x)
            return x_new
        else
            return x
        end
    end

    # thermalization
    x = mu
    for _ in 1:Int(1e3)
        x = update(x, alg)
    end
    reset!(alg)

    # production
    for i in 1:Int(1e6)
        x = update(x, alg)
        measure!(measurements, x, i)
    end

    # verify histogram consistency
    hist_timeseries = fit(Histogram, measurements[:timeseries].data, bins)
    @assert hist_timeseries == measurements[:histogram].data

    hist_measured = normalize(measurements[:histogram].data)
    theoretical_pdf(x) = pdf(Normal(mu, sigma), x)
    kld = kldivergence(hist_measured, theoretical_pdf)

    mean_x = mean(measurements[:timeseries].data)
    std_x = std(measurements[:timeseries].data)
    samples = length(measurements[:timeseries].data)
    expected_mean_error = sigma / sqrt(samples)
    expected_std_error = sigma / sqrt(2 * samples)

    pass = true
    pass &= check(abs(mean_x - mu) < expected_mean_error * 3, "mean within 3-sigma\n")
    pass &= check(abs(std_x - sigma) < expected_std_error * 3, "std within 3-sigma\n")
    pass &= check(acceptance_rate(alg) > 0.5, "acceptance rate > 0.5\n")
    pass &= check(kld < 0.5, "KL divergence small\n")

    return pass
end

function test_metropolis_2d_gaussian()
    rng = MersenneTwister(100)

    mu = [1.0, 2.0]
    logweight(x) = -0.5 * ((x[1] - mu[1])^2 + (x[2] - mu[2])^2)

    alg = Metropolis(rng, logweight)

    measurements = Measurements([
        :x => (s -> s[1]) => Float64[],
        :y => (s -> s[2]) => Float64[]
    ], interval=1)

    function update(x::Vector{Float64}, alg::Metropolis)::Vector{Float64}
        x_new = x + randn(alg.rng, length(x))
        if accept!(alg, x_new, x)
            return x_new
        else
            return x
        end
    end

    x = mu
    for _ in 1:1000
        x = update(x, alg)
    end
    reset!(alg)

    samples = 50000
    for i in 1:samples
        x = update(x, alg)
        measure!(measurements, x, i)
    end

    samples_x = measurements[:x].data
    samples_y = measurements[:y].data
    hist_x = normalize(fit(Histogram, samples_x, -2.0:0.4:5.0, closed=:left))
    hist_y = normalize(fit(Histogram, samples_y, -1.0:0.4:6.0, closed=:left))

    kld_x = kldivergence(hist_x, x -> pdf(Normal(mu[1], 1.0), x))
    kld_y = kldivergence(hist_y, y -> pdf(Normal(mu[2], 1.0), y))

    pass = true
    pass &= check(abs(mean(samples_x) - mu[1]) < 0.1, "mean x correct\n")
    pass &= check(abs(mean(samples_y) - mu[2]) < 0.1, "mean y correct\n")
    pass &= check(acceptance_rate(alg) > 0.4, "acceptance rate > 0.4\n")
    pass &= check(kld_x < 0.5, "KL divergence x small\n")
    pass &= check(kld_y < 0.5, "KL divergence y small\n")

    return pass
end

function test_metropolis_acceptance_tracking()
    rng = MersenneTwister(200)
    logweight(x) = -0.5 * x^2

    alg = Metropolis(rng, logweight)

    measurements = Measurements([
        :timeseries => (x -> x) => Float64[]
    ], interval=1)

    function update(x::Float64, alg::Metropolis)::Float64
        x_new = x + randn(alg.rng) * 1.0
        if accept!(alg, x_new, x)
            return x_new
        else
            return x
        end
    end

    x = 0.0
    for i in 1:10000
        x = update(x, alg)
        measure!(measurements, x, i)
    end

    acceptance = acceptance_rate(alg)

    pass = true
    pass &= check(alg.steps == 10000, "steps counted\n")
    pass &= check(acceptance > 0.0 && acceptance < 1.0, "acceptance rate in (0,1)\n")
    pass &= check(length(measurements[:timeseries].data) == 10000, "all samples collected\n")

    reset!(alg)
    pass &= check(alg.steps == 0, "steps reset\n")
    pass &= check(alg.accepted == 0, "accepted reset\n")
    pass &= check(acceptance_rate(alg) == 0.0, "acceptance rate reset\n")

    return pass
end

function test_metropolis_temperature_effects()
    pass = true

    for β in [0.5, 1.0, 2.0]
        rng = MersenneTwister(300 + Int(β * 100))

        energy(x) = x^2
        logweight(E) = -β * E

        alg = Metropolis(rng, logweight)

        measurements = Measurements([
            :timeseries => (x -> x) => Float64[]
        ], interval=1)

        function update(x::Float64, alg::Metropolis)::Float64
            x_new = x + randn(alg.rng) * 0.5
            delta_E = energy(x_new) - energy(x)
            if accept!(alg, delta_E)
                return x_new
            else
                return x
            end
        end

        x = 0.0
        n_samples = 10000
        for i in 1:n_samples
            x = update(x, alg)
            measure!(measurements, x, i)
        end

        # at finite temperature, variance sigma^2 = 1/(2*beta)
        samples_data = measurements[:timeseries].data
        est_var = var(samples_data)
        expected_var = 1.0 / (2.0 * β)

        pass &= check(est_var > expected_var * 0.7, "beta=$β: variance not too low\n")
        pass &= check(est_var < expected_var * 1.3, "beta=$β: variance not too high\n")
        pass &= check(length(samples_data) == n_samples, "beta=$β: all samples collected\n")
    end

    return pass
end

function test_metropolis_proposal_invariance()
    logweight(x) = -0.5 * (x - 0.5)^2

    samples = 20000

    # narrow proposal
    rng_a = MersenneTwister(400)
    alg_a = Metropolis(rng_a, logweight)
    measurements_a = Measurements([:timeseries => (x -> x) => Float64[]], interval=1)

    function update_a(x::Float64, alg::Metropolis)::Float64
        x_new = x + randn(alg.rng) * 0.5
        accept!(alg, x_new, x) ? x_new : x
    end

    x_a = 0.0
    for i in 1:samples
        x_a = update_a(x_a, alg_a)
        measure!(measurements_a, x_a, i)
    end

    # wide proposal
    rng_b = MersenneTwister(400)
    alg_b = Metropolis(rng_b, logweight)
    measurements_b = Measurements([:timeseries => (x -> x) => Float64[]], interval=1)

    function update_b(x::Float64, alg::Metropolis)::Float64
        x_new = x + randn(alg.rng) * 2.0
        accept!(alg, x_new, x) ? x_new : x
    end

    x_b = 0.0
    for i in 1:samples
        x_b = update_b(x_b, alg_b)
        measure!(measurements_b, x_b, i)
    end

    samples_a = measurements_a[:timeseries].data
    samples_b = measurements_b[:timeseries].data
    edges = -2.0:0.3:4.0
    hist_a = normalize(fit(Histogram, samples_a, edges, closed=:left))
    hist_b = normalize(fit(Histogram, samples_b, edges, closed=:left))
    theoretical_pdf(x) = pdf(Normal(0.5, 1.0), x)

    kld_a = kldivergence(hist_a, theoretical_pdf)
    kld_b = kldivergence(hist_b, theoretical_pdf)

    pass = true
    pass &= check(abs(mean(samples_a) - 0.5) < 0.1, "narrow: mean correct\n")
    pass &= check(abs(mean(samples_b) - 0.5) < 0.1, "wide: mean correct\n")
    pass &= check(kld_a < 0.5, "narrow: KL divergence small\n")
    pass &= check(kld_b < 0.5, "wide: KL divergence small\n")
    pass &= check(abs(acceptance_rate(alg_a) - acceptance_rate(alg_b)) > 0.1, "different acceptance rates\n")
    pass &= check(length(samples_a) == samples && length(samples_b) == samples, "all samples collected\n")

    return pass
end

function test_metropolis_boltzmann_overloads_and_constructor()
    rng = MersenneTwister(500)
    β = 0.75

    ens = BoltzmannEnsemble(β)
    lw = logweight(ens)

    e_int = 4
    e_real = 2.5
    e_vec = [1, -2, 3]
    e_mat = [1.0 2.0; -3.0 4.0]

    pass = true
    pass &= check(lw(e_int) == -β * e_int, "lw(Int)\n")
    pass &= check(lw(e_real) == -β * e_real, "lw(Real)\n")
    pass &= check(lw(e_vec) == -β * sum(e_vec), "lw(Vector)\n")
    pass &= check(lw(e_mat) == -β * sum(e_mat), "lw(Matrix)\n")

    # convenience constructor
    alg = Metropolis(rng; β=β)
    pass &= check(alg.rng === rng, "rng stored\n")
    pass &= check(ensemble(alg) isa BoltzmannEnsemble, "BoltzmannEnsemble created\n")
    pass &= check(alg.steps == 0, "steps == 0\n")
    pass &= check(alg.accepted == 0, "accepted == 0\n")
    pass &= check(logweight(ensemble(alg), e_int) == -β * e_int, "logweight via ensemble\n")
    pass &= check(logweight(ensemble(alg), e_vec) == -β * sum(e_vec), "logweight via ensemble (vec)\n")

    return pass
end

function test_glauber_basics()
    rng = MersenneTwister(777)
    β = 0.8

    glauber = Glauber(rng; β=β)
    pass = true

    pass &= check(glauber.rng === rng, "rng stored\n")
    pass &= check(glauber.steps == 0, "steps == 0\n")
    pass &= check(glauber.accepted == 0, "accepted == 0\n")

    # strongly favorable move should almost always be accepted
    accepted = 0
    for _ in 1:1_000
        accepted += accept!(glauber, -8.0)
    end
    pass &= check(accepted > 990, "favorable moves accepted (>990/1000)\n")
    pass &= check(glauber.steps == 1_000, "steps counted\n")
    pass &= check(0.0 <= acceptance_rate(glauber) <= 1.0, "acceptance rate in [0,1]\n")

    return pass
end

function test_heatbath_basics()
    rng = MersenneTwister(778)
    β = 0.8

    hb = HeatBath(rng; β=β)

    pass = true
    pass &= check(hb.rng === rng, "rng stored\n")
    pass &= check(hb.β == β, "beta stored\n")
    pass &= check(hb.steps == 0, "steps == 0\n")

    return pass
end

@testset "Metropolis" begin
    @testset "Importance Sampling" begin
        @test test_importance_sampling_basics()
    end
    @testset "1D Gaussian" begin
        @test test_metropolis_1d_gaussian()
    end
    @testset "2D Gaussian" begin
        @test test_metropolis_2d_gaussian()
    end
    @testset "Acceptance tracking" begin
        @test test_metropolis_acceptance_tracking()
    end
    @testset "Temperature effects" begin
        @test test_metropolis_temperature_effects()
    end
    @testset "Proposal invariance" begin
        @test test_metropolis_proposal_invariance()
    end
    @testset "Boltzmann overloads and constructor" begin
        @test test_metropolis_boltzmann_overloads_and_constructor()
    end
    @testset "Glauber basics" begin
        @test test_glauber_basics()
    end
    @testset "HeatBath basics" begin
        @test test_heatbath_basics()
    end
end
