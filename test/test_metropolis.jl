using MonteCarloX
using Random
using StatsBase
using StatsBase: normalize, kldivergence
using Distributions
using Test

"""
Test Metropolis sampler on a 1D Gaussian with scheduled measurements and KL divergence check
"""
function test_metropolis_1d_gaussian(; verbose=false)
    rng = MersenneTwister(42)
    
    # Target distribution: N(μ=1.0, σ=1.0)
    μ = 1.0; σ = 1.0
    logweight(x) = -0.5 * ((x - μ) / σ)^2
    
    # Create Metropolis sampler with custom log weight function
    alg = Metropolis(rng, logweight)
    
    # Create measurement for tracking samples (discretize histogram with precision ~ σ/10 from μ-10*σ to μ+10*σ)
    bins = -10.0:0.1:10.0
    measurements = Measurements([
        :timeseries => (x->x) => Float64[],
        :histogram => (x->x) => fit(Histogram, Float64[], bins)
    ], interval=10)
    
    # Local update function - only depends on x and alg
    step = σ
    function update(x::Float64, alg::Metropolis)::Float64
        x_new = x + randn(alg.rng) * step
        log_ratio = alg.logweight(x_new) - alg.logweight(x)
        if accept!(alg, log_ratio)
            return x_new
        else
            return x
        end
    end
    
    # Thermalization
    x = μ
    for _ in 1:Int(1e3)
        x = update(x, alg)
    end
    reset_statistics!(alg)
    
    # Production run - collect samples using Measurement framework
    for i in 1:Int(1e6)
        x = update(x, alg)
        measure!(measurements, x, i)
    end
    
    # create histogram from timeseries and compare to histogram from measurements
    hist_timeseries = fit(Histogram, measurements[:timeseries].data, bins)
    @assert hist_timeseries == measurements[:histogram].data
    
    # Normalize histograms to probability density
    hist_measured = normalize(measurements[:histogram].data)
    
    # Define theoretical Gaussian PDF at bin edges
    theoretical_pdf(x) = pdf(Normal(μ, σ), x)
    
    # Calculate KL divergence
    kld = kldivergence(hist_measured, theoretical_pdf)
    
    # Calculate statistics
    mean_x = mean(measurements[:timeseries].data)
    std_x = std(measurements[:timeseries].data)

    # expected error from central limit theorem: σ/√N ~ 0.01 for mean, σ/√(2N) ~ 0.007 for std
    samples = length(measurements[:timeseries].data)
    expected_mean_error = σ / sqrt(samples)
    expected_std_error = σ / sqrt(2 * samples)
    
    if verbose
        println("1D Gaussian Test:")
        println("  Target: μ=$(μ), σ=$(σ)")
        println("  Sampled mean: $(mean_x), expected 3-sigma deviation: $(μ) +/- $(3*expected_mean_error)")
        println("  Sampled std: $(std_x), expected 3-sigma deviation: $(σ) +/- $(3*expected_std_error)")
        println("  Histogram bins: $(length(hist_measured.edges[1]) - 1)")
        println("  KL divergence: $(kld)")
        println("  Acceptance rate: $(acceptance_rate(alg))")
    end

    # Check convergence
    pass = true
    pass &= abs(mean_x - μ) < expected_mean_error * 3  # Allow for 3-sigma error
    pass &= abs(std_x - σ   ) < expected_std_error * 3  # Allow for 3-sigma error
    pass &= acceptance_rate(alg) > 0.5  # Typical for 1D Gaussian with step size ~ σ
    pass &= kld < 0.5  # KL divergence should be small
    
    return pass
end

"""
Test Metropolis sampler on 2D Gaussian distribution
"""
function test_metropolis_2d_gaussian(; verbose=false)
    rng = MersenneTwister(100)
    
    # Target: 2D Gaussian centered at (1.0, 2.0)
    μx, μy = 1.0, 2.0
    logweight(x, y) = -0.5 * ((x - μx)^2 + (y - μy)^2)
    
    alg = Metropolis(rng, logweight)
    
    # Create measurements for tracking both coordinates
    measurements = Measurements([
        :x => (s -> s[1]) => Float64[],
        :y => (s -> s[2]) => Float64[]
    ], interval=1)
    
    # Local update function - only depends on x, y and alg
    function update(x::Float64, y::Float64, alg::Metropolis)::Tuple{Float64, Float64}
        x_new = x + randn(alg.rng)
        y_new = y + randn(alg.rng)
        log_ratio = alg.logweight(x_new, y_new) - alg.logweight(x, y)
        if accept!(alg, log_ratio)
            return (x_new, y_new)
        else
            return (x, y)
        end
    end
    
    # Thermalization
    x, y = μx, μy
    for _ in 1:1000
        x, y = update(x, y, alg)
    end
    reset_statistics!(alg)
    
    # Production run using scheduled measurements
    samples = 50000
    for i in 1:samples
        x, y = update(x, y, alg)
        measure!(measurements, (x, y), i)
    end
    
    # Create histograms for each dimension
    samples_x = measurements[:x].data
    samples_y = measurements[:y].data
    hist_x_edges = -2.0:0.4:5.0
    hist_y_edges = -1.0:0.4:6.0
    hist_x = fit(Histogram, samples_x, hist_x_edges, closed=:left)
    hist_y = fit(Histogram, samples_y, hist_y_edges, closed=:left)
    
    # Normalize histograms
    hist_x = normalize(hist_x)
    hist_y = normalize(hist_y)
    
    # Theoretical marginal distributions
    dist_x = Normal(μx, 1.0)
    dist_y = Normal(μy, 1.0)
    theoretical_pdf_x(x) = pdf(dist_x, x)
    theoretical_pdf_y(y) = pdf(dist_y, y)
    
    # Calculate KL divergences
    kld_x = kldivergence(hist_x, theoretical_pdf_x)
    kld_y = kldivergence(hist_y, theoretical_pdf_y)
    
    mean_x = mean(samples_x)
    mean_y = mean(samples_y)
    
    if verbose
        println("2D Gaussian Test:")
        println("  Target: μ=($(μx), $(μy))")
        println("  Sampled: μ=($(mean_x), $(mean_y))")
        println("  KL divergence X: $(kld_x)")
        println("  KL divergence Y: $(kld_y)")
        println("  Acceptance rate: $(acceptance_rate(alg))")
    end
    
    pass = true
    pass &= abs(mean_x - μx) < 0.1
    pass &= abs(mean_y - μy) < 0.1
    pass &= acceptance_rate(alg) > 0.4
    pass &= kld_x < 0.5  # KL divergence should be small
    pass &= kld_y < 0.5
    
    return pass
end

"""
Test acceptance rate statistics tracking using Measurement framework
"""
function test_metropolis_acceptance_tracking(; verbose=false)
    rng = MersenneTwister(200)
    
    # Simple target: standard Gaussian
    logweight(x) = -0.5 * x^2
    
    alg = Metropolis(rng, logweight)
    
    # Create measurement for tracking samples
    measurements = Measurements([
        :timeseries => (x -> x) => Float64[]
    ], interval=1)
    
    # Local update function - only depends on x and alg
    function update(x::Float64, alg::Metropolis)::Float64
        x_new = x + randn(alg.rng) * 1.0
        log_ratio = alg.logweight(x_new) - alg.logweight(x)
        if accept!(alg, log_ratio)
            return x_new
        else
            return x
        end
    end
    
    # Run with different proposal widths to get varying acceptance rates
    x = 0.0
    for i in 1:10000
        x = update(x, alg)
        measure!(measurements, x, i)
    end
    
    acceptance = acceptance_rate(alg)
    
    if verbose
        println("Acceptance Rate Test:")
        println("  Total steps: $(alg.steps)")
        println("  Accepted: $(alg.accepted)")
        println("  Acceptance rate: $(acceptance)")
        println("  Sample mean: $(mean(measurements[:timeseries].data))")
    end
    
    pass = true
    pass &= alg.steps == 10000
    pass &= acceptance > 0.0 && acceptance < 1.0
    pass &= length(measurements[:timeseries].data) == 10000  # All samples collected
    
    # Test reset
    reset_statistics!(alg)
    pass &= alg.steps == 0
    pass &= alg.accepted == 0
    pass &= acceptance_rate(alg) == 0.0
    
    return pass
end

"""
Test Metropolis with BoltzmannLogWeight showing temperature effects using Measurement framework
"""
function test_metropolis_temperature_effects(; verbose=false)
    pass = true
    
    for β_inv in [0.5, 1.0, 2.0]  # Different temperatures
        rng = MersenneTwister(300 + Int(β_inv * 100))
        
        # Energy landscape: quadratic potential E(x) = x^2
        energy(x) = x^2
        logweight(x) = -β_inv * energy(x)
        
        alg = Metropolis(rng, logweight)
        
        # Create measurement for tracking samples
        measurements = Measurements([
            :timeseries => (x -> x) => Float64[]
        ], interval=1)
        
        # Local update function - only depends on x and alg
        function update(x::Float64, alg::Metropolis)::Float64
            x_new = x + randn(alg.rng) * 0.5
            log_ratio = alg.logweight(x_new) - alg.logweight(x)
            if accept!(alg, log_ratio)
                return x_new
            else
                return x
            end
        end
        
        # Run sampler using measurements
        x = 0.0
        samples = 10000
        for i in 1:samples
            x = update(x, alg)
            measure!(measurements, x, i)
        end
        
        # At finite temperature, variance σ² ≈ 1/(2β)
        samples_data = measurements[:timeseries].data
        est_var = var(samples_data)
        expected_var = 1.0 / (2.0 * β_inv)
        
        if verbose
            println("Temperature test β=$(β_inv):")
            println("  Estimated variance: $(est_var)")
            println("  Expected variance: $(expected_var)")
            println("  Ratio: $(est_var / expected_var)")
            println("  Samples collected: $(length(samples_data))")
        end
        
        # Allow 30% deviation due to finite sampling
        pass &= est_var > expected_var * 0.7
        pass &= est_var < expected_var * 1.3
        pass &= length(samples_data) == samples  # All samples collected
    end
    
    return pass
end

"""
Test that different proposal distributions converge to same target using Measurements and KL divergence
"""
function test_metropolis_proposal_invariance(; verbose=false)
    logweight(x) = -0.5 * (x - 0.5)^2
    
    samples = 20000
    
    # Test A: narrow proposal
    rng_a = MersenneTwister(400)
    alg_a = Metropolis(rng_a, logweight)
    
    measurements_a = Measurements([
        :timeseries => (x -> x) => Float64[]
    ], interval=1)
    
    function update_a(x::Float64, alg::Metropolis)::Float64
        x_new = x + randn(alg.rng) * 0.5  # Narrow proposal
        log_ratio = alg.logweight(x_new) - alg.logweight(x)
        if accept!(alg, log_ratio)
            return x_new
        else
            return x
        end
    end
    
    x_a = 0.0
    for i in 1:samples
        x_a = update_a(x_a, alg_a)
        measure!(measurements_a, x_a, i)
    end
    
    # Test B: wide proposal
    rng_b = MersenneTwister(400)
    alg_b = Metropolis(rng_b, logweight)
    
    measurements_b = Measurements([
        :timeseries => (x -> x) => Float64[]
    ], interval=1)
    
    function update_b(x::Float64, alg::Metropolis)::Float64
        x_new = x + randn(alg.rng) * 2.0  # Wide proposal
        log_ratio = alg.logweight(x_new) - alg.logweight(x)
        if accept!(alg, log_ratio)
            return x_new
        else
            return x
        end
    end
    
    x_b = 0.0
    for i in 1:samples
        x_b = update_b(x_b, alg_b)
        measure!(measurements_b, x_b, i)
    end
    
    # Create histograms for both
    samples_a = measurements_a[:timeseries].data
    samples_b = measurements_b[:timeseries].data
    edges = -2.0:0.3:4.0
    hist_a = fit(Histogram, samples_a, edges, closed=:left)
    hist_b = fit(Histogram, samples_b, edges, closed=:left)
    
    # Normalize histograms
    hist_a = normalize(hist_a)
    hist_b = normalize(hist_b)
    
    # Theoretical target distribution
    dist_target = Normal(0.5, 1.0)
    theoretical_pdf(x) = pdf(dist_target, x)
    
    # Calculate KL divergences
    kld_a = kldivergence(hist_a, theoretical_pdf)
    kld_b = kldivergence(hist_b, theoretical_pdf)
    
    mean_a = mean(samples_a)
    mean_b = mean(samples_b)
    
    if verbose
        println("Proposal Invariance Test:")
        println("  Target mean: 0.5")
        println("  Narrow proposal mean: $(mean_a)")
        println("  Wide proposal mean: $(mean_b)")
        println("  KL divergence (narrow): $(kld_a)")
        println("  KL divergence (wide): $(kld_b)")
        println("  Narrow acceptance: $(acceptance_rate(alg_a))")
        println("  Wide acceptance: $(acceptance_rate(alg_b))")
    end
    
    pass = true
    pass &= abs(mean_a - 0.5) < 0.1
    pass &= abs(mean_b - 0.5) < 0.1
    # Both should converge to same target distribution
    pass &= kld_a < 0.5
    pass &= kld_b < 0.5
    # Different acceptance rates but same target distribution
    pass &= abs(acceptance_rate(alg_a) - acceptance_rate(alg_b)) > 0.1
    pass &= length(samples_a) == samples && length(samples_b) == samples
    
    return pass
end

"""
Test BoltzmannLogWeight call overloads and Metropolis convenience constructor.
"""
function test_metropolis_boltzmann_overloads_and_constructor(; verbose=false)
    rng = MersenneTwister(500)
    β = 0.75

    lw = BoltzmannLogWeight(β)

    # Scalar overload on Real (including integer inputs)
    e_int = 4
    e_real = 2.5

    # AbstractArray overload
    e_vec = [1, -2, 3]
    e_mat = [1.0 2.0; -3.0 4.0]

    pass = true
    pass &= lw(e_int) == -β * e_int
    pass &= lw(e_real) == -β * e_real
    pass &= lw(e_vec) == -β * sum(e_vec)
    pass &= lw(e_mat) == -β * sum(e_mat)

    # Metropolis convenience constructor with β keyword
    alg = Metropolis(rng; β=β)

    pass &= alg.rng === rng
    pass &= alg.logweight isa BoltzmannLogWeight
    pass &= alg.steps == 0
    pass &= alg.accepted == 0
    pass &= alg.logweight(e_int) == -β * e_int
    pass &= alg.logweight(e_vec) == -β * sum(e_vec)

    if verbose
        println("Boltzmann Overloads + Constructor Test:")
        println("  β: $(β)")
        println("  lw(Int): $(lw(e_int))")
        println("  lw(Real): $(lw(e_real))")
        println("  lw(Vector): $(lw(e_vec))")
        println("  lw(Matrix): $(lw(e_mat))")
        println("  Constructor logweight type: $(typeof(alg.logweight))")
    end

    return pass
end

"""
Test Glauber basics.
"""
function test_glauber_basics(; verbose=false)
    rng = MersenneTwister(777)
    β = 0.8

    glauber = Glauber(rng; β=β)
    pass = true

    pass &= glauber.rng === rng
    pass &= glauber.steps == 0
    pass &= glauber.accepted == 0

    # Strongly favorable move should almost always be accepted
    accepted = 0
    for _ in 1:1_000
        accepted += accept!(glauber, 5.0)
    end
    pass &= accepted > 980
    pass &= glauber.steps == 1_000
    pass &= 0.0 <= acceptance_rate(glauber) <= 1.0

    if verbose
        println("Glauber basics:")
        println("  Glauber acceptance rate (log_ratio=5): $(acceptance_rate(glauber))")
    end

    return pass
end

"""
Test HeatBath basics.
"""
function test_heatbath_basics(; verbose=false)
    rng = MersenneTwister(778)
    β = 0.8
    pass = true

    # HeatBath constructor and counters
    hb = HeatBath(rng; β=β)
    pass &= hb.rng === rng
    pass &= hb.β == β
    pass &= hb.steps == 0

    if verbose
        println("HeatBath basics:")
        println("  HeatBath β: $(hb.β)")
    end

    return pass
end

function run_metropolis_testsets(; verbose=false)
    @testset "Metropolis" begin
        @testset "1D Gaussian" begin
            @test test_metropolis_1d_gaussian(verbose=verbose)
        end
        @testset "2D Gaussian" begin
            @test test_metropolis_2d_gaussian(verbose=verbose)
        end
        @testset "Acceptance tracking" begin
            @test test_metropolis_acceptance_tracking(verbose=verbose)
        end
        @testset "Temperature effects" begin
            @test test_metropolis_temperature_effects(verbose=verbose)
        end
        @testset "Proposal invariance" begin
            @test test_metropolis_proposal_invariance(verbose=verbose)
        end
        @testset "Boltzmann overloads and constructor" begin
            @test test_metropolis_boltzmann_overloads_and_constructor(verbose=verbose)
        end
        @testset "Glauber basics" begin
            @test test_glauber_basics(verbose=verbose)
        end
        @testset "HeatBath basics" begin
            @test test_heatbath_basics(verbose=verbose)
        end
    end
    return true
end
