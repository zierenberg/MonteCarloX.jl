using MonteCarloX
using StatsBase
using Test

# source == target: no reweighting, uniform weights.
function test_reweight_identity()
    pass = true
    args = Float64[1, 2, 3, 4, 5]
    obs = Float64[10, 20, 30, 40, 50]
    src = FunctionEnsemble(x -> -0.5x)

    iw = reweight(args, src, src)
    w = weights(iw)

    pass &= check(all(x -> isapprox(x, 1 / length(args)), w), "identity: weights uniform\n")
    pass &= check(isapprox(ess(iw), length(args)), "identity: ess == N\n")
    pass &= check(isapprox(log_normalization(iw), 0.0; atol = 1e-12), "identity: log_normalization == 0\n")
    pass &= check(isapprox(mean(obs, w), sum(obs) / length(obs)), "identity: mean == plain mean\n")
    return pass
end

# Log-space implementation must match the naive linear computation on well-scaled inputs.
function test_reweight_matches_naive()
    pass = true
    args = Float64[0.3, 1.1, -2.0, 5.0, 0.0]
    obs = Float64[1, 2, 3, 4, 5]
    logp_src(x) = -0.4x
    logp_tgt(x) = -0.7x + 0.2x^2

    g = [logp_tgt(a) - logp_src(a) for a in args]
    wlin = exp.(g)
    exp_lognorm = log(sum(wlin)) - log(length(args))
    exp_ess = sum(wlin)^2 / sum(abs2, wlin)
    wn = wlin ./ sum(wlin)
    exp_mean = sum(obs .* wn)

    iw = reweight(args, logp_src, logp_tgt)   # bare callables → coerced to FunctionEnsemble
    pass &= check(isapprox(log_normalization(iw), exp_lognorm), "naive: log_normalization\n")
    pass &= check(isapprox(ess(iw), exp_ess), "naive: ess\n")
    pass &= check(isapprox(collect(weights(iw)), wn), "naive: weights\n")
    pass &= check(isapprox(mean(obs, weights(iw)), exp_mean), "naive: weighted mean\n")
    return pass
end

# A large additive offset in the target overflows naive exp(g), but log-space is safe and the
# offset cancels everywhere except log_normalization (where it shifts by exactly the offset).
function test_reweight_overflow_safe()
    pass = true
    args = Float64[0.3, 1.1, -2.0, 5.0, 0.0]
    shift = 1000.0

    iw0 = reweight(args, x -> -0.4x, x -> -0.7x)
    iws = reweight(args, x -> -0.4x, x -> -0.7x + shift)

    pass &= check(all(isfinite, weights(iws)), "overflow: weights finite\n")
    pass &= check(isfinite(ess(iws)) && isfinite(log_normalization(iws)), "overflow: scalars finite\n")
    pass &= check(isapprox(collect(weights(iws)), collect(weights(iw0))), "overflow: weights unchanged by offset\n")
    pass &= check(isapprox(ess(iws), ess(iw0)), "overflow: ess unchanged by offset\n")
    pass &= check(isapprox(log_normalization(iws), log_normalization(iw0) + shift), "overflow: log_normalization shifts by offset\n")
    return pass
end

# Enumerate all microstates under uniform sampling and reweight to a Boltzmann target:
# mean(E, weights) must equal the exact canonical average Σ E e^{-βE} / Σ e^{-βE}.
function test_reweight_boltzmann_enumeration()
    pass = true
    # 2x2 Ising energy spectrum: degeneracies {-8:2, 0:12, 8:2} (16 microstates).
    energies = Float64[fill(-8.0, 2); fill(0.0, 12); fill(8.0, 2)]
    β = 0.3

    exact_num = sum(E * exp(-β * E) for E in energies)
    exact_den = sum(exp(-β * E) for E in energies)
    exact_mean = exact_num / exact_den

    iw = reweight(energies, x -> 0.0, BoltzmannEnsemble(β = β))   # uniform source → canonical target
    pass &= check(isapprox(mean(energies, weights(iw)), exact_mean), "boltzmann: canonical <E> exact\n")

    # log_normalization = log(Z_target/Z_source), Z_source = N (uniform, w=1 each).
    exact_lognorm = log(exact_den) - log(length(energies))
    pass &= check(isapprox(log_normalization(iw), exact_lognorm), "boltzmann: log_normalization exact\n")
    return pass
end

# ConstantEnsemble: constant log-weight, linear only at c == 0.
function test_constant_ensemble()
    pass = true
    pass &= check(logweight(ConstantEnsemble(2.0), 999.0) == 2.0, "constant: logweight is c\n")
    pass &= check(logweight(ConstantEnsemble(), 3.0) == 0.0, "constant: default c == 0\n")
    pass &= check(linear_logweight(ConstantEnsemble(0.0)) == true, "constant: c=0 is linear\n")
    pass &= check(linear_logweight(ConstantEnsemble(1.5)) == false, "constant: c≠0 not linear\n")
    return pass
end

# Default (flat) target strips the source weighting: wᵢ ∝ exp(-logweight(source, argᵢ)).
function test_reweight_flat_default()
    pass = true
    args = Float64[0.3, 1.1, -2.0, 5.0, 0.0]
    source(x) = -0.4x

    iw_default  = reweight(args, source)
    iw_constant = reweight(args, source, ConstantEnsemble())
    iw_explicit = reweight(args, source, x -> 0.0)

    pass &= check(isapprox(collect(weights(iw_default)), collect(weights(iw_constant))), "flat: default == ConstantEnsemble\n")
    pass &= check(isapprox(collect(weights(iw_default)), collect(weights(iw_explicit))), "flat: default == explicit flat\n")

    g = [-source(a) for a in args]
    wn = exp.(g) ./ sum(exp.(g))
    pass &= check(isapprox(collect(weights(iw_default)), wn), "flat: weights strip source bias\n")

    # A nonzero constant leaves the (self-normalized) weights untouched and shifts
    # log_normalization by exactly the constant.
    iwc = reweight(args, source, ConstantEnsemble(5.0))
    pass &= check(isapprox(collect(weights(iwc)), collect(weights(iw_default))), "flat: constant offset leaves weights unchanged\n")
    pass &= check(isapprox(log_normalization(iwc), log_normalization(iw_default) + 5.0), "flat: offset shifts log_normalization\n")
    return pass
end

# Binned log-density (log-DOS) → per-bin weights: replaces distribution_from_logdos.
function test_reweight_binned_logdos()
    pass = true

    logdos = BinnedObject(-1:1, 0.0)
    logdos.values .= log.([1.0, 2.0, 4.0])
    β = 0.5
    E = get_centers(logdos)
    P = weights(reweight(logdos, -β .* E))

    wlin = exp.(logdos.values .- β .* E)
    expected = wlin ./ sum(wlin)
    pass &= check(isapprox(sum(P), 1.0; atol = 1e-12), "binned: normalized\n")
    pass &= check(all(isapprox.(P, expected; atol = 1e-12)), "binned: values match naive\n")
    pass &= check(isapprox(mean(E, weights(reweight(logdos, -β .* E))), sum(E .* expected); atol = 1e-12), "binned: mean(E, P)\n")

    # NaN and -Inf mark empty/forbidden bins: zero weight, normalization over the rest.
    masked = BinnedObject(-1:2, 0.0)
    masked.values .= [0.0, NaN, -Inf, log(3.0)]
    Pm = weights(reweight(masked, zero(get_centers(masked))))
    pass &= check(Pm[2] == 0.0 && Pm[3] == 0.0, "binned: non-finite bins get zero weight\n")
    pass &= check(isapprox(sum(Pm), 1.0; atol = 1e-12), "binned: masked normalized\n")
    pass &= check(isapprox(Pm[1], 0.25; atol = 1e-12) && isapprox(Pm[4], 0.75; atol = 1e-12), "binned: masked values\n")

    threw = try
        bad = BinnedObject(0:2, NaN)
        reweight(bad, zeros(3))
        false
    catch err
        err isa ArgumentError
    end
    pass &= check(threw, "binned: all-masked throws ArgumentError\n")

    threw = try
        reweight(logdos, zeros(2))
        false
    catch err
        err isa DimensionMismatch
    end
    pass &= check(threw, "binned: length mismatch throws\n")

    return pass
end

@testset "Reweighting" begin
    @testset "identity (source == target)" begin
        @test test_reweight_identity()
    end
    @testset "matches naive linear" begin
        @test test_reweight_matches_naive()
    end
    @testset "overflow safe" begin
        @test test_reweight_overflow_safe()
    end
    @testset "boltzmann enumeration" begin
        @test test_reweight_boltzmann_enumeration()
    end
    @testset "constant ensemble" begin
        @test test_constant_ensemble()
    end
    @testset "flat default target" begin
        @test test_reweight_flat_default()
    end
    @testset "binned log-density" begin
        @test test_reweight_binned_logdos()
    end
end
