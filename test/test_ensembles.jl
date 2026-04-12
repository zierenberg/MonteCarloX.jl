using MonteCarloX
using Test

struct BadEnsemble <: AbstractEnsemble end

function test_abstract_ensemble_constructor()
    ens = BadEnsemble()
    pass = true

    threw = try; logweight(ens); false; catch err; err isa ArgumentError; end
    pass &= check(threw, "logweight(BadEnsemble) throws\n")

    threw = try; logweight(ens, 1.0); false; catch err; err isa ArgumentError; end
    pass &= check(threw, "logweight(BadEnsemble, x) throws\n")

    threw = try; update!(ens); false; catch err; err isa ArgumentError; end
    pass &= check(threw, "update!(BadEnsemble) throws\n")

    return pass
end

function test_boltzmann_ensemble_constructor()
    pass = true

    ens1 = BoltzmannEnsemble(beta=2.0)
    pass &= check(ens1.beta == 2.0, "BoltzmannEnsemble(beta=2.0)\n")

    ens2 = BoltzmannEnsemble(β=3.0)
    pass &= check(ens2.beta == 3.0, "BoltzmannEnsemble(beta=3.0)\n")

    ens3 = BoltzmannEnsemble(T=1.0)
    pass &= check(isapprox(ens3.beta, 1.0), "BoltzmannEnsemble(T=1.0)\n")

    # error: both beta and beta
    threw = try; BoltzmannEnsemble(beta=2.0, β=3.0); false
    catch err; err isa ArgumentError && contains(err.msg, "Specify only one of"); end
    pass &= check(threw, "error: both beta and beta\n")

    threw = try; BoltzmannEnsemble(beta=2.0, T=1.0); false
    catch err; err isa ArgumentError && contains(err.msg, "Specify exactly one"); end
    pass &= check(threw, "error: beta and T\n")

    threw = try; BoltzmannEnsemble(β=2.0, T=1.0); false
    catch err; err isa ArgumentError && contains(err.msg, "Specify exactly one"); end
    pass &= check(threw, "error: beta and T\n")

    threw = try; BoltzmannEnsemble(); false
    catch err; err isa ArgumentError && contains(err.msg, "Specify exactly one"); end
    pass &= check(threw, "error: no arguments\n")

    # logweight methods
    lw = logweight(ens1)
    pass &= check(lw(1.0) == -2.0, "logweight scalar\n")
    pass &= check(lw([1.0, 2.0, 3.0]) == -12.0, "logweight vector\n")

    return pass
end

function test_multicanonical_ensemble_constructor()
    pass = true

    bins = 0.0:1.0:4.0
    lw = BinnedObject(bins, 0.0)
    ens1 = MulticanonicalEnsemble(lw)
    pass &= check(ens1.logweight === lw, "from BinnedObject\n")
    pass &= check(ens1.histogram isa BinnedObject, "histogram created\n")
    pass &= check(all(iszero, ens1.histogram.values), "histogram zeroed\n")

    hist = BinnedObject(bins, 0.0)
    ens2 = MulticanonicalEnsemble(lw, hist)
    pass &= check(ens2.logweight === lw, "explicit logweight\n")
    pass &= check(ens2.histogram === hist, "explicit histogram\n")

    ens3 = MulticanonicalEnsemble(bins; init=-1.0, record_visits=false)
    pass &= check(ens3.record_visits == false, "record_visits=false\n")
    pass &= check(ens3.logweight isa BinnedObject, "from bins\n")
    pass &= check(all(w -> isapprox(w, -1.0), ens3.logweight.values), "from bins with init\n")

    # mismatched domains
    other_lw = BinnedObject(0.0:1.0:5.0, 0.0)
    threw = try; MulticanonicalEnsemble(lw, other_lw); false; catch err; err isa AssertionError; end
    pass &= check(threw, "mismatched domains throw\n")

    # logweight methods
    ens = MulticanonicalEnsemble(bins, init=-2.0)
    pass &= check(logweight(ens)(2.0) === logweight(ens, 2.0), "logweight(ens, x) consistent\n")

    return pass
end

function test_wang_landau_ensemble_constructor()
    pass = true

    bins = 0.0:1.0:4.0
    lw = BinnedObject(bins, 0.0)
    ens1 = WangLandauEnsemble(lw)
    pass &= check(ens1.logweight === lw, "from BinnedObject\n")
    pass &= check(isapprox(ens1.logf, 1.0), "default logf=1.0\n")

    ens2 = WangLandauEnsemble(lw; logf=0.5)
    pass &= check(isapprox(ens2.logf, 0.5), "custom logf\n")

    ens3 = WangLandauEnsemble(bins)
    pass &= check(ens3.logweight isa BinnedObject, "from bins\n")
    pass &= check(all(iszero, ens3.logweight.values), "logweight zeroed\n")

    ens4 = WangLandauEnsemble(bins; init=-1.0)
    pass &= check(all(w -> isapprox(w, -1.0), ens4.logweight.values), "from bins with init\n")

    # logweight methods
    ens = WangLandauEnsemble(bins, init=-2.0)
    pass &= check(logweight(ens)(2.0) === logweight(ens, 2.0), "logweight(ens, x) consistent\n")

    return pass
end

function test_function_ensemble_constructor()
    pass = true

    f(x) = -x^2
    ens1 = FunctionEnsemble(f)
    pass &= check(logweight(ens1, 2.0) == -4.0, "FunctionEnsemble with named function\n")

    ens2 = FunctionEnsemble(x -> -exp(x))
    pass &= check(isapprox(logweight(ens2, 0.0), -1.0), "FunctionEnsemble with lambda\n")

    lw = logweight(ens1)
    pass &= check(lw isa Function, "logweight returns callable\n")
    pass &= check(lw(3.0) == -9.0, "logweight callable result\n")

    return pass
end

@testset "Ensemble Constructors" begin
    @test test_abstract_ensemble_constructor()
    @test test_boltzmann_ensemble_constructor()
    @test test_multicanonical_ensemble_constructor()
    @test test_wang_landau_ensemble_constructor()
    @test test_function_ensemble_constructor()
end
