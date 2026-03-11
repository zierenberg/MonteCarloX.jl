using MonteCarloX
using Test

struct BadEnsemble <: AbstractEnsemble end

"""
Test AbstractEnsemble constructor with various valid cases and error conditions.
"""
function test_abstract_ensemble_constructor(; verbose=false)
    # test that failures come when trying to constructing a new ensemble type that doesn't implement relevant functions
    ens = BadEnsemble()
    pass = true

    # logweight(ens)
    pass &= try
        logweight(ens)
        false
    catch err
        if verbose; println("✓ logweight(ens) throws error for BadEnsemble"); end
        err isa ArgumentError
    end
    
    # loweight(ens, x)
    pass &= try
        logweight(ens, 1.0)
        false
    catch err
        if verbose; println("✓ logweight(ens, 1.0) throws error for BadEnsemble"); end
        err isa ArgumentError
    end

    # update!(ens)
    pass &= try
        update!(ens)
        false
    catch err
        if verbose; println("✓ update!(ens) throws error for BadEnsemble"); end
        err isa ArgumentError
    end

    return pass
end


"""
Test BoltzmannEnsemble constructor with various valid cases and error conditions.
"""
function test_boltzmann_ensemble_constructor(; verbose=false)
    pass = true

    # Test 1: Constructor with beta keyword
    ens1 = BoltzmannEnsemble(beta=2.0)
    pass &= ens1.beta == 2.0
    if verbose; println("✓ BoltzmannEnsemble(beta=2.0)"); end

    # Test 2: Constructor with β keyword (Greek letter)
    ens2 = BoltzmannEnsemble(β=3.0)
    pass &= ens2.beta == 3.0
    if verbose; println("✓ BoltzmannEnsemble(β=3.0)"); end

    # Test 3: Constructor with T keyword (inverse temperature)
    ens3 = BoltzmannEnsemble(T=1.0)
    pass &= isapprox(ens3.beta, 1.0)
    if verbose; println("✓ BoltzmannEnsemble(T=1.0)"); end

    # Test 4: Constructor with T=0.5 should give beta=2.0
    ens4 = BoltzmannEnsemble(T=0.5)
    pass &= isapprox(ens4.beta, 2.0)
    if verbose; println("✓ BoltzmannEnsemble(T=0.5) => beta=2.0"); end

    # Test 5: Positional constructor
    ens5 = BoltzmannEnsemble(1.5)
    pass &= ens5.beta == 1.5
    if verbose; println("✓ BoltzmannEnsemble(1.5)"); end

    # Error Test 1: Specifying both beta and β should throw
    error1 = false
    try
        BoltzmannEnsemble(beta=2.0, β=3.0)
    catch err
        error1 = err isa ArgumentError && contains(err.msg, "Specify only one of")
    end
    pass &= error1
    if verbose; println("✓ Error when specifying both beta and β"); end

    # Error Test 2: Specifying beta and T should throw
    error2 = false
    try
        BoltzmannEnsemble(beta=2.0, T=1.0)
    catch err
        error2 = err isa ArgumentError && contains(err.msg, "Specify exactly one")
    end
    pass &= error2
    if verbose; println("✓ Error when specifying both beta and T"); end

    # Error Test 3: Specifying β and T should throw
    error3 = false
    try
        BoltzmannEnsemble(β=2.0, T=1.0)
    catch err
        error3 = err isa ArgumentError && contains(err.msg, "Specify exactly one")
    end
    pass &= error3
    if verbose; println("✓ Error when specifying both β and T"); end

    # Error Test 4: Specifying neither beta/β nor T should throw
    error4 = false
    try
        BoltzmannEnsemble()
    catch err
        error4 = err isa ArgumentError && contains(err.msg, "Specify exactly one")
    end
    pass &= error4
    if verbose; println("✓ Error when specifying neither beta nor T"); end

    # Test logweight methods
    ens = BoltzmannEnsemble(β=2.0)
    lw = logweight(ens)
    pass &= lw(1.0) == -2.0
    pass &= lw([1.0, 2.0, 3.0]) == -12.0  # -2.0 * sum([1,2,3])
    if verbose; println("✓ logweight methods work correctly"); end

    if verbose
        println("BoltzmannEnsemble constructor test: $(pass)")
    end

    return pass
end

"""
Test MulticanonicalEnsemble constructor with various valid cases and error conditions.
"""
function test_multicanonical_ensemble_constructor(; verbose=false)
    pass = true

    # Test 1: Constructor from BinnedObject (logweight only)
    bins = 0.0:1.0:4.0
    lw = BinnedObject(bins, 0.0)
    ens1 = MulticanonicalEnsemble(lw)
    pass &= ens1.logweight === lw
    pass &= ens1.histogram isa BinnedObject
    pass &= all(iszero, ens1.histogram.weights)
    if verbose; println("✓ MulticanonicalEnsemble from BinnedObject"); end

    # Test 2: Constructor with explicit histogram
    hist = BinnedObject(bins, 0.0)
    ens2 = MulticanonicalEnsemble(lw, hist)
    pass &= ens2.logweight === lw
    pass &= ens2.histogram === hist
    if verbose; println("✓ MulticanonicalEnsemble with explicit histogram"); end

    # Test 3: Constructor with record_visits flag
    ens3 = MulticanonicalEnsemble(lw; record_visits=false)
    pass &= ens3.record_visits == false
    if verbose; println("✓ MulticanonicalEnsemble with record_visits=false"); end

    # Test 4: Constructor from bins (creates BinnedObject internally)
    ens4 = MulticanonicalEnsemble(bins)
    pass &= ens4.logweight isa BinnedObject
    pass &= all(iszero, ens4.logweight.weights)
    if verbose; println("✓ MulticanonicalEnsemble from bins"); end

    # Test 5: Constructor from bins with init value
    ens5 = MulticanonicalEnsemble(bins; init=-1.0)
    pass &= all(w -> isapprox(w, -1.0), ens5.logweight.weights)
    if verbose; println("✓ MulticanonicalEnsemble from bins with init"); end

    # Test 6: Default record_visits should be true
    ens6 = MulticanonicalEnsemble(lw)
    pass &= ens6.record_visits == true
    if verbose; println("✓ MulticanonicalEnsemble default record_visits=true"); end

    # Test 7: Mismatched domains should throw
    other_bins = 0.0:1.0:5.0
    other_lw = BinnedObject(other_bins, 0.0)
    error1 = false
    try
        MulticanonicalEnsemble(lw, other_lw)
    catch err
        error1 = err isa AssertionError
    end
    pass &= error1
    if verbose; println("✓ Error when histogram domain doesn't match logweight"); end

    # Test 8: logweight methods
    ens = MulticanonicalEnsemble(bins, init=-2.0)
    lw_ens = logweight(ens)
    pass &= lw_ens isa BinnedObject
    pass &= lw_ens(2.0) === ens.logweight(2.0)
    pass &= lw_ens(2.0) === logweight(ens, 2.0)
    if verbose; println("✓ logweight methods work correctly"); end

    # Test 9: get_centers and get_values should be called on logweight, not ensemble
    ens = MulticanonicalEnsemble(bins, init=-1.5)
    lw = logweight(ens)
    centers = get_centers(lw)
    pass &= centers isa Vector
    pass &= length(centers) == 4  # BinnedObject has 4 bins for range 0:1:4
    values = get_values(lw)
    pass &= all(v -> isapprox(v, -1.5), values)
    if verbose; println("✓ get_centers and get_values work via logweight"); end

    # Test 10: record_visits field
    ens_record = MulticanonicalEnsemble(lw; record_visits=true)
    ens_no_record = MulticanonicalEnsemble(lw; record_visits=false)
    pass &= ens_record.record_visits == true
    pass &= ens_no_record.record_visits == false
    if verbose; println("✓ record_visits field works correctly"); end

    if verbose
        println("MulticanonicalEnsemble constructor test: $(pass)")
    end

    return pass
end

"""
Test WangLandauEnsemble constructor with various valid cases and error conditions.
"""
function test_wang_landau_ensemble_constructor(; verbose=false)
    pass = true

    # Test 1: Constructor from BinnedObject (logweight only)
    bins = 0.0:1.0:4.0
    lw = BinnedObject(bins, 0.0)
    ens1 = WangLandauEnsemble(lw)
    pass &= ens1.logweight === lw
    pass &= isapprox(ens1.logf, 1.0)
    if verbose; println("✓ WangLandauEnsemble from BinnedObject"); end

    # Test 2: Constructor with custom logf parameter
    ens2 = WangLandauEnsemble(lw; logf=0.5)
    pass &= isapprox(ens2.logf, 0.5)
    if verbose; println("✓ WangLandauEnsemble with custom logf"); end

    # Test 3: Constructor from bins (creates BinnedObject internally)
    ens3 = WangLandauEnsemble(bins)
    pass &= ens3.logweight isa BinnedObject
    pass &= all(iszero, ens3.logweight.weights)
    if verbose; println("✓ WangLandauEnsemble from bins"); end

    # Test 4: Constructor from bins with init value
    ens4 = WangLandauEnsemble(bins; init=-1.0)
    pass &= all(w -> isapprox(w, -1.0), ens4.logweight.weights)
    if verbose; println("✓ WangLandauEnsemble from bins with init"); end

    # Test 5: Constructor from bins with custom logf
    ens5 = WangLandauEnsemble(bins; logf=0.3)
    pass &= isapprox(ens5.logf, 0.3)
    if verbose; println("✓ WangLandauEnsemble from bins with init and logf"); end

    # Test 6: logweight methods
    ens = WangLandauEnsemble(bins, init=-2.0)
    lw_ens = logweight(ens)
    pass &= lw_ens isa BinnedObject
    pass &= lw_ens(2.0) === ens.logweight(2.0)
    pass &= lw_ens(2.0) === logweight(ens, 2.0)
    if verbose; println("✓ logweight methods work correctly"); end

    # Test 7: get_centers and get_values should be called on logweight, not ensemble
    ens = WangLandauEnsemble(bins, init=-1.5)
    lw = logweight(ens)
    centers = get_centers(lw)
    pass &= centers isa Vector
    pass &= length(centers) == 4  # BinnedObject has 4 bins for range 0:1:4
    values = get_values(lw)
    pass &= all(v -> isapprox(v, -1.5), values)
    if verbose; println("✓ get_centers and get_values work via logweight"); end


    if verbose
        println("WangLandauEnsemble constructor test: $(pass)")
    end

    return pass
end

"""
Test FunctionEnsemble constructor and behavior.
"""
function test_function_ensemble_constructor(; verbose=false)
    pass = true

    # Test 1: Constructor with a simple function
    f(x) = -x^2
    ens1 = FunctionEnsemble(f)
    pass &= logweight(ens1, 2.0) == -4.0
    if verbose; println("✓ FunctionEnsemble with simple function"); end

    # Test 2: Constructor with lambda
    ens2 = FunctionEnsemble(x -> -exp(x))
    pass &= isapprox(logweight(ens2, 0.0), -1.0)
    if verbose; println("✓ FunctionEnsemble with lambda"); end

    # Test 3: logweight method returns closure
    lw = logweight(ens1)
    pass &= lw isa Function
    pass &= lw(3.0) == -9.0
    if verbose; println("✓ logweight method returns callable"); end

    if verbose
        println("FunctionEnsemble constructor test: $(pass)")
    end

    return pass
end

function run_ensemble_testsets(;verbose=false)
    @testset "Ensemble Constructors" begin
        @test test_abstract_ensemble_constructor(verbose=verbose)
        @test test_boltzmann_ensemble_constructor(verbose=verbose)
        @test test_multicanonical_ensemble_constructor(verbose=verbose)
        @test test_wang_landau_ensemble_constructor(verbose=verbose)
        @test test_function_ensemble_constructor(verbose=verbose)
    end
end
