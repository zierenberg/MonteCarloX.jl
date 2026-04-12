using MonteCarloX
using Random
using StatsBase
using Test
using Serialization

function _tmp_checkpoint_file(prefix="mcx_")
    joinpath(tempdir(), "$(prefix)_$(abs(rand(Int))).ckpt")
end

function test_checkpoint_restore_importance_sampling()
    pass = true

    for (name, alg) in [
        ("Metropolis", Metropolis(MersenneTwister(1), x -> -0.5x^2)),
        ("Glauber",    Glauber(MersenneTwister(2), x -> -0.5x^2)),
        ("HeatBath",   HeatBath(MersenneTwister(3); β=0.5)),
    ]
        alg.steps = 100
        hasproperty(alg, :accepted) && (alg.accepted = 42)
        next_rand = rand(copy(alg.rng))

        file = _tmp_checkpoint_file(name)
        ckpt = init_checkpoint(file, (alg=alg,); sweep=100)
        pass &= check(isfile(file), "$name: checkpoint file created\n")

        state = restore(file)
        pass &= check(state.sweep == 100, "$name: metadata restored\n")
        pass &= check(rand(state.alg.rng) == next_rand, "$name: rng restored\n")

        finalize!(ckpt)
        pass &= check(!isfile(file), "$name: finalize! cleans up\n")
    end

    return pass
end

function test_checkpoint_restore_ensembles()
    pass = true

    # Multicanonical ensemble: logweight + histogram
    alg_muca = Multicanonical(MersenneTwister(2), 0:10)
    ens_muca = ensemble(alg_muca)
    ens_muca.logweight.values .= 0:10
    ens_muca.histogram.values .= 2 .* (0:10)

    file = _tmp_checkpoint_file("ens")
    ckpt = init_checkpoint(file, (alg=alg_muca,); sweep=0)
    state = restore(file)
    pass &= check(ensemble(state.alg) == ens_muca, "muca ensemble restored\n")
    finalize!(ckpt)

    # WangLandau ensemble: logweight + logf
    alg_wl = WangLandau(MersenneTwister(3), 0:5; logf=2.0)
    ens_wl = ensemble(alg_wl)
    ens_wl.logweight.values .= [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    ens_wl.logf = 1.5

    file = _tmp_checkpoint_file("ens")
    ckpt = init_checkpoint(file, (alg=alg_wl,); sweep=0)
    state = restore(file)
    pass &= check(ensemble(state.alg) == ens_wl, "wl ensemble restored\n")
    finalize!(ckpt)

    return pass
end

function test_checkpoint_restore_kmc()
    alg = Gillespie(MersenneTwister(5))
    alg.steps = 42
    alg.time  = 3.14

    file = _tmp_checkpoint_file("kmc")
    ckpt = init_checkpoint(file, (alg=alg,); step=42)
    state = restore(file)

    pass = check(state.alg == alg, "KMC algorithm restored\n")

    finalize!(ckpt)
    return pass
end

function test_relink()
    rng = MersenneTwister(7)
    alg = Metropolis(rng, x -> -x^2)

    file = _tmp_checkpoint_file("relink")
    ckpt = init_checkpoint(file, (alg=alg,); sweep=0)

    # simulate restore + relink
    state = restore(file)
    alg2 = state.alg
    relink!(ckpt, (alg=alg2,))
    checkpoint!(ckpt; sweep=1)

    state2 = restore(file)
    pass = check(state2.sweep == 1, "relink + checkpoint wrote updated sweep\n")

    finalize!(ckpt)
    return pass
end

function test_deterministic_restart()
    # Reference: single uninterrupted run of 2000 steps
    rng_ref = MersenneTwister(42)
    alg_ref = Metropolis(rng_ref, x -> -0.5x^2)
    x_ref   = 0.0
    step    = 1.0
    samples_ref = Float64[]
    for i in 1:2000
        x_new = x_ref + randn(alg_ref.rng) * step
        x_ref = accept!(alg_ref, x_new, x_ref) ? x_new : x_ref
        i > 1000 && push!(samples_ref, x_ref)
    end

    # Checkpointed run: 1000 steps -> checkpoint -> restore -> 1000 more
    rng_chk = MersenneTwister(42)
    alg_chk = Metropolis(rng_chk, x -> -0.5x^2)
    x_chk   = 0.0
    for _ in 1:1000
        x_new = x_chk + randn(alg_chk.rng) * step
        x_chk = accept!(alg_chk, x_new, x_chk) ? x_new : x_chk
    end

    file = _tmp_checkpoint_file("det")
    # Checkpoint the RNG separately so we can reconstruct the algorithm
    # with a fresh callable (anonymous functions cannot survive serialize
    # round-trips due to world-age issues).
    ckpt = init_checkpoint(file, (rng=copy(alg_chk.rng), x=x_chk,
                                   steps=alg_chk.steps, accepted=alg_chk.accepted); sweep=1000)

    state = restore(file)
    alg_chk = Metropolis(state.rng, x -> -0.5x^2)
    alg_chk.steps    = state.steps
    alg_chk.accepted = state.accepted
    x_chk   = state.x

    samples_chk = Float64[]
    for _ in 1:1000
        x_new = x_chk + randn(alg_chk.rng) * step
        x_chk = accept!(alg_chk, x_new, x_chk) ? x_new : x_chk
        push!(samples_chk, x_chk)
    end

    pass = samples_ref == samples_chk
    check(pass, "deterministic restart produces identical trajectory\n")

    finalize!(ckpt)
    return pass
end

function test_metadata_conflict()
    alg = Metropolis(MersenneTwister(1), x -> -x^2)
    file = _tmp_checkpoint_file("conflict")
    ckpt = init_checkpoint(file, (alg=alg,); sweep=0)

    pass = false
    try
        checkpoint!(ckpt; alg="conflict")
    catch e
        pass = e isa ArgumentError && occursin("conflicts", e.msg)
    end
    check(pass, "metadata key conflict detected\n")

    finalize!(ckpt)
    return pass
end

@testset "Checkpointing: checkpoint / restore" begin
    @test test_checkpoint_restore_importance_sampling()
    @test test_checkpoint_restore_ensembles()
    @test test_checkpoint_restore_kmc()
    @test test_relink()
    @test test_deterministic_restart()
    @test test_metadata_conflict()
end
