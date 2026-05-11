using Random
using Statistics
using Printf
using MonteCarloX
using MCXSoftMatter

# CLI: julia --project=. benchmark.jl [updates_global] [updates_equi] [repeats] [--kernels]
const NUMERIC_ARGS = [arg for arg in ARGS if !startswith(arg, "--")]
const RUN_KERNELS = "--kernels" in ARGS
const UPDATES_GLOBAL = length(NUMERIC_ARGS) >= 1 ? parse(Int, NUMERIC_ARGS[1]) : 500_000
const UPDATES_EQUI = length(NUMERIC_ARGS) >= 2 ? parse(Int, NUMERIC_ARGS[2]) : 5_000
const REPEATS = length(NUMERIC_ARGS) >= 3 ? parse(Int, NUMERIC_ARGS[3]) : 10
const CHAIN_RELAX_STEPS = 10_000

@inline make_rng(seed::Integer) = Xoshiro(seed)

@inline function run_updates!(sys, alg, move!::Function, nsteps::Int; chain::Bool=false)
    if chain
        for _ in 1:nsteps
            move!(sys, alg; chain=true)
        end
    else
        for _ in 1:nsteps
            move!(sys, alg)
        end
    end
    return nothing
end

function setup_particle_gas(; N::Int, L::Float64)
    lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
    sys = ParticleGas(; D=3, N=N, L=L, pair_potential=lj)
    return sys
end

function setup_bead_spring(; num_poly::Int, length_poly::Int, L::Float64,
                           delta::Float64, with_bending::Bool=false)
    lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
    fene = FENEPotential(spring_constant=30.0, l0=0.0, l_max=1.5)
    if with_bending
        bend = CosineBendingPotential(3.0)
        return BeadSpringPolymer(; D=3, num_poly=num_poly, length_poly=length_poly, L=L,
            pair_potential=lj, bond_potential=fene, bending_potential=bend), delta
    end
    return BeadSpringPolymer(; D=3, num_poly=num_poly, length_poly=length_poly, L=L,
        pair_potential=lj, bond_potential=fene), delta
end

function benchmark_case(case::AbstractString, mode::AbstractString,
                        seed::Integer, β::Real, setup_sys::Function, move!::Function;
                        init_type::Symbol, chain::Bool=false, pre_relax_chain::Bool=false)
    rng_sys = make_rng(seed)
    setup = setup_sys()
    sys = setup isa Tuple ? setup[1] : setup
    Δ = setup isa Tuple ? setup[2] : nothing
    init!(sys, init_type; rng=rng_sys)
    alg = Metropolis(make_rng(seed + 1); β=β)

    if pre_relax_chain
        # Chain translations preserve internal geometry; short monomer relaxation helps
        # remove pathological random-walk overlaps before timing rigid-chain moves.
        run_updates!(sys, alg, (s, a; chain=false) -> translate!(s, a, Δ; chain=chain), CHAIN_RELAX_STEPS; chain=false)
    end

    run_updates!(sys, alg, move!, UPDATES_EQUI; chain=chain)

    cpu_ms_runs = Vector{Float64}(undef, REPEATS)
    ns_per_update_runs = Vector{Float64}(undef, REPEATS)
    updates_runs = Vector{Int}(undef, REPEATS)
    for r in 1:REPEATS
        reset!(alg)
        t0 = time_ns()
        run_updates!(sys, alg, move!, UPDATES_GLOBAL; chain=chain)
        t1 = time_ns()

        elapsed_ns = t1 - t0
        steps_alg = steps(alg)
        steps_alg >= 0 || throw(ArgumentError("Benchmark case $(case) / $(mode) has zero proposals (e.g. due to geometric rejection)"))

        cpu_ms_runs[r] = elapsed_ns / 1.0e6
        ns_per_update_runs[r] = steps_alg > 0 ? elapsed_ns / steps_alg : NaN
        updates_runs[r] = steps_alg
    end

    cpu_ms = median(cpu_ms_runs)
    ns_per_update = median(ns_per_update_runs)
    nupdates = round(Int, median(updates_runs))
        geom_acc = nupdates / UPDATES_GLOBAL
        mc_acc = acceptance_rate(alg)
    final_E = energy(sys)

        @printf("%-44s %-18s %-7s %8d %6.2f %10.3f %12.3f %10d %9.4f %9.4f %12.3f\n",
            case, mode, "xoshiro", seed, Float64(β), cpu_ms, ns_per_update,
            nupdates, geom_acc, mc_acc, final_E)
    return nothing
end

function run_kernel_microbenchmarks()
    println()
    println("Kernel microbenchmarks")
    println("Configuration: loops=20000")
    @printf("%-44s %12s\n", "kernel", "ns/call")
    @printf("%-44s %12s\n", "--------------------------------------------", "------------")

    loops = 20_000

    gas = setup_particle_gas(; N=256, L=32.0)
    init!(gas, :random; rng=make_rng(4001))
    t0 = time_ns()
    for rep in 1:loops
        i = 1 + ((rep - 1) % gas.N)
        MCXSoftMatter._energy_of_particle(gas, i)
    end
    t1 = time_ns()
    @printf("%-44s %12.3f\n", "ParticleGas local energy", (t1 - t0) / loops)

    poly = first(setup_bead_spring(; num_poly=6, length_poly=12, L=20.0, delta=0.08))
    init!(poly, :random_walk; rng=make_rng(4002))
    n_total = length(poly.positions)
    t0 = time_ns()
    for rep in 1:loops
        idx = 1 + ((rep - 1) % n_total)
        MCXSoftMatter._monomer_energy(poly, idx)
    end
    t1 = time_ns()
    @printf("%-44s %12.3f\n", "BeadSpring monomer local energy", (t1 - t0) / loops)

    t0 = time_ns()
    for rep in 1:loops
        n = 1 + ((rep - 1) % poly.num_poly)
        start_idx = poly.offsets[n] + 1
        M   = poly.lengths[n]
        acc = 0.0
        for k in 0:M-1
            acc += MCXSoftMatter._pair_energy_of_all(poly, start_idx + k)
        end
        acc
    end
    t1 = time_ns()
    @printf("%-44s %12.3f\n", "BeadSpring chain pair-only sweep", (t1 - t0) / loops)

    return nothing
end

println("MCXSoftMatter benchmark")
println("Configuration: updates_equi=$(UPDATES_EQUI), updates_global=$(UPDATES_GLOBAL), repeats=$(REPEATS)")
println()
@printf("%-44s %-18s %-7s %8s %6s %10s %12s %10s %9s %9s %12s\n",
    "case", "mode", "rng", "seed", "beta", "cpu_ms", "ns/update", "updates", "geom.acc", "MC acc", "final_E")
@printf("%-44s %-18s %-7s %8s %6s %10s %12s %10s %9s %9s %12s\n",
    "--------------------------------------------", "------------------", "-------", "--------", "------", "----------", "------------", "----------", "---------", "---------", "------------")

cases = [
    (name="ParticleGas(D=3,N=256,LJ)", mode="translate!", seed=2030, β=1.0,
    setup_fn=() -> setup_particle_gas(; N=256, L=32.0),
    move_fn=((sys, alg) -> translate!(sys, alg, 0.20)), init_type=:random, chain=false, pre_relax_chain=false),
    (name="BeadSpring(D=3,6x12,LJ+FENE)", mode="translate!", seed=2031, β=1.0,
     setup_fn=() -> setup_bead_spring(; num_poly=6, length_poly=12, L=20.0, delta=0.08),
        move_fn=((sys, alg; chain=false) -> translate!(sys, alg, 0.08; chain=chain)), init_type=:random_walk, chain=false, pre_relax_chain=false),
    (name="BeadSpring(D=3,6x12,LJ+FENE)", mode="translate! (chain)", seed=2032, β=1.0,
     setup_fn=() -> setup_bead_spring(; num_poly=6, length_poly=12, L=20.0, delta=0.05),
        move_fn=((sys, alg; chain=false) -> translate!(sys, alg, 0.05; chain=chain)), init_type=:random_walk, chain=true, pre_relax_chain=true),
    (name="BeadSpring(D=3,6x12,LJ+FENE+bending)", mode="translate!", seed=2033, β=1.0,
     setup_fn=() -> setup_bead_spring(; num_poly=6, length_poly=12, L=20.0, delta=0.08, with_bending=true),
        move_fn=((sys, alg; chain=false) -> translate!(sys, alg, 0.08; chain=chain)), init_type=:random_walk, chain=false, pre_relax_chain=false),
]

for case in cases
    benchmark_case(case.name, case.mode, case.seed, case.β, case.setup_fn, case.move_fn;
        init_type=case.init_type, chain=case.chain, pre_relax_chain=case.pre_relax_chain)
end

RUN_KERNELS && run_kernel_microbenchmarks()
