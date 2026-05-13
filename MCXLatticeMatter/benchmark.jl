using Random
using Statistics
using Printf
using MonteCarloX
using MCXLatticeMatter

# CLI: julia --project=. benchmark.jl [updates_global] [updates_equi] [repeats]
const UPDATES_GLOBAL = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 600_000
const UPDATES_EQUI = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 6_000
const REPEATS = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 10
const BETA = 0.5

@inline make_rng(seed::Integer) = Xoshiro(seed)

@inline function run_updates!(sys, alg, move!::Function, nsteps::Int)
    for _ in 1:nsteps
        move!(sys, alg)
    end
    return nothing
end

function setup_lattice(seed::Integer; dims::Vector{Int}, num_poly::Int, length_poly::Int)
    sys = LatticePolymer(; dims=dims, num_poly=num_poly, length_poly=length_poly,
        J_intra=0.5, J_inter=1.0)
    return sys
end

function benchmark_case(case::AbstractString, mode::AbstractString,
                        seed::Integer, β::Real, setup_sys::Function, move!::Function)
    rng_sys = make_rng(seed)
    sys = setup_sys(seed)
    init!(sys, :random; rng=rng_sys)
    alg = Metropolis(make_rng(seed + 1); β=β)

    run_updates!(sys, alg, move!, UPDATES_EQUI)

    cpu_ms_runs = Vector{Float64}(undef, REPEATS)
    ns_per_update_runs = Vector{Float64}(undef, REPEATS)
    updates_runs = Vector{Int}(undef, REPEATS)
    for r in 1:REPEATS
        reset!(alg)
        t0 = time_ns()
        run_updates!(sys, alg, move!, UPDATES_GLOBAL)
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

        @printf("%-30s %-15s %-7s %8d %6.2f %10.3f %12.3f %10d %9.4f %9.4f %12.3f\n",
            case, mode, "xoshiro", seed, Float64(β), cpu_ms, ns_per_update,
            nupdates, geom_acc, mc_acc, final_E)
    return nothing
end

println("MCXLatticeMatter benchmark")
println("Configuration: updates_equi=$(UPDATES_EQUI), updates_global=$(UPDATES_GLOBAL), repeats=$(REPEATS)")
println()
@printf("%-30s %-15s %-7s %8s %6s %10s %12s %10s %9s %9s %12s\n",
    "case", "mode", "rng", "seed", "beta", "cpu_ms", "ns/update", "updates", "geom.acc", "MC acc", "final_E")
@printf("%-30s %-15s %-7s %8s %6s %10s %12s %10s %9s %9s %12s\n",
    "------------------------------", "---------------", "-------", "--------", "------", "----------", "------------", "----------", "---------", "---------", "------------")

cases = [
    (name="LatticePolymer(24x24, 8x20)", mode="slither!", seed=2040,
     setup_fn=seed -> setup_lattice(seed; dims=[24, 24], num_poly=8, length_poly=20),
    move_fn=slither!),
    (name="LatticePolymer(24x24, 8x20)", mode="translate!", seed=2041,
     setup_fn=seed -> setup_lattice(seed; dims=[24, 24], num_poly=8, length_poly=20),
     move_fn=translate!),
    (name="LatticePolymer(24x24, 8x20)", mode="pivot!", seed=2042,
     setup_fn=seed -> setup_lattice(seed; dims=[24, 24], num_poly=8, length_poly=20),
        move_fn=pivot!),
    (name="LatticePolymer(20x20, 12x12)", mode="double_bridge!", seed=2043,
     setup_fn=seed -> setup_lattice(seed; dims=[20, 20], num_poly=12, length_poly=12),
        move_fn=double_bridge!),
]

for case in cases
    benchmark_case(case.name, case.mode, case.seed, BETA, case.setup_fn, case.move_fn)
end

# print warning about acceptance rates of pivot and double bridge moves
    @printf("\nNote: double bridge is not ergodic; this test relies on suitable initial conditions.\n")
