# MCXSpins top-performance benchmark: how close does the composed SpinSystem get to
# hand-optimized "custom C"-style code?
#
# Contenders (2D Ising 64×64 at β = 0.44, plus Blume–Capel 48×48):
#   local     — LocalTightIsing: a hand-rolled typewriter-sweep kernel with inlined
#               neighbor arithmetic and tabulated acceptance. No framework, no proposal,
#               no bookkeeping: the speed ceiling a dedicated C program would hit.
#   MCX table — composed SpinSystem + TableMetropolis (tabulated acceptance, random site,
#               full delta/cache bookkeeping)
#   MCX cont  — composed SpinSystem + MetropolisAlgorithm (continuous exp() acceptance)
#
# Run from the repo root:
#     julia --project=MCXSpins MCXSpins/benchmarks/benchmark_mcxspins.jl [sweeps] [equi]
#
# Caveat: `local` sweeps sites in typewriter order while MCX picks random sites — identical
# stationary distribution, different memory-access pattern. The comparison bounds the total
# framework cost (generic hooks + cache updates + rng site picks), not one isolated factor.

using Random
using Printf
using MonteCarloX
using MCXSpins

const SWEEPS_GLOBAL = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100_000
const SWEEPS_EQUI = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1_000
const BETA_ISING = 0.44
const BETA_BC = 0.40

@inline function benchmark_steps(alg)
    hasproperty(alg, :steps) || throw(ArgumentError("Benchmark algorithm $(typeof(alg)) does not expose a steps counter"))
    return getproperty(alg, :steps)
end

function benchmark_case(case::AbstractString, mode::AbstractString, rng::AbstractString, seed::Integer, β::Real, sys, alg)
    run_updates!(sys, alg, SWEEPS_EQUI)

    steps_before = benchmark_steps(alg)
    t0 = time_ns()
    run_updates!(sys, alg, SWEEPS_GLOBAL)
    t1 = time_ns()
    steps_after = benchmark_steps(alg)

    elapsed_ns = t1 - t0
    nsteps_measured = steps_after - steps_before
    nsteps_measured > 0 || throw(ArgumentError("Benchmark case $(case) / $(mode) produced no updates; steps_before=$(steps_before), steps_after=$(steps_after)"))
    cpu_ms = elapsed_ns / 1.0e6
    ns_per_flip = elapsed_ns / nsteps_measured
    final_E = energy(sys)

    @printf("%-22s %-8s %-7s %8d %6.2f %10.3f %10.6f %10d %10.3f\n",
            case, mode, rng, seed, Float64(β), cpu_ms, ns_per_flip, nsteps_measured, final_E)
    return nothing
end

@inline function run_updates!(sys, alg, sweeps::Int)
    nsteps = sweeps * length(sys.spins)
    for _ in 1:nsteps
        spin_flip!(sys, alg)
    end
    return nothing
end

# ── Tabulated Metropolis (benchmark-only) ─────────────────────────────────────
#
# Zero-exp() acceptance for 2D-lattice-Ising-like ΔE ∈ {−8,−4,0,4,8}, which the interactions
# design delivers whenever J is an integer (caches store coupling-free Int sums). Deliberately
# NOT part of MCXSpins: the acceptance table silently assumes the Ising ΔE spectrum, which is
# too easy to misuse outside a benchmark.

mutable struct TableMetropolis{R<:AbstractRNG} <: MonteCarloX.AbstractMarkovChainMonteCarlo
    rng::R
    p4::Float64
    p8::Float64
    steps::Int
    accepted::Int
end
TableMetropolis(rng::AbstractRNG; β::Real) = TableMetropolis(rng, exp(-4β), exp(-8β), 0, 0)

# The scalar passed to accept! IS the integer ΔE (not a logR) — a specialized fast path.
@inline function MonteCarloX.accept!(alg::TableMetropolis, dE::Int)
    alg.steps += 1
    dE <= 0 && (alg.accepted += 1; return true)
    p = dE == 4 ? alg.p4 : alg.p8
    accepted = rand(alg.rng) < p
    alg.accepted += accepted
    return accepted
end

@inline function MCXSpins.spin_flip!(sys::MCXSpins.AbstractSpinSystem, alg::TableMetropolis)
    i = pick_site(alg.rng, length(sys.spins))
    s_new = propose_state(alg.rng, sys, i)
    δs = MCXSpins.delta_sys(sys, i, s_new)
    ΔE = delta_energy(sys, i, s_new, δs)
    MonteCarloX.accept!(alg, ΔE) && modify!(sys, i, s_new, δs)
    return nothing
end

# ── Hand-rolled speed ceiling: inlined typewriter sweep, tabulated acceptance ──

mutable struct LocalTightIsing
    spins::Vector{Int8}
    L::Int
end

mutable struct LocalTightMetropolis{R<:AbstractRNG}
    rng::R
    p4::Float64
    p8::Float64
    steps::Int
    accepted::Int
end

function LocalTightIsing(L::Int, rng::AbstractRNG)
    spins = Vector{Int8}(undef, L * L)
    @inbounds for i in eachindex(spins)
        spins[i] = rand(rng, Bool) ? Int8(1) : Int8(-1)
    end
    return LocalTightIsing(spins, L)
end

LocalTightMetropolis(rng::AbstractRNG; β::Real) =
    LocalTightMetropolis(rng, exp(-4β), exp(-8β), 0, 0)

Base.length(sys::LocalTightIsing) = length(sys.spins)

function MCXSpins.energy(sys::LocalTightIsing)
    e = 0
    spins = sys.spins
    L = sys.L
    @inbounds for y in 1:sys.L
        ym = y == 1 ? sys.L : y - 1
        yp = y == sys.L ? 1 : y + 1
        row = (y - 1) * sys.L
        rowm = (ym - 1) * sys.L
        rowp = (yp - 1) * L
        for x in 1:L
            xm = x == 1 ? L : x - 1
            xp = x == L ? 1 : x + 1
            s = Int(spins[row + x])
            lsum = Int(spins[row + xm]) + Int(spins[row + xp]) +
                   Int(spins[rowm + x]) + Int(spins[rowp + x])
            e += s * lsum
        end
    end
    return -(e ÷ 2)
end

@inline function run_updates!(sys::LocalTightIsing, alg::LocalTightMetropolis, sweeps::Int)
    L = sys.L
    spins = sys.spins
    for _ in 1:sweeps
        @inbounds for y in 1:L
            ym = y == 1 ? L : y - 1
            yp = y == L ? 1 : y + 1
            row = (y - 1) * L
            rowm = (ym - 1) * L
            rowp = (yp - 1) * L
            for x in 1:L
                xm = x == 1 ? L : x - 1
                xp = x == L ? 1 : x + 1
                i = row + x
                s = Int(spins[i])
                lsum = Int(spins[row + xm]) + Int(spins[row + xp]) +
                       Int(spins[rowm + x]) + Int(spins[rowp + x])
                ide = s * lsum
                alg.steps += 1
                accepted = ide <= 0 || rand(alg.rng) < (ide == 2 ? alg.p4 : alg.p8)
                if accepted
                    spins[i] = Int8(-s)
                    alg.accepted += 1
                end
            end
        end
    end
    return nothing
end

function main()
    println("MCXSpins top-performance benchmark (vs hand-rolled kernel)")
    println("Configuration: sweeps_equi=$(SWEEPS_EQUI), sweeps_global=$(SWEEPS_GLOBAL)")
    println()
    @printf("%-22s %-8s %-7s %8s %6s %10s %10s %10s %10s\n",
            "case", "mode", "rng", "seed", "beta", "cpu_ms", "ns/flip", "updates", "final_E")
    @printf("%-22s %-8s %-7s %8s %6s %10s %10s %10s %10s\n",
            "----------------------", "--------", "-------", "--------", "------", "----------", "----------", "----------", "----------")

    seed = 2026

    # Hand-rolled speed ceiling
    sys_local = LocalTightIsing(64, Xoshiro(seed))
    alg_local = LocalTightMetropolis(Xoshiro(seed); β=BETA_ISING)
    benchmark_case("local Ising 64x64", "table", "xoshiro", seed, BETA_ISING, sys_local, alg_local)

    # Composed SpinSystem: tabulated and continuous, two RNGs
    sys_tbl_x = IsingSystem([64, 64])
    init!(sys_tbl_x, :random; rng=Xoshiro(seed))
    benchmark_case("MCX ising 64x64", "table", "xoshiro", seed, BETA_ISING, sys_tbl_x,
                   TableMetropolis(Xoshiro(seed); β=BETA_ISING))

    sys_tbl_m = IsingSystem([64, 64])
    init!(sys_tbl_m, :random; rng=MersenneTwister(seed))
    benchmark_case("MCX ising 64x64", "table", "mt", seed, BETA_ISING, sys_tbl_m,
                   TableMetropolis(MersenneTwister(seed); β=BETA_ISING))

    sys_cont_x = IsingSystem([64, 64])
    init!(sys_cont_x, :random; rng=Xoshiro(seed))
    benchmark_case("MCX ising 64x64", "cont", "xoshiro", seed, BETA_ISING, sys_cont_x,
                   MetropolisAlgorithm(Xoshiro(seed); β=BETA_ISING))

    sys_cont_m = IsingSystem([64, 64])
    init!(sys_cont_m, :random; rng=MersenneTwister(seed))
    benchmark_case("MCX ising 64x64", "cont", "mt", seed, BETA_ISING, sys_cont_m,
                   MetropolisAlgorithm(MersenneTwister(seed); β=BETA_ISING))

    # Blume-Capel (3-state, crystal field + field term)
    sys_bc = BlumeCapelSystem([48, 48]; J=1.0, D=0.5, h=0.1)
    init!(sys_bc, :random; rng=Xoshiro(2028))
    benchmark_case("MCX blume_capel 48x48", "cont", "xoshiro", seed, BETA_BC, sys_bc,
                   MetropolisAlgorithm(Xoshiro(seed); β=BETA_BC))

    return nothing
end

main()
