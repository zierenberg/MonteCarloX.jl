# MCXSpins top-performance benchmark: how close does the composed SpinSystem get to
# hand-optimized "custom C"-style code?
#
# Contenders (2D Ising 64×64 at β = 0.44, plus Blume–Capel 48×48):
#   local     — LocalTightIsing: a hand-rolled typewriter-sweep kernel with inlined
#               neighbor arithmetic and tabulated acceptance. No framework, no proposal,
#               no bookkeeping — the Julia stand-in for the speed ceiling.
#   C         — the identical physics as a standalone compiled C program
#               (references/ising_cpu_modes.c): typewriter sweep, tabulated or continuous
#               acceptance, mt/xoshiro RNG. The true ceiling; skipped if no `cc` is found.
#   MCX table — composed SpinSystem + TableMetropolis (tabulated acceptance, random site,
#               full delta/cache bookkeeping)
#   MCX cont  — composed SpinSystem + MetropolisAlgorithm (continuous exp() acceptance)
#
# Run from the repo root:
#     julia --project=MCXSpins MCXSpins/benchmarks/benchmark_mcxspins.jl [sweeps] [equi] [--record[=path]]
#
# Absolute ns/flip is machine-bound; the durable regression signal is the FACTOR relative
# to the hand-rolled kernel measured in the SAME run (framework cost in machine-free
# units). `--record` appends both to a TSV baseline (default: baseline.tsv next to this
# script) tagged with date/commit/julia/cpu.
#
# Caveat: `local` and `C` sweep sites in typewriter order while MCX picks random sites —
# identical stationary distribution, different memory-access pattern. The comparison bounds
# the total framework cost (generic hooks + cache updates + rng site picks), not one isolated
# factor.

using Random
using Printf
using Dates
using MonteCarloX
using MCXSpins

const ARGS_POS = filter(a -> !startswith(a, "--"), ARGS)
const RECORD_PATH = let i = findfirst(a -> a == "--record" || startswith(a, "--record="), ARGS)
    i === nothing ? nothing :
        ARGS[i] == "--record" ? joinpath(@__DIR__, "baseline.tsv") :
                                String(split(ARGS[i], '='; limit=2)[2])
end
const SWEEPS_GLOBAL = length(ARGS_POS) >= 1 ? parse(Int, ARGS_POS[1]) : 100_000
const SWEEPS_EQUI = length(ARGS_POS) >= 2 ? parse(Int, ARGS_POS[2]) : 1_000
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
    return (case=case, mode=mode, rng=rng, ns=ns_per_flip)
end

# The regression-proof record: factors relative to the hand-rolled kernel of the SAME run.
function report_factors(results)
    ref = results[1].ns
    println("\nFactor vs hand-rolled kernel (same run — compare THIS across machines/commits):")
    for r in results
        @printf("  %-22s %-8s %-7s %6.2f\n", r.case, r.mode, r.rng, r.ns / ref)
    end
    return ref
end

function record_results(path, results)
    ref = results[1].ns
    commit = try readchomp(`git -C $(@__DIR__) rev-parse --short HEAD`) catch; "unknown" end
    header = !isfile(path)
    open(path, "a") do io
        header && println(io, "date\tcommit\tjulia\tcpu\tsweeps\tcase\tmode\trng\tns_per_flip\tfactor")
        for r in results
            @printf(io, "%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\t%.3f\t%.3f\n",
                    Dates.format(now(), "yyyy-mm-dd"), commit, string(VERSION),
                    Sys.CPU_NAME, SWEEPS_GLOBAL, r.case, r.mode, r.rng, r.ns, r.ns / ref)
        end
    end
    println("Recorded ", length(results), " rows to ", path)
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

# ── Real compiled-C ceiling: references/ising_cpu_modes.c ─────────────────────
#
# Same physics as LocalTightIsing (L=64, β=0.44, 10⁵ sweeps are compile-time constants in
# the C source, matching the defaults here). Compiled once with the system `cc`; returns the
# binary path, or `nothing` if no working compiler is found so the C rows are simply skipped.

const C_SOURCE = normpath(joinpath(@__DIR__, "..", "references", "ising_cpu_modes.c"))

function compile_c_kernel()
    cc = Sys.which("cc")
    cc === nothing && return nothing
    bin = joinpath(mktempdir(), "ising_cpu_modes")
    try
        run(pipeline(`$cc -O3 -march=native -std=c11 -o $bin $C_SOURCE -lm`; stderr=devnull))
    catch
        return nothing
    end
    return bin
end

# Run one C mode (`table-std`/`cont-std`) with the given RNG and parse its self-reported row.
function benchmark_c_case(bin, case, c_mode, disp_mode, rng, seed, β)
    out = split(last(split(readchomp(`$bin $c_mode $rng $seed`), '\n')))
    cpu_ms  = parse(Float64, out[5])
    ns      = parse(Float64, out[6])
    updates = parse(Int, out[7])
    final_E = -parse(Int, out[8])          # C uses the +Σs·neighbors/2 sign convention; flip to match
    @printf("%-22s %-8s %-7s %8d %6.2f %10.3f %10.6f %10d %10.3f\n",
            case, disp_mode, rng, seed, Float64(β), cpu_ms, ns, updates, Float64(final_E))
    return (case=case, mode=disp_mode, rng=rng, ns=ns)
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
    results = NamedTuple[]

    # Hand-rolled speed ceiling — FIRST: it is the normalization reference
    sys_local = LocalTightIsing(64, Xoshiro(seed))
    alg_local = LocalTightMetropolis(Xoshiro(seed); β=BETA_ISING)
    push!(results, benchmark_case("local Ising 64x64", "table", "xoshiro", seed, BETA_ISING, sys_local, alg_local))

    # Real compiled-C ceiling, one row per (acceptance, rng) MCX also runs; skipped without `cc`
    cbin = compile_c_kernel()
    if cbin === nothing
        println("  (skipping compiled-C rows: no working `cc` found)")
    else
        push!(results, benchmark_c_case(cbin, "C ising 64x64", "table-std", "table", "xoshiro", seed, BETA_ISING))
        push!(results, benchmark_c_case(cbin, "C ising 64x64", "table-std", "table", "mt", seed, BETA_ISING))
        push!(results, benchmark_c_case(cbin, "C ising 64x64", "cont-std", "cont", "xoshiro", seed, BETA_ISING))
        push!(results, benchmark_c_case(cbin, "C ising 64x64", "cont-std", "cont", "mt", seed, BETA_ISING))
    end

    # Composed SpinSystem: tabulated and continuous, two RNGs
    sys_tbl_x = IsingSystem([64, 64])
    init!(sys_tbl_x, :random; rng=Xoshiro(seed))
    push!(results, benchmark_case("MCX ising 64x64", "table", "xoshiro", seed, BETA_ISING, sys_tbl_x,
                                  TableMetropolis(Xoshiro(seed); β=BETA_ISING)))

    sys_tbl_m = IsingSystem([64, 64])
    init!(sys_tbl_m, :random; rng=MersenneTwister(seed))
    push!(results, benchmark_case("MCX ising 64x64", "table", "mt", seed, BETA_ISING, sys_tbl_m,
                                  TableMetropolis(MersenneTwister(seed); β=BETA_ISING)))

    sys_cont_x = IsingSystem([64, 64])
    init!(sys_cont_x, :random; rng=Xoshiro(seed))
    push!(results, benchmark_case("MCX ising 64x64", "cont", "xoshiro", seed, BETA_ISING, sys_cont_x,
                                  MetropolisAlgorithm(Xoshiro(seed); β=BETA_ISING)))

    sys_cont_m = IsingSystem([64, 64])
    init!(sys_cont_m, :random; rng=MersenneTwister(seed))
    push!(results, benchmark_case("MCX ising 64x64", "cont", "mt", seed, BETA_ISING, sys_cont_m,
                                  MetropolisAlgorithm(MersenneTwister(seed); β=BETA_ISING)))

    # Blume-Capel (3-state, crystal field + field term)
    sys_bc = BlumeCapelSystem([48, 48]; J=1.0, D=0.5, h=0.1)
    init!(sys_bc, :random; rng=Xoshiro(2028))
    push!(results, benchmark_case("MCX blume_capel 48x48", "cont", "xoshiro", seed, BETA_BC, sys_bc,
                                  MetropolisAlgorithm(Xoshiro(seed); β=BETA_BC)))

    report_factors(results)
    RECORD_PATH === nothing || record_results(RECORD_PATH, results)
    return nothing
end

main()
