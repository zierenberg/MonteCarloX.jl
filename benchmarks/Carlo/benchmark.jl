import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Random, DelimitedFiles
using StatsBase: weights, mean
using MonteCarloX, MCXSpins
using Carlo, Carlo.JobTools
import Ising

datadir = get(ENV, "MCX_EXAMPLE_DATA", normpath(joinpath(@__DIR__, "..", "..", "docs", "src", "data")))
factors_file = joinpath(datadir, "benchmarks.tsv")
bench_file(name) = joinpath(datadir, "bench_$(name).tsv")

function recfactor(comparison, case, ref_ns, mcx_ns)
    new = !isfile(factors_file)
    open(factors_file, "a") do io
        new && println(io, "comparison\tcase\tcpu\tref_ns_per_flip\tmcx_ns_per_flip\tspeedup_mcx")
        println(io, string(comparison, "\t", case, "\t", Sys.CPU_NAME, "\t",
                           ref_ns, "\t", mcx_ns, "\t", ref_ns / mcx_ns))
    end
end

quiet(f) = redirect_stdout(() -> redirect_stderr(f, devnull), devnull)
float_val(x) = parse(Float64, strip(first(split(string(x), '±'))))
ns_per_flip(run, nflips; reps=3) = 1e9 * minimum(@elapsed(run()) for _ in 1:reps) / nflips
mcx_sweep!(sys, alg, n) = (for _ in 1:n; spin_flip!(sys, alg); end)

function mcx_ising_em(L, β; therm=2_000, sweeps=20_000)
    sys = IsingSystem([L, L]); N = L * L
    init!(sys, :random, rng=MersenneTwister(3))
    alg = MetropolisAlgorithm(Xoshiro(4); β=β)
    mcx_sweep!(sys, alg, therm * N)
    e = m = 0.0
    for _ in 1:sweeps
        mcx_sweep!(sys, alg, N)
        e += energy(sys) / N; m += abs(magnetization(sys)) / N
    end
    return e / sweeps, m / sweeps
end

function mcx_ising_time(L, β; sweeps=5_000)
    sys = IsingSystem([L, L]); N = L * L
    init!(sys, :random, rng=MersenneTwister(2))
    alg = MetropolisAlgorithm(Xoshiro(1); β=β)
    mcx_sweep!(sys, alg, 200 * N)
    return ns_per_flip(() -> mcx_sweep!(sys, alg, sweeps * N), sweeps * N)
end

function carlo_e(Ts; L=8, sweeps=20_000, therm=2_000)
    tm = TaskMaker()
    tm.sweeps = sweeps
    tm.thermalization = therm
    tm.binsize = 100
    tm.Lx = L
    foreach(T -> (tm.T = T; task(tm)), Ts)
    dir = joinpath(mktempdir(), "job")
    job = JobInfo(dir, Ising.MC; tasks=make_tasks(tm),
                  checkpoint_time="30:00", run_time="24:00:00")
    quiet(() -> start(job, ["run", "--single"]))
    return [float_val(r["Energy"]) for r in Carlo.ResultTools.dataframe(dir * ".results.json")]
end

function carlo_kernel_time(L, T; sweeps=5_000)
    params = Dict(:Lx => L, :T => T)
    mc = Ising.MC(params)
    ctx = Carlo.MCContext(sweeps, 0, Xoshiro(1), Carlo.Measurements(100))
    Carlo.init!(mc, ctx, params)
    for _ in 1:200
        Carlo.sweep!(mc, ctx)
    end
    return ns_per_flip(() -> (for _ in 1:sweeps; Carlo.sweep!(mc, ctx); end), sweeps * L * L)
end

function main()
    L, Ts = 8, [1.0, 1.429, 2.0, 2.429, 3.0]
    logdos = logdos_exact_ising2D(L)
    E = get_centers(logdos)
    e_exact = [mean(E, weights(reweight(logdos, -E ./ T))) / L^2 for T in Ts]
    e_ref = carlo_e(Ts; L=L)
    e_mcx = [mcx_ising_em(L, 1 / T; therm=2_000, sweeps=20_000)[1] for T in Ts]

    writedlm(bench_file("carlo"),
             [["T" "e_ref" "e_mcx" "e_exact"]; hcat(Ts, e_ref, e_mcx, e_exact)], '\t')

    for T in (1.0, 2.286, 3.0)
        recfactor("Carlo.jl kernel (2D Ising 16×16)", "T=$T",
                  carlo_kernel_time(16, T), mcx_ising_time(16, 1 / T))
    end
end

main()
