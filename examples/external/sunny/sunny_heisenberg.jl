import Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()

using Random, Statistics, Printf
using Plots
using Sunny
using MCXSpins: HeisenbergSystem, init!, energy, magnetization, spin_flip!
using MonteCarloX: MetropolisAlgorithm, accept!, acceptance_rate, reset!

L, therm, prod = 8, 2000, 10_000
Ts = [0.5, 1.0, 1.443, 3.0]
const SEED = 42

function sweep_sunny!(sys, sampler)
    Sunny.step!(sys, sampler)
end

function sweep_mcx_sunny!(sys, alg, N)
    for _ in 1:N
        site = rand(sys.rng, Sunny.eachsite(sys))
        prop = Sunny.propose_uniform(sys, site)
        accept!(alg, Sunny.local_energy_change(sys, site, prop)) && Sunny.setspin!(sys, prop, site)
    end
end

function sweep_mcx_native!(sys, alg, N)
    for _ in 1:N; spin_flip!(sys, alg); end
end

function run_sunny(Ts, therm, prod)
    latvecs = Sunny.lattice_vectors(1, 1, 1, 90, 90, 90)
    sys = Sunny.System(Sunny.Crystal(latvecs, [[0, 0, 0]]), [1 => Sunny.Moment(s=1, g=2)], :dipole; dims=(L, L, L))
    Sunny.set_exchange!(sys, -1.0, Sunny.Bond(1, 1, (1, 0, 0)))
    es, ms, ts = [], [], []
    for (i, T) in enumerate(Ts)
        Sunny.randomize_spins!(sys)
        copy!(sys.rng, Xoshiro(SEED + 100 + i))
        sampler = Sunny.LocalSampler(kT=T, nsweeps=1.0, propose=Sunny.propose_uniform)
        for _ in 1:therm; sweep_sunny!(sys, sampler); end
        t = @elapsed begin
            e = m = 0.0
            for _ in 1:prod
                sweep_sunny!(sys, sampler)
                e += Sunny.energy_per_site(sys)
                mv = sum(sys.dipoles)
                m += sqrt(sum(abs2, mv)) / length(sys.dipoles)
            end
            e /= prod; m /= prod
        end
        push!(es, e); push!(ms, m); push!(ts, t)
    end
    es, ms, ts
end

function run_mcx_sunny(Ts, therm, prod)
    latvecs = Sunny.lattice_vectors(1, 1, 1, 90, 90, 90)
    sys = Sunny.System(Sunny.Crystal(latvecs, [[0, 0, 0]]), [1 => Sunny.Moment(s=1, g=2)], :dipole; dims=(L, L, L))
    Sunny.set_exchange!(sys, -1.0, Sunny.Bond(1, 1, (1, 0, 0)))
    N = L^3
    es, ms, ts = [], [], []
    for (i, T) in enumerate(Ts)
        Sunny.randomize_spins!(sys)
        copy!(sys.rng, Xoshiro(SEED + 100 + i))
        alg = MetropolisAlgorithm(Xoshiro(SEED + 200 + i); β=1/T)
        for _ in 1:therm; sweep_mcx_sunny!(sys, alg, N); end
        reset!(alg)
        t = @elapsed begin
            e = m = 0.0
            for _ in 1:prod
                sweep_mcx_sunny!(sys, alg, N)
                e += Sunny.energy_per_site(sys)
                mv = sum(sys.dipoles)
                m += sqrt(sum(abs2, mv)) / length(sys.dipoles)
            end
            e /= prod; m /= prod
        end
        push!(es, e); push!(ms, m); push!(ts, t)
    end
    es, ms, ts
end

function run_mcx_native(Ts, therm, prod)
    sys = HeisenbergSystem([L, L, L])
    N = L^3
    es, ms, ts = [], [], []
    for (i, T) in enumerate(Ts)
        init!(sys, :random, rng=MersenneTwister(SEED + i))
        alg = MetropolisAlgorithm(Xoshiro(SEED + 100 + i); β=1/T)
        for _ in 1:therm; sweep_mcx_native!(sys, alg, N); end
        reset!(alg)
        t = @elapsed begin
            e = m = 0.0
            for _ in 1:prod
                sweep_mcx_native!(sys, alg, N)
                e += energy(sys) / N
                m += sqrt(sum(abs2, magnetization(sys))) / N
            end
            e /= prod; m /= prod
        end
        push!(es, e); push!(ms, m); push!(ts, t)
    end
    es, ms, ts
end

# Warmup
run_sunny([Ts[1]], 2, 2)
run_mcx_sunny([Ts[1]], 2, 2)
run_mcx_native([Ts[1]], 2, 2)

e_s, m_s, t_s = run_sunny(Ts, therm, prod)
e_bs, m_bs, t_bs = run_mcx_sunny(Ts, therm, prod)
e_n, m_n, t_n = run_mcx_native(Ts, therm, prod)

println("T       e_sunny      e_mcx_sunny  e_mcx_native  m_sunny      m_mcx_sunny  m_mcx_native  t_sunny(s)  t_mcx_sunny  t_native(s)")
for i in eachindex(Ts)
    @printf "%-7.3f %12.6f %12.6f %12.6f  %12.6f %12.6f %12.6f  %10.4f %11.4f %11.4f\n" Ts[i] e_s[i] e_bs[i] e_n[i] m_s[i] m_bs[i] m_n[i] t_s[i] t_bs[i] t_n[i]
end

p1 = plot(Ts, e_s; marker=:o, xlabel="T", ylabel="e per site", label="Sunny")
plot!(p1, Ts, e_bs; marker=:d, label="MCX on Sunny")
plot!(p1, Ts, e_n; marker=:s, label="MCX native")

p2 = plot(Ts, m_s; marker=:o, xlabel="T", ylabel="|m| per site", label="Sunny")
plot!(p2, Ts, m_bs; marker=:d, label="MCX on Sunny")
plot!(p2, Ts, m_n; marker=:s, label="MCX native")

plot(p1, p2; layout=(1, 2), size=(900, 380))
savefig("heisenberg_comparison.pdf")
