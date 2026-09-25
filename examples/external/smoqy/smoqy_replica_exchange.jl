# # Replica exchange over a helper Hamiltonian term (SmoQyDQMC)
#
# Split the Hamiltonian into the physical part and one helper term whose strength you control,
# ```math
# \hat{H}(\lambda) = \hat{H}_0 + \lambda \hat{H}_1 ,
# ```
# and run replicas in ``\lambda``. Two replicas differ only through ``\lambda``, so their exchange
# ratio collapses to
# ```math
# \log R = (\lambda_i - \lambda_j)(X_i - X_j) , \qquad X = \partial S / \partial \lambda ,
# ```
# with ``X`` the configuration's contribution from ``\hat{H}_1`` alone. All of ``\hat{H}_0``
# cancels — here that includes the fermion determinant, which is what makes the swap cheap.
#
# The helper term need not be invented. It is enough that the Hamiltonian already contains one
# that flattens the barrier when turned up.
#
# ## The model
#
# [SmoQyDQMC's optical Su-Schrieffer-Heeger chain](https://smoqysuite.github.io/SmoQyDQMC.jl/stable/examples/ossh_chain/),
# ```math
# \hat{H} = \sum_i \left( \tfrac{1}{2M}\hat{P}^2_i + \tfrac{1}{2}M\Omega^2 \hat{X}^2_i \right)
#         - \sum_{\sigma,i} \left[ t - \alpha (\hat{X}_{i+1} - \hat{X}_i) \right]
#           (\hat{c}^\dagger_{\sigma,i+1}\hat{c}^{\phantom\dagger}_{\sigma,i} + \mathrm{h.c.}),
# ```
# a site phonon whose displacement modulates the neighbouring hopping. At half filling the chain
# dimerizes,
# ```math
# \Delta = \frac{1}{N}\sum_i (-1)^i \, \overline{x}_i ,
# ```
# with ``\pm\Delta`` degenerate across a free-energy barrier that local updates do not cross.
#
# ## The helper term
#
# Take ``\hat{H}_1 = \sum_i \tfrac{1}{2} M \hat{X}_i^2``, the phonons' own harmonic potential, so
# ``\lambda = \Omega^2`` and the physical chain is the lowest ``\lambda``. Raising ``\lambda``
# stiffens the oscillators, confines ``x \to 0``, and melts the barrier.
#
# The appealing helper — ``\hat{H}_1 = -\hat{H}_\text{e-ph}``, switching off ``\alpha`` to decouple
# the phonons — is not available, because ``X`` must be computable from the stored configuration
# and ``\alpha`` enters the determinant. That leaves the bosonic action, where the harmonic
# ``\Omega^2`` and the quartic ``\Omega_4^2`` are the two candidates.
#
# Run: `julia -t auto examples/external/smoqy/smoqy_replica_exchange.jl` (`--rerun` recomputes).

import Pkg; Pkg.activate(@__DIR__)  #src

using Random, Statistics, Plots, Printf, DelimitedFiles, Markdown
using SmoQyDQMC, MonteCarloX
import SmoQyDQMC.LatticeUtilities as lu
import SmoQyDQMC.JDQMCFramework as dqmcf

SEED = 42
L, β, Δτ = 8, 4.0, 0.1          # chain length, inverse temperature, imaginary-time step
Ω, α, μ = 1.0, 0.75, 0.0        # physical phonon frequency, e-ph coupling, chemical potential
λ_phys, λ_max, n_replicas = Ω^2, 2.5^2, 12   # geometric λ schedule, λ = Ω²
λ_sched = exp.(range(log(λ_phys), log(λ_max), length=n_replicas))
n_hmc = 5                       # EFA-HMC updates between exchange attempts
n_therm, n_measure = 200, 1_500
nothing #hide

# ## The ensemble
#
# Each replica carries a coupling strength, with log-weight ``-\lambda X``:

struct TiltedEnsemble{T<:Real} <: AbstractEnsemble
    λ::T
end

MonteCarloX.logweight(e::TiltedEnsemble, X::Real) = -e.λ * X
MonteCarloX.linear_logweight(::TiltedEnsemble) = true

# ## The replica
#
# SmoQyDQMC mutates `G`, the propagators and the path integral in place, but returns these four
# scalars by value. Each replica is its own chain, so each keeps its own set.

mutable struct Bookkeeping
    logdetG::Float64
    sgndetG::Float64
    δG::Float64       # running maximum of the error corrected by stabilization
    δθ::Float64
end

# SmoQyDQMC's own `ossh_chain` setup, without its measurement and checkpoint machinery.

function build_replica(; λ, seed, Nt=10, n_stab=10)
    rng = Xoshiro(seed)
    unit_cell = lu.UnitCell(lattice_vecs=[[1.0]], basis_vecs=[[0.0]])
    lattice = lu.Lattice(L=[L], periodic=[true])
    geometry = ModelGeometry(unit_cell, lattice)
    bond = lu.Bond(orbitals=(1, 1), displacement=[1])
    add_bond!(geometry, bond)

    tbm = TightBindingModel(model_geometry=geometry, t_bonds=[bond], t_mean=[1.0], μ=μ, ϵ_mean=[0.0])
    epm = ElectronPhononModel(model_geometry=geometry, tight_binding_model=tbm)
    phonon_id = add_phonon_mode!(electron_phonon_model=epm,
                                 phonon_mode=PhononMode(basis_vec=[0.0], Ω_mean=sqrt(λ)))
    add_ssh_coupling!(electron_phonon_model=epm, tight_binding_model=tbm,
                      ssh_coupling=SSHCoupling(model_geometry=geometry, tight_binding_model=tbm,
                                               phonon_ids=(phonon_id, phonon_id),
                                               bond=bond, α_mean=α))

    tbp = TightBindingParameters(tight_binding_model=tbm, model_geometry=geometry, rng=rng)
    epp = ElectronPhononParameters(β=β, Δτ=Δτ, electron_phonon_model=epm,
                                   tight_binding_parameters=tbp, model_geometry=geometry, rng=rng)
    fpi = FermionPathIntegral(tight_binding_parameters=tbp, β=β, Δτ=Δτ)
    initialize!(fpi, epp)
    B = initialize_propagators(fpi, symmetric=true, checkerboard=true)
    fgc = dqmcf.FermionGreensCalculator(B, β, Δτ, n_stab)
    fgc_alt = dqmcf.FermionGreensCalculator(fgc)
    G = zeros(eltype(B[1]), size(B[1]))
    logdetG, sgndetG = dqmcf.calculate_equaltime_greens!(G, fgc)
    hmc = EFAHMCUpdater(electron_phonon_parameters=epp, G=G, Nt=Nt, Δt=π/(2Nt))
    (; epp, fpi, B, fgc, fgc_alt, G, hmc, rng, state = Bookkeeping(logdetG, sgndetG, 0.0, 0.0))
end

# The order parameter, and the coordinate ``X`` conjugate to ``\lambda``.

function dimerization(r)
    x = r.epp.x
    sum((isodd(i) ? -1 : 1) * mean(@view x[i, :]) for i in axes(x, 1)) / size(x, 1)
end

function coordinate(r)
    (; x, Δτ, phonon_parameters) = r.epp
    M = phonon_parameters.M
    s = 0.0
    @inbounds for l in axes(x, 2), n in axes(x, 1)
        s += Δτ * M[n] / 2 * x[n, l]^2
    end
    s
end

# ## 1. SmoQyDQMC native
#
# One chain at the physical ``\lambda``: EFA-HMC plus the two global moves SmoQyDQMC ships for this
# kind of barrier — `reflection_update!` (``x \to -x`` on one mode) and `swap_update!` (exchange
# two modes' fields).

function sweep_native!(r, n_hmc)
    s = r.state

    for _ in 1:n_hmc
        (_, s.logdetG, s.sgndetG, s.δG, s.δθ) = hmc_update!(
            r.G, s.logdetG, s.sgndetG, r.epp, r.hmc;
            fermion_path_integral         = r.fpi,
            fermion_greens_calculator     = r.fgc,
            fermion_greens_calculator_alt = r.fgc_alt,
            B = r.B, δG_max = 1e-6, δG = s.δG, δθ = s.δθ, rng = r.rng)
    end

    (accepted_reflection, s.logdetG, s.sgndetG) = reflection_update!(
        r.G, s.logdetG, s.sgndetG, r.epp;
        fermion_path_integral         = r.fpi,
        fermion_greens_calculator     = r.fgc,
        fermion_greens_calculator_alt = r.fgc_alt,
        B = r.B, rng = r.rng)

    (accepted_swap, s.logdetG, s.sgndetG) = swap_update!(
        r.G, s.logdetG, s.sgndetG, r.epp;
        fermion_path_integral         = r.fpi,
        fermion_greens_calculator     = r.fgc,
        fermion_greens_calculator_alt = r.fgc_alt,
        B = r.B, rng = r.rng)

    accepted_reflection, accepted_swap
end

function run_native()
    r = build_replica(; λ=λ_phys, seed=SEED)
    for _ in 1:n_therm; sweep_native!(r, n_hmc); end
    Δ, p_refl, p_swap = Float64[], 0.0, 0.0
    t_run = @elapsed for _ in 1:n_measure
        (acc_refl, acc_swap) = sweep_native!(r, n_hmc)
        p_refl += acc_refl
        p_swap += acc_swap
        push!(Δ, dimerization(r))
    end
    Δ, (p_refl, p_swap) ./ n_measure, t_run
end

# ## 2. EFA-HMC + MCX replica exchange in λ
#
# MonteCarloX owns the exchange and nothing else: it scores swaps with `X` and moves ensembles
# between replicas, never seeing the Hamiltonian, the phonon fields or the determinant. `sweep!`
# needs no `accept!` because `hmc_update!` is already a complete Metropolis step.
#
# After a swap, ``\Omega`` is read live out of `phonon_parameters`, so assigning it suffices. The
# EFA accelerator instead caches its ``\Omega``-dependent mass matrix at construction, so it stays
# with its ``\lambda`` and the replica borrows it — safe because the assignment is a permutation.

function run_replica_exchange()
    replicas = [build_replica(; λ=λ_sched[i], seed=SEED + i) for i in 1:n_replicas]
    re = ReplicaExchange([TiltedEnsemble(λ) for λ in λ_sched]; seed=SEED, rng=Xoshiro)

    accel = Dict(λ_sched[i] => replicas[i].hmc for i in 1:n_replicas)

    ## One unit of sampling between exchange attempts; below the first two lines it is all
    ## SmoQyDQMC.
    function sweep!(replica, alg)
        λ = ensemble(alg).λ                          # λ is whichever ensemble it now carries
        replica.epp.phonon_parameters.Ω .= sqrt(λ)   # model follows λ

        s = replica.state
        (accepted, s.logdetG, s.sgndetG, s.δG, s.δθ) = hmc_update!(
            replica.G, s.logdetG, s.sgndetG, replica.epp, accel[λ];
            fermion_path_integral         = replica.fpi,
            fermion_greens_calculator     = replica.fgc,
            fermion_greens_calculator_alt = replica.fgc_alt,
            B = replica.B, δG_max = 1e-6, δG = s.δG, δθ = s.δθ, rng = replica.rng)
        accepted
    end

    for _ in 1:n_therm
        advance!(sweep!, re, replicas, n_hmc)
        attempt_exchange!(re, coordinate.(replicas))
    end
    Δ = Float64[]
    t_run = @elapsed for _ in 1:n_measure
        advance!(sweep!, re, replicas, n_hmc)
        attempt_exchange!(re, coordinate.(replicas))
        physical = findfirst(r -> ensemble_index(re, r) == 1, 1:n_replicas)
        push!(Δ, dimerization(replicas[physical]))
    end
    Δ, acceptance_rates(re), t_run
end

# ## 3. Comparison
#
# Sign changes of `Δ` count tunneling events. `⟨|Δ|⟩` is the dimerization amplitude and must agree
# between the runs: the exchange fixes ergodicity, it does not change the physics.

datadir   = get(ENV, "MCX_EXAMPLE_DATA", normpath(joinpath(@__DIR__, "..", "..", "..", "docs", "src", "data")))  #hide
trace_file = joinpath(datadir, "smoqy_ossh_L$(L)_dimerization.tsv")  #hide
meta_file  = joinpath(datadir, "smoqy_ossh_L$(L)_summary.tsv")       #hide
rerun      = "--rerun" in ARGS || "--reset" in ARGS                  #hide

nflips(Δ) = count(i -> sign(Δ[i]) != sign(Δ[i-1]), 2:length(Δ))                                   #hide
runstage(label, f) = (print(stderr, label, " ... "); flush(stderr); r = f(); println(stderr, "done"); r)  #hide
if rerun || !isfile(trace_file)                                                                   #hide
    Δ1, p1, t1 = runstage("SmoQyDQMC native  ", run_native)                                       #hide
    Δ2, p2, t2 = runstage("MCX λ-exchange      ", run_replica_exchange)                             #hide
    mkpath(datadir)                                                                               #hide
    writedlm(trace_file, [["native" "replica_exchange"]; hcat(Δ1, Δ2)], '\t')                     #hide
    writedlm(meta_file,                                                                           #hide
        [["variant" "run_s" "mean_D" "abs_D" "flips" "p_a" "p_b"];                                #hide
         ["SmoQyDQMC native" t1 mean(Δ1) mean(abs, Δ1) nflips(Δ1) p1[1] p1[2]];                   #hide
         ["MCX λ-exchange" t2 mean(Δ2) mean(abs, Δ2) nflips(Δ2) mean(p2) minimum(p2)]], '\t')       #hide
else                                                                                              #hide
    println(stderr, "loaded precomputed results from $(relpath(trace_file)) (pass --rerun to recompute)")  #src
end                                                                                               #hide

# `p`: reflection/swap acceptance for the native run, mean/min over neighbouring λ for the other.

meta = readdlm(meta_file, '\t'; header=true)[1]                                                   #hide
println("\noSSH chain: L=$L, β=$β, Δτ=$Δτ, Ω=$Ω, α=$α  (threads=$(Threads.nthreads()))")          #hide
io = IOBuffer()                                                                                   #hide
println(io, "| variant | run [s] | ⟨Δ⟩ | ⟨\\|Δ\\|⟩ | sign flips | p |")                           #hide
println(io, "|---|---:|---:|---:|---:|---:|")                                                     #hide
for r in axes(meta, 1)                                                                            #hide
    row = @sprintf("| %s | %.1f | %+.3f | %.3f | %d / %d | %.2f / %.2f |",                        #hide
        meta[r,1], meta[r,2], meta[r,3], meta[r,4], meta[r,5], n_measure, meta[r,6], meta[r,7])   #hide
    println(io, row)                                                                              #hide
    println(@sprintf("%-18s %6.1f s  <D>=%+.3f  <|D|>=%.3f  flips=%4d/%d  p=%.2f/%.2f",           #hide
        meta[r,1], meta[r,2], meta[r,3], meta[r,4], meta[r,5], n_measure, meta[r,6], meta[r,7]))  #hide
end                                                                                               #hide
Markdown.parse(String(take!(io)))                                                                 #hide

# The native run sits in one well throughout: its global moves are accepted at a healthy rate,
# they just do not move this collective coordinate.

tr = readdlm(trace_file, '\t'; header=true)[1]                                                    #hide
## one colour per run, same order in both panels, so identity never depends on plot order          #hide
c_native, c_exchange = "#2a78d6", "#eb6834"                                                       #hide
pt = plot(; xlabel="measurement", ylabel="Δ", legend=:topright)                                   #hide
plot!(pt, tr[:, 1]; label="SmoQyDQMC native", lw=1, color=c_native)                               #hide
plot!(pt, tr[:, 2]; label="MCX λ-exchange", lw=1, color=c_exchange)                               #hide
ph = plot(; xlabel="Δ", ylabel="count", legend=:topright)                                         #hide
histogram!(ph, tr[:, 1]; bins=60, label="SmoQyDQMC native",                                       #hide
           fillcolor=c_native, fillalpha=0.55, linecolor=c_native, lw=0.5)                        #hide
histogram!(ph, tr[:, 2]; bins=60, label="MCX λ-exchange",                                         #hide
           fillcolor=c_exchange, fillalpha=0.55, linecolor=c_exchange, lw=0.5)                    #hide
## combined subplots do not reserve room for the outer axis labels — set the margins explicitly    #hide
plt = plot(pt, ph; layout=(1, 2), size=(950, 390),                                                #hide
           left_margin=6Plots.mm, bottom_margin=6Plots.mm)                                        #hide
savefig(plt, joinpath(@__DIR__, "smoqy_dimerization.png"))                                        #hide
plt                                                                                               #hide
