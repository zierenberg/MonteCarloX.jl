# Wang-Landau: Sunny vs MCX Bridge vs MCX Native
#
# This example demonstrates Wang-Landau (flat-histogram, online weight adaptation)
# on the 2D Ising model in three implementations: Sunny native, MCX bridge, and MCX native.
# The related offline method is Multicanonical; see examples/mcmc/muca_Ising2D.jl.

using Random, Statistics, Plots, Printf
using Sunny, MonteCarloX, MCXSpins

const SEED = 42
const L = 8
const n_iters = 10           # Wang-Landau iterations (logf halved each time)
const nsweeps_per_check = 500 # sweeps between flatness checks
const flatness_p = 0.8        # Sunny flatness criterion (fraction of min/mean)
const flatness_ratio = 1.3    # MCX flatness criterion (max/mean <= this => flat)

# ============================================================================
# SECTION 1: SUNNY WANG-LANDAU
# ============================================================================
# WangLandau wraps the system; step_ensemble! does sweeps + online ln_g update.
# After each iteration: reset histogram, halve ln_f.

function run_sunny_wl(n_iters, nsweeps_per_check)
    latvecs = Sunny.lattice_vectors(1, 1, 10, 90, 90, 90)
    sys = Sunny.System(Sunny.Crystal(latvecs, [[0, 0, 0]]),
                       [1 => Sunny.Moment(s=1, g=-1)], :dipole; dims=(L, L, 1))
    Sunny.set_exchange!(sys, -1.0, Sunny.Bond(1, 1, (1, 0, 0)))
    Sunny.polarize_spins!(sys, (0, 0, 1))

    wl = Sunny.WangLandau(; sys, bin_size=1/L^2, bounds=(-2.0, 2.0), propose=Sunny.propose_flip)
    for _ in 1:n_iters
        for _ in 1:100
            Sunny.step_ensemble!(wl, nsweeps_per_check)
            Sunny.check_flat(wl.hist; p=flatness_p) && break
        end
        Sunny.reset!(wl.hist)
        wl.ln_f /= 2
    end

    E = Sunny.get_keys(wl.ln_g) .* L^2
    ln_g = Sunny.get_vals(wl.ln_g)
    E, ln_g
end

# ============================================================================
# SECTION 2: MCX BRIDGE (Sunny system + MCX WangLandauAlgorithm)
# ============================================================================
# MCX's WangLandauAlgorithm wraps a MetropolisHastings engine with a WangLandauEnsemble.
# spin_flip! detects the nonlinear ensemble and passes absolute energies to accept!.
# We drive the Sunny system manually, reusing the Sunny energy query.

function sweep_wl_bridge!(sys, alg, n_sweeps)
    N = length(collect(Sunny.eachsite(sys)))
    for _ in 1:n_sweeps, _ in 1:N
        site = rand(alg.rng, Sunny.eachsite(sys))
        prop = Sunny.propose_flip(sys, site)
        ΔE = Sunny.local_energy_change(sys, site, prop)
        E_old = Sunny.energy_per_site(sys)
        accept!(alg, E_old + ΔE/N, E_old) && Sunny.setspin!(sys, prop, site)
    end
end

function run_bridge_wl(n_iters, nsweeps_per_check)
    latvecs = Sunny.lattice_vectors(1, 1, 10, 90, 90, 90)
    sys = Sunny.System(Sunny.Crystal(latvecs, [[0, 0, 0]]),
                       [1 => Sunny.Moment(s=1, g=-1)], :dipole; dims=(L, L, 1))
    Sunny.set_exchange!(sys, -1.0, Sunny.Bond(1, 1, (1, 0, 0)))
    Sunny.polarize_spins!(sys, (0, 0, 1))

    ΔE_bin = 4.0 / L^2
    E_bins = range(-2.0 - ΔE_bin/2, 2.0 + ΔE_bin/2, step=ΔE_bin)
    alg = WangLandauAlgorithm(Xoshiro(SEED), E_bins)
    for _ in 1:n_iters
        for _ in 1:100
            sweep_wl_bridge!(sys, alg, nsweeps_per_check)
            flatness(ensemble(alg).histogram, minimum(E_bins), maximum(E_bins)) <= flatness_ratio && break
        end
        reset!(alg)
        update_logweight!(ensemble(alg))  # logf *= 0.5
    end

    E = get_centers(ensemble(alg).logweight)
    ln_g = -ensemble(alg).logweight.values  # WL adapts logweight down; DOS is -logweight
    E, ln_g
end

# ============================================================================
# SECTION 3: MCX NATIVE (MCXSpins IsingSystem + MCX WangLandauAlgorithm)
# ============================================================================
# spin_flip! automatically uses accept!(alg, E_new, E_old) for nonlinear ensembles.
# See src/updates/spin_flip.jl: linear_logweight check dispatches to absolute-energy accept!.

sweep!(sys, alg, n) = (for _ in 1:n, _ in 1:length(sys.spins); spin_flip!(sys, alg); end)

function run_mcx_wl(n_iters, nsweeps_per_check)
    sys = IsingSystem([L, L])
    init!(sys, :random; rng=Xoshiro(SEED))

    E_bins = get_centers(logdos_exact_ising2D(L))  # exact energy levels as bin centers
    alg = WangLandauAlgorithm(Xoshiro(SEED), E_bins)
    for _ in 1:n_iters
        for _ in 1:100
            sweep!(sys, alg, nsweeps_per_check)
            flatness(ensemble(alg).histogram, minimum(E_bins), maximum(E_bins)) <= flatness_ratio && break
        end
        reset!(alg)
        update_logweight!(ensemble(alg))  # logf *= 0.5
    end

    E = get_centers(ensemble(alg).logweight)
    ln_g = -ensemble(alg).logweight.values
    E, ln_g
end

# ============================================================================
# MAIN: TIMING COMPARISON
# ============================================================================

println("Wang-Landau: 2D Ising, L=$L, n_iters=$n_iters")
println("="^70)

# Warmup
run_sunny_wl(1, 50)
run_bridge_wl(1, 50)
run_mcx_wl(1, 50)

t_sunny  = @elapsed run_sunny_wl(n_iters, nsweeps_per_check)
t_bridge = @elapsed run_bridge_wl(n_iters, nsweeps_per_check)
t_mcx    = @elapsed run_mcx_wl(n_iters, nsweeps_per_check)

println("Timing Comparison (n_iters=$n_iters, nsweeps_per_check=$nsweeps_per_check):")
println("-"^70)
println(@sprintf "  Sunny native:   %8.4f s" t_sunny)
println(@sprintf "  MCX bridge:     %8.4f s" t_bridge)
println(@sprintf "  MCX native:     %8.4f s" t_mcx)
println(@sprintf "  Speedup MCX/Sunny:  %.2f x" t_sunny / t_mcx)
println()
println("Notes:")
println("  - Sunny: step_ensemble! drives sweeps + online ln_g update")
println("  - Bridge: MCX WangLandauAlgorithm on Sunny system (manual sweep)")
println("  - MCX native: spin_flip! auto-detects nonlinear ensemble → accept!(E_new, E_old)")