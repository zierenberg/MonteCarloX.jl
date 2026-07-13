# # Multicanonical Sampling of a Lennard-Jones Gas
#
# We use multicanonical sampling in the total energy to drive a dilute Lennard-Jones
# gas through the condensation–evaporation transition. At subcritical temperatures
# and low density the system shows a first-order-like droplet transition: a canonical
# energy histogram is bimodal with a suppressed barrier region. MUCA flattens this
# histogram, enabling efficient sampling of both the gas and droplet phases.
#
# Reference: Zierenberg & Janke, Phys. Rev. E (2015).

using Random, StatsBase, Plots, DelimitedFiles
using MonteCarloX, MCXSoftMatter
import MCXSoftMatter: translate!

datadir   = get(ENV, "MCX_EXAMPLE_DATA", normpath(joinpath(@__DIR__, "..", "..", "docs", "src", "data")))  # hide
hist_file = joinpath(datadir, "muca_LJgas_histogram.tsv")     # hide
lw_file   = joinpath(datadir, "muca_LJgas_logweight.tsv")     # hide
diag_file = joinpath(datadir, "muca_LJgas_diagnostics.tsv")   # hide

N              = 16
rho            = 0.01
e_min, e_max   = -3.3, 0.0     # per-particle energy bounds (prior knowledge)
T_max, T_min   = 1.0, 0.5
num_iter       = 20
sweeps_equil   = 10_000
sweeps_measure = 1_000_000
dE             = 0.1
nothing # hide

# ## System and sweep
#
# A 3D Lennard-Jones gas with a truncated-shifted potential at ``r_{\rm cut}=2.5\sigma``.
# One sweep proposes a short local translation of every particle plus one long
# translation (to help particles escape the droplet).

r0    = 1.0
sigma = r0 / 2^(1/6)
lj    = LennardJonesPotential(epsilon = 1.0, sigma = sigma, r_cutoff = 2.5 * sigma)

function sweep!(sys, alg, dx_short, dx_long)
    for _ in 1:num_particles(sys)
        translate!(sys, alg, dx_short)
    end
    translate!(sys, alg, dx_long)
    return nothing
end

# ## Multicanonical iteration
#
# We set up a MUCA ensemble over the energy range (with a buffer for boundary
# suppression) and initialise the weights to a ``T_{\max}`` canonical ramp. Each
# iteration thermalizes, records the energy histogram, and refines the weights with
# the recursive scheme, reapplying temperature ramps outside the target range and
# tracking acceptance, flatness, and round trips.

if !isfile(lw_file)                                            # hide
sys = ParticleGas(; N = N, rho = rho, pair_potential = lj)
init!(sys, :random; rng = Xoshiro(42))
dx_short = 0.1
dx_long  = mean(sys.env.L) / 4

E_min, E_max = e_min * N, e_max * N
alg = Multicanonical(Xoshiro(42), E_min - N:dE:E_max + N; warn_overwrite = false)
ens = ensemble(alg)
rt  = Roundtrips(E_min, E_max)
set_logweight!(ens, E -> -E / T_max)

E = collect(get_centers(ens.histogram))
H = zeros(length(E), num_iter)
W = zeros(length(E), num_iter)
acceptrate    = zeros(num_iter)
flatness_log  = zeros(num_iter)
roundtrip_log = zeros(num_iter)

for iter in 1:num_iter
    for _ in 1:sweeps_equil; sweep!(sys, alg, dx_short, dx_long); end
    reset!(alg)
    reset!(rt)
    for _ in 1:round(Int, sweeps_measure / num_iter * iter)
        sweep!(sys, alg, dx_short, dx_long)
        update!(rt, energy(sys))
    end
    update_logweight!(ens; mode = :recursive)
    extend!(ens, :high; anchor = E_max, slope = -1 / T_max)   # temperature ramps outside range
    extend!(ens, :low;  anchor = E_min, slope = -1 / T_min)
    H[:, iter]          = ens.histogram.values
    W[:, iter]          = ens.logweight.values
    acceptrate[iter]    = acceptance_rate(alg)
    flatness_log[iter]  = flatness(ens.histogram, E_min, E_max)
    roundtrip_log[iter] = rt.count
end

header = permutedims(["E"; ["iter$(it)" for it in 1:num_iter]])   # hide
mkpath(datadir)                                                   # hide
writedlm(hist_file, [header; hcat(E, H)], '\t')                   # hide
writedlm(lw_file,   [header; hcat(E, W)], '\t')                   # hide
writedlm(diag_file, ["iter" "acceptrate" "flatness" "roundtrips"; # hide
                     hcat(1:num_iter, acceptrate, flatness_log, roundtrip_log)], '\t')  # hide
end                                                                # hide
hh = readdlm(hist_file, '\t'; header = true)[1]                   # hide
ll = readdlm(lw_file,   '\t'; header = true)[1]                   # hide
dd = readdlm(diag_file, '\t'; header = true)[1]                   # hide
E = hh[:, 1]; H = hh[:, 2:end]; W = ll[:, 2:end]                  # hide
acceptrate = dd[:, 2]; flatness_log = dd[:, 3]; roundtrip_log = dd[:, 4]  # hide
num_iter = size(H, 2)                                             # hide
nothing # hide

# ## Convergence
#
# Acceptance rate, histogram flatness (`max/mean`, → 1 when flat), and round trips
# through the energy range.

p1 = plot(acceptrate; xlabel = "iteration", ylabel = "acceptance rate",
          label = nothing, ylims = (0, 1))
p2 = plot(flatness_log; xlabel = "iteration", ylabel = "flatness (max/mean)",
          label = nothing, yscale = :log10)
hline!(p2, [2.0]; ls = :dash, color = :gray, label = "threshold")
p3 = plot(roundtrip_log; xlabel = "iteration", ylabel = "round trips", label = nothing)
plot(p1, p2, p3; layout = (1, 3), size = (980, 260), margin = 4Plots.mm)

# ## Histograms and log-weights
#
# Converged iterations show a flat energy histogram spanning from the dilute gas
# (``E \approx 0``) to the condensed droplet (``E \ll 0``).

cols = palette(:viridis, max(num_iter, 2))
EN   = E ./ N

ph = plot(xlabel = "E/N", ylabel = "counts", title = "LJ gas MUCA (N = $N)", legend = false)
for it in 1:num_iter
    plot!(ph, EN, H[:, it]; lw = 2, color = cols[it])
end
pw = plot(xlabel = "E/N", ylabel = "log w", title = "LJ gas MUCA (N = $N)", legend = false)
for it in 1:num_iter
    plot!(pw, EN, W[:, it] .- W[1, it]; lw = 2, color = cols[it])
end
plot(ph, pw; layout = (1, 2), size = (960, 320), margin = 4Plots.mm)
