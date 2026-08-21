import Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()

using Random, Printf, Sunny

# CP2 skyrmion quench using Sunny's Langevin (SLL) integrator.
# See: https://sunnysuite.github.io/Sunny.jl/stable/examples/06_CP2_Skyrmions.html
#
# MCX gap analysis — what would be needed for an MCX bridge:
#
#   1. State space: The model uses :SUN mode (SU(3) coherent states on CP²).
#      MCX has IsingSystem (Z₂) and HeisenbergSystem (S²/SU(2)). A CP2System
#      (SU(3)/U(2), 4-real-dimensional manifold) does not exist in MCXSpins.
#
#   2. Dynamics: Sunny.Langevin implements the stochastic Landau-Lifshitz (SLL)
#      equation on CP^{N-1}. There is no accept/reject step. MCX algorithms
#      (Metropolis, Glauber, HeatBath) all require ΔE and a Bool decision.
#      A new abstract type AbstractSDEIntegrator + advance! would be needed.
#
#   3. For Heisenberg (S²) specifically: a LangevinSpin algorithm is feasible
#      in MCX — the SLL update is just cross products + normalization. CP²
#      requires SU(3) algebra and exponential maps on the manifold, which is
#      out of scope without significant new infrastructure.

const SEED = 42
L = 40

latvecs = Sunny.lattice_vectors(1, 1, 10, 90, 90, 120)
cryst = Sunny.Crystal(latvecs, [[0, 0, 0]])
sys = Sunny.System(cryst, [1 => Sunny.Moment(s=1, g=-1)], :SUN; dims=(L, L, 1))

J1, J2, Δ = -1.0, 2.0 / (1 + sqrt(5.0)), 2.6
Sunny.set_exchange!(sys, J1 * [1 0 0; 0 1 0; 0 0 Δ], Sunny.Bond(1, 1, [1, 0, 0]))
Sunny.set_exchange!(sys, J2 * [1 0 0; 0 1 0; 0 0 Δ], Sunny.Bond(1, 1, [1, 2, 0]))
Sunny.set_field!(sys, [0, 0, 15.5])
Sunny.set_onsite_coupling!(sys, S -> 19.0 * S[3]^2, 1)

integrator = Sunny.Langevin(; damping=0.05, kT=0.0)
copy!(sys.rng, Xoshiro(SEED))
Sunny.randomize_spins!(sys)
integrator.dt = 0.01

τs = [4.0, 16.0, 256.0]
energies = Float64[]
ts_wall = Float64[]

t = 0.0
for (i, τ) in enumerate(τs)
    nsteps = round(Int, (τ - t) / integrator.dt)
    elapsed = @elapsed for _ in 1:nsteps
        Sunny.step!(sys, integrator)
    end
    push!(energies, Sunny.energy_per_site(sys))
    push!(ts_wall, elapsed)
    t = τ
end

println("CP2 skyrmion quench  (L=$(L), dt=$(integrator.dt))")
println("τ           e/site       wall(s)")
for i in eachindex(τs)
    @printf "%-10.1f  %12.6f  %8.3f\n" τs[i] energies[i] ts_wall[i]
end
println()
println("MCX bridge: not applicable — Langevin has no accept/reject.")
println("  Heisenberg LangevinSpin on S²: feasible addition to MCX")
println("  CP2 LangevinSpin on CP²:        requires SUNSystem + SU(3) algebra")
