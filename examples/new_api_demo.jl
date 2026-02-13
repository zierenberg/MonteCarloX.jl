#!/usr/bin/env julia
"""
Demonstration of the new MonteCarloX API with SpinSystems.

This example shows:
1. Ising model simulation
2. Blume-Capel model simulation
3. New measurement framework
4. Comparison of different systems with the same algorithm

Based on the API developed in examples/api.ipynb
"""

using Random
using StatsBase

# Add the package to load path (for development)
# In production, just use: using MonteCarloX
push!(LOAD_PATH, @__DIR__ * "/../src")
using MonteCarloX
using MonteCarloX.SpinSystems

function run_ising_simulation()
    println("="^70)
    println("Ising Model Simulation")
    println("="^70)
    
    # Setup
    rng = MersenneTwister(42)
    L = 8
    β = 0.4  # Inverse temperature
    
    println("\nConfiguration:")
    println("  Lattice: $(L)×$(L)")
    println("  β = $β")
    println("  Total sites: $(L*L)")
    
    # Create system
    sys = Ising([L, L], J=1, periodic=true)
    init!(sys, :random, rng=rng)
    
    println("  Initial energy: $(energy(sys))")
    println("  Initial magnetization: $(magnetization(sys))")
    
    # Create algorithm
    alg = Metropolis(rng, β=β)
    
    # Setup measurements
    measurements = Measurements([
        :energy => energy => Float64[],
        :magnetization => magnetization => Float64[]
    ], interval=10)
    
    # Thermalization
    println("\nThermalization...")
    N = L * L
    for i in 1:N*1000
        spin_flip!(sys, alg)
    end
    
    reset_statistics!(alg)
    
    # Production
    println("Production run...")
    for i in 1:N*10000
        spin_flip!(sys, alg)
        measure!(measurements, sys, i)
    end
    
    # Results
    energies = measurements[:energy].data
    mags = measurements[:magnetization].data
    
    println("\nResults:")
    println("  Acceptance rate: $(round(acceptance_rate(alg), digits=4))")
    println("  ⟨E⟩/N = $(round(mean(energies)/N, digits=4))")
    println("  ⟨|M|⟩/N = $(round(mean(mags)/N, digits=4))")
    println("  σ(E)/N = $(round(std(energies)/N, digits=4))")
    
    return measurements
end

function run_blume_capel_simulation()
    println("\n" * "="^70)
    println("Blume-Capel Model Simulation")
    println("="^70)
    
    # Setup
    rng = MersenneTwister(123)
    L = 8
    β = 0.4
    D = 0.5  # Crystal field
    
    println("\nConfiguration:")
    println("  Lattice: $(L)×$(L)")
    println("  β = $β")
    println("  D = $D (crystal field)")
    println("  Total sites: $(L*L)")
    
    # Create system
    sys = BlumeCapel([L, L], J=1, D=D, periodic=true)
    init!(sys, :random, rng=rng)
    
    println("  Initial energy: $(energy(sys))")
    println("  Initial magnetization: $(magnetization(sys))")
    
    # Create algorithm (same as Ising!)
    alg = Metropolis(rng, β=β)
    
    # Setup measurements
    measurements = Measurements([
        :energy => energy => Float64[],
        :magnetization => magnetization => Float64[]
    ], interval=10)
    
    # Thermalization
    println("\nThermalization...")
    N = L * L
    for i in 1:N*1000
        spin_flip!(sys, alg)
    end
    
    reset_statistics!(alg)
    
    # Production
    println("Production run...")
    for i in 1:N*10000
        spin_flip!(sys, alg)
        measure!(measurements, sys, i)
    end
    
    # Results
    energies = measurements[:energy].data
    mags = measurements[:magnetization].data
    
    println("\nResults:")
    println("  Acceptance rate: $(round(acceptance_rate(alg), digits=4))")
    println("  ⟨E⟩/N = $(round(mean(energies)/N, digits=4))")
    println("  ⟨|M|⟩/N = $(round(mean(mags)/N, digits=4))")
    println("  σ(E)/N = $(round(std(energies)/N, digits=4))")
    
    return measurements
end

function main()
    println("\n" * "="^70)
    println("MonteCarloX.jl - New API Demonstration")
    println("="^70)
    println("\nThis example demonstrates the new API design where:")
    println("  • Systems (Ising, BlumeCapel) are in SpinSystems submodule")
    println("  • Algorithms (Metropolis) are in MonteCarloX core")
    println("  • Measurements use the new framework from api.ipynb")
    println("  • The same algorithm works with different systems!")
    
    # Run simulations
    ising_results = run_ising_simulation()
    bc_results = run_blume_capel_simulation()
    
    println("\n" * "="^70)
    println("Summary")
    println("="^70)
    println("\nKey features of the new API:")
    println("  ✓ Clean separation: MonteCarloX (algorithms) + SpinSystems (models)")
    println("  ✓ Flexible measurements with interval/preallocated schedules")
    println("  ✓ Reusable algorithms across different systems")
    println("  ✓ No system definitions in MonteCarloX core")
    println("  ✓ Based on proven design from api.ipynb")
    
    println("\n" * "="^70)
    println("Success! The new API is working.")
    println("="^70)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
