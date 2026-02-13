#!/usr/bin/env julia
"""
Simple Ising model example using the new MonteCarloX API.

This example demonstrates:
1. Creating an Ising model (from SpinSystems submodule)
2. Setting up a Metropolis algorithm with Boltzmann weights
3. Configuring measurements with the new measurement framework
4. Running equilibrium Monte Carlo simulation
5. Analyzing results
"""

using Random
using StatsBase

# Load MonteCarloX with the new API
push!(LOAD_PATH, @__DIR__ * "/../src")
using MonteCarloX
using MonteCarloX.SpinSystems

function main()
    println("="^60)
    println("Ising Model Simulation - New API")
    println("="^60)
    
    # Setup
    rng = MersenneTwister(42)
    L = 8
    β = 0.4  # Inverse temperature
    
    println("\nSystem configuration:")
    println("  Lattice size: $(L)×$(L)")
    println("  Inverse temperature β: $β")
    println("  Total spins: $(L*L)")
    
    # Create Ising system
    sys = Ising([L, L], J=1, periodic=true)
    init!(sys, :random, rng=rng)
    
    # Create Metropolis algorithm with Boltzmann weight
    alg = Metropolis(rng, β=β)
    
    # Setup measurements
    measurements = Measurements([
        :energy => energy => Float64[],
        :magnetization => magnetization => Float64[]
    ], interval=10)
    
    # Thermalization
    println("\nThermalizing...")
    N = L * L
    for i in 1:N*1000
        spin_flip!(sys, alg)
    end
    println("  Thermalization complete: $(alg.steps) steps")
    
    # Reset statistics after thermalization
    reset_statistics!(alg)
    
    # Production run
    println("\nProduction run...")
    for i in 1:N*10000
        spin_flip!(sys, alg)
        measure!(measurements, sys, i)
    end
    
    # Results
    println("\n" * "="^60)
    println("Results:")
    println("="^60)
    println("  Total steps: $(alg.steps)")
    println("  Acceptance rate: $(round(acceptance_rate(alg), digits=4))")
    
    energies = measurements[:energy].data
    mags = measurements[:magnetization].data
    
    avg_E = mean(energies) / N
    avg_M = mean(mags) / N
    
    println("\nObservables (per spin):")
    println("  Average energy: $(round(avg_E, digits=4))")
    println("  Average magnetization: $(round(avg_M, digits=4))")
    
    println("\n" * "="^60)
    println("Simulation completed successfully!")
    println("="^60)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
