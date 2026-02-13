# Migration Guide: Old API → New API

This guide helps you transition from the old MonteCarloX API to the new design.

## Overview of Changes

### Old API
- Systems and algorithms mixed in core
- Limited type structure
- Ad-hoc measurements
- Function-based approach

### New API  
- Clean separation: algorithms in core, systems in submodules
- Rich type hierarchy
- Structured measurement framework
- Object-oriented approach with types

## Key Differences

### 1. System Definitions

**Old**: Systems were scattered or mixed with examples
```julia
# Systems were not consistently organized
```

**New**: Systems in dedicated submodules
```julia
using MonteCarloX
using MonteCarloX.SpinSystems

sys = Ising([8, 8], J=1, periodic=true)
init!(sys, :random, rng=rng)
```

### 2. Algorithm Setup

**Old**: Algorithm types with limited state
```julia
alg = Metropolis()
```

**New**: Algorithms hold state and parameters
```julia
alg = Metropolis(rng, β=0.4)
# or
alg = Metropolis(rng, logweight=custom_weight)
```

### 3. Acceptance Testing

**Old**: Functional approach
```julia
if accept(Metropolis(), rng, β, ΔE)
    # apply change
end
```

**New**: Object-oriented with tracking
```julia
log_ratio = alg.logweight(ΔE)
if accept!(alg, log_ratio)
    modify!(sys, i, ΔE)
end
```

### 4. Measurements

**Old**: Manual tracking
```julia
energy_data = Float64[]
for i in 1:N
    update!(sys, alg)
    if i % interval == 0
        push!(energy_data, energy(sys))
    end
end
```

**New**: Structured measurement framework
```julia
measurements = Measurements([
    :energy => energy => Float64[]
], interval=10)

for i in 1:N
    update!(sys, alg)
    measure!(measurements, sys, i)
end

# Access results
energy_data = measurements[:energy].data
```

## Common Migration Patterns

### Pattern 1: Simple Metropolis Simulation

**Old Style:**
```julia
using Random
using MonteCarloX

# Setup (system definition unclear)
rng = MersenneTwister(42)
β = 0.4

# Run
energy_data = Float64[]
for i in 1:10000
    # update with accept()
    if accept(Metropolis(), rng, β, ΔE)
        # apply update
    end
    
    # manual measurement
    if i % 10 == 0
        push!(energy_data, calculate_energy())
    end
end

# Analysis
mean_E = sum(energy_data) / length(energy_data)
```

**New Style:**
```julia
using Random
using MonteCarloX
using MonteCarloX.SpinSystems

# Setup
rng = MersenneTwister(42)
sys = Ising([8, 8], J=1, periodic=true)
init!(sys, :random, rng=rng)
alg = Metropolis(rng, β=0.4)

# Configure measurements
measurements = Measurements([
    :energy => energy => Float64[]
], interval=10)

# Run
for i in 1:10000
    spin_flip!(sys, alg)
    measure!(measurements, sys, i)
end

# Analysis
mean_E = mean(measurements[:energy].data)
acc_rate = acceptance_rate(alg)
```

**Benefits:**
- Clearer system definition
- Automatic measurement handling
- Built-in statistics tracking
- Type-safe operations

### Pattern 2: Multiple Observables

**Old Style:**
```julia
energy_data = Float64[]
mag_data = Float64[]

for i in 1:N
    update!()
    if i % interval == 0
        push!(energy_data, calc_energy())
        push!(mag_data, calc_mag())
    end
end
```

**New Style:**
```julia
measurements = Measurements([
    :energy => energy => Float64[],
    :magnetization => magnetization => Float64[]
], interval=10)

for i in 1:N
    spin_flip!(sys, alg)
    measure!(measurements, sys, i)
end

# Easy access
E = measurements[:energy].data
M = measurements[:magnetization].data
```

### Pattern 3: Custom Observable

**Old Style:**
```julia
custom_data = Float64[]
for i in 1:N
    update!()
    if i % interval == 0
        value = custom_calculation(sys)
        push!(custom_data, value)
    end
end
```

**New Style:**
```julia
measurements = Measurements([
    :custom => (sys -> custom_calculation(sys)) => Float64[]
], interval=10)

for i in 1:N
    spin_flip!(sys, alg)
    measure!(measurements, sys, i)
end
```

### Pattern 4: Different Systems, Same Algorithm

**Old Style:**
```julia
# Needed separate code for each system type
```

**New Style:**
```julia
rng = MersenneTwister(42)
alg = Metropolis(rng, β=0.4)

# Works with Ising
sys1 = Ising([8, 8])
init!(sys1, :random, rng=rng)
for i in 1:1000
    spin_flip!(sys1, alg)
end

# Works with BlumeCapel
sys2 = BlumeCapel([8, 8], D=0.5)
init!(sys2, :random, rng=rng)
reset_statistics!(alg)
for i in 1:1000
    spin_flip!(sys2, alg)
end
```

## Checklist for Migration

When migrating code, follow these steps:

1. **Identify Systems**
   - [ ] Find system definitions in old code
   - [ ] Check if system is in SpinSystems
   - [ ] If not, implement as new AbstractSystem

2. **Update Algorithm Creation**
   - [ ] Change from `Metropolis()` to `Metropolis(rng, β=β)`
   - [ ] Move weight function into algorithm or use BoltzmannLogWeight

3. **Convert Measurements**
   - [ ] Replace manual arrays with Measurements
   - [ ] Use interval or preallocated schedule
   - [ ] Update analysis code to access .data

4. **Update Acceptance Logic**
   - [ ] Change from `accept()` function to `accept!()` method
   - [ ] Use algorithm's logweight
   - [ ] Let algorithm track statistics

5. **Test Equivalence**
   - [ ] Run both versions with same RNG seed
   - [ ] Compare final statistics
   - [ ] Verify acceptance rates match

## Creating New Systems

If your code uses a custom system not in SpinSystems:

### 1. Define System Type
```julia
mutable struct MySystem <: AbstractSystem
    state::Vector{Float64}
    # ... other fields
end
```

### 2. Implement Observables
```julia
my_energy(sys::MySystem) = sum(sys.state.^2) / 2
```

### 3. Implement Update
```julia
function my_update!(sys::MySystem, alg::Metropolis)
    i = rand(alg.rng, 1:length(sys.state))
    old_val = sys.state[i]
    new_val = old_val + (2rand(alg.rng) - 1) * 0.5
    
    ΔE = (new_val^2 - old_val^2) / 2
    log_ratio = alg.logweight(ΔE)
    
    if accept!(alg, log_ratio)
        sys.state[i] = new_val
    end
end
```

### 4. Use It
```julia
sys = MySystem(randn(100))
alg = Metropolis(rng, β=1.0)

measurements = Measurements([
    :energy => my_energy => Float64[]
], interval=10)

for i in 1:10000
    my_update!(sys, alg)
    measure!(measurements, sys, i)
end
```

## Common Pitfalls

### Pitfall 1: Forgetting to Initialize System
**Wrong:**
```julia
sys = Ising([8, 8])
spin_flip!(sys, alg)  # Spins not initialized!
```

**Correct:**
```julia
sys = Ising([8, 8])
init!(sys, :random, rng=rng)
spin_flip!(sys, alg)
```

### Pitfall 2: Not Resetting Statistics
**Wrong:**
```julia
# Thermalization
for i in 1:1000
    spin_flip!(sys, alg)
end

# Production (includes thermalization in acceptance!)
for i in 1:10000
    spin_flip!(sys, alg)
end
println(acceptance_rate(alg))  # Wrong!
```

**Correct:**
```julia
# Thermalization
for i in 1:1000
    spin_flip!(sys, alg)
end

reset_statistics!(alg)

# Production
for i in 1:10000
    spin_flip!(sys, alg)
end
println(acceptance_rate(alg))  # Correct!
```

### Pitfall 3: Accessing Measurement Data Incorrectly
**Wrong:**
```julia
measurements = Measurements([...], interval=10)
# ...
data = measurements[:energy]  # This is a Measurement object!
```

**Correct:**
```julia
measurements = Measurements([...], interval=10)
# ...
data = measurements[:energy].data  # This is the Vector!
```

## Getting Help

If you encounter issues during migration:

1. Check `examples/new_api_demo.jl` for complete working examples
2. Read `docs/API_DESIGN.md` for detailed explanations
3. Look at SpinSystems implementations for reference
4. Open an issue on GitHub with your specific case

## Benefits of Migration

Migrating to the new API provides:

- ✅ Cleaner, more maintainable code
- ✅ Better type safety and compiler optimizations
- ✅ Easier testing and debugging
- ✅ Automatic measurement handling
- ✅ Built-in statistics tracking
- ✅ Reusable algorithms across systems
- ✅ Clear separation of concerns
- ✅ Extensibility for future features

## Timeline

The new API is available now on this branch. The old API is maintained for backward compatibility but may be deprecated in future releases. We recommend:

- **New projects**: Use new API from the start
- **Existing projects**: Migrate when convenient
- **Critical code**: Test thoroughly during migration
