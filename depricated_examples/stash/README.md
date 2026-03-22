# Stashed Examples

This directory contains legacy examples from the old MonteCarloX API. These are kept for reference purposes but may not work with the current API.

## Contents

- `Ising.jl` - Old Ising model example
- `contact_process.jl` - Contact process simulation
- `gillespie_sir_hetero_rates.jl` - SIR model with Gillespie algorithm
- `glif-net.jl` - Neural network example
- `hawkes_process.jl` - Hawkes process simulation
- `ising_muca.jl` - Ising model with multicanonical sampling
- `nonequil_muca_sir.jl` - Non-equilibrium SIR with multicanonical

## Status

⚠️ **These examples use the old API and are not maintained.**

For working examples with the new API, see the `notebooks/` directory:
- `notebooks/simple_ising.jl` - Basic Ising model example
- `notebooks/new_api_demo.jl` - Comprehensive demonstration
- `notebooks/api.ipynb` - API development notebook

## Migration

If you need to use these examples:
1. Refer to `docs/MIGRATION_GUIDE.md` for how to update to the new API
2. Check the working examples in `notebooks/` for the new patterns
3. Consult `PROJECT_OVERVIEW.md` for the current architecture

## Future

These examples may be:
- Migrated to the new API and moved to `notebooks/`
- Removed if no longer relevant
- Kept as historical reference
