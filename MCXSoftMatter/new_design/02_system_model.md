# ParticleSystem — Design

## Scope

| In scope | Future | Out of scope |
|---|---|---|
| Noble gas, simple fluids | Lipid membranes | Atomistic proteins |
| Homopolymers | Explicit-solvent mixtures | Rigid molecules |
| Diblock / heteropolymers | Colloids | |
| | `HardWallBox` environment | |

Lattice polymers and HP models → `MCXLatticeMatter`.

---

## Molecule Types

```julia
abstract type AbstractMolecule end
```

### Monatomic — gas particles

Zero-size type. `Vector{Monatomic}(undef, N)` costs only the array header (~0 bytes
per element). All intramolecular methods compile away.

```julia
struct Monatomic <: AbstractMolecule end

particle_range(::Monatomic, i::Int) = i:i
is_bonded_neighbor(::Monatomic, ::Monatomic) = false
molecule_energy(::Monatomic, positions, env) = 0.0   # compiled away
```

### Polymer — a single chain

Each `Polymer` is a lightweight descriptor into the flat `positions` array.
It owns its intramolecular potentials and topology, but not the positions themselves.

```julia
struct Polymer{TBond, TBend} <: AbstractMolecule
    offset::Int              # 0-based start index into positions
    length::Int              # number of monomers
    bond::TBond
    bend::TBend
end

particle_range(p::Polymer) = p.offset+1 : p.offset+p.length
```

Bond exclusion uses flat indices — requires a reverse lookup (see Core Type below):

```julia
# 1-2 exclusion only (correct for CG models)
@inline function is_bonded_neighbor(sys, i, j)
    mol_i = sys.molecule_id[i]
    mol_i != sys.molecule_id[j] && return false
    abs(sys.monomer_k[i] - sys.monomer_k[j]) == 1
end
```

### NoBend — zero-cost bending placeholder

```julia
struct NoBend end
@inline bend_energy(::NoBend, positions, env) = 0.0   # compiled away
```

### Molecule energy

Intramolecular energy is cached on the system, not on individual molecules.
See Energy Cache below.

---

## Energy Cache

The cache type mirrors the molecule type — fields exist only when relevant.

```julia
mutable struct CacheMonatomic{T}
    pair::T
end

mutable struct CachePolymer{T}
    pair::T
    bond::T
    bend::T
end

@inline total_energy(c::CacheMonatomic) = c.pair
@inline total_energy(c::CachePolymer) = c.pair + c.bond + c.bend
```

- `CacheMonatomic` — gas systems: only pair energy, no bond/bend fields at all.
- `CachePolymer` — polymer systems: pair + bond + bend, each accessible individually.
- Named constructors wire the correct cache type automatically.
- All caches are O(1) access — single values, not per-molecule sums.

---

## Core Type

```julia
mutable struct ParticleSystem{D, T<:AbstractFloat,
                               Env <: AbstractEnvironment{D,T},
                               TPair,
                               TMol <: AbstractMolecule,
                               TCache,
                               Accel}
    env::Env
    positions::Vector{SVector{D,T}}
    molecules::Vector{TMol}
    molecule_id::Vector{Int}    # molecule_id[i] = which molecule owns particle i
    monomer_k::Vector{Int}      # monomer_k[i]   = position within its molecule
    pair::TPair                 # SinglePotential | PairTable{S}
    cache::TCache               # CacheMonatomic{T} | CachePolymer{T}
    accel::Accel                # CellList{D,K} | NoCellList
end
```

### Design decisions

- **`molecules` is always a vector** — `Vector{Monatomic}` for gas (zero-size elements),
  `Vector{Polymer{...}}` for polymer solutions. Uniform interface for MC move selection:
  `rand(rng, 1:length(molecules))`.

- **Positions stay flat** — one contiguous `Vector{SVector{D,T}}` for all particles.
  Critical for pair energy performance and cell list compatibility. Polymers are
  descriptors (offset + length) into this flat array.

- **Per-particle reverse lookups** (`molecule_id`, `monomer_k`) — needed for O(1)
  bond exclusion in the pair energy loop. For `Monatomic`, `molecule_id[i] = i` and
  `monomer_k[i] = 1` (trivial, but uniform). These could be omitted for pure gas
  systems via a wrapper type, but the overhead is negligible.

- **No `species::Vector{Int}`** — species dispatch lives in `TPair`. `SinglePotential`
  ignores species arguments entirely (zero-cost for homogeneous systems).
  `PairTable` carries species IDs internally. See Pair Potentials below.

- **`accel` on system, not environment** — cell list is mutable state coupled to
  positions. Environment stays stateless (pure metric + boundary).

- **Energy cache dispatches on molecule type** — `CacheMonatomic` for gas (pair only),
  `CachePolymer` for polymers (pair + bond + bend). Cache type is a parameter on the
  system, wired automatically by named constructors. Each energy component is a single
  scalar, O(1) access.

### Total energy

```julia
@inline total_energy(sys) = total_energy(sys.cache)
```

---

## Pair Potentials

```julia
# One species — noble gas, homopolymer
struct SinglePotential{TPot}
    potential::TPot
end
@inline (sp::SinglePotential)(i, j, r_sq) = sp.potential(r_sq)  # ignores i, j
```

Uniform calling convention: always `pair(i, j, r_sq)`. No two-argument form.

```julia
# S species — diblock, heteropolymer; stack-allocated for S ≤ ~10
struct PairTable{S, TPot}
    species::Vector{Int}                    # species[i] = species of particle i
    table::NTuple{S, NTuple{S, TPot}}
end
@inline function (pt::PairTable)(i::Int, j::Int, r_sq)
    @inbounds pt.table[pt.species[i]][pt.species[j]](r_sq)
end
```

Species IDs live on `PairTable`, not on `ParticleSystem`. This means:
- `SinglePotential` path has **zero species overhead** — no vector, no lookup
- `PairTable` owns its species data — self-contained, no coupling to system
- The inner loop signature is uniform: `pair(i, j, r_sq)` for both

`PairTable{1,LJ}` compiles identically to `SinglePotential{LJ}`. Lorentz-Berthelot
mixing rules available as a `PairTable{S}(pot; ε, σ)` constructor convenience.

---

## Pair Energy Inner Loop

One function, one cell-list dispatch. Bond exclusion compiles away for `Monatomic`.

```julia
# Full pair energy for particle idx (with bond exclusion)
@inline function _local_pair_energy(sys::ParticleSystem, idx::Int)
    _pair_energy_loop(sys, idx, Val(true))
end

# Pair energy without bond exclusion (for rigid-chain moves where bonds don't change)
@inline function _local_pair_energy_no_excl(sys::ParticleSystem, idx::Int)
    _pair_energy_loop(sys, idx, Val(false))
end

@inline function _pair_energy_loop(sys::ParticleSystem{D,T}, idx::Int,
                                    ::Val{exclude}) where {D,T,exclude}
    E = zero(T)
    pos_i = sys.positions[idx]
    @inbounds for j in _neighbor_indices(sys.accel, sys.env, idx)
        j == idx && continue
        exclude && is_bonded_neighbor(sys, idx, j) && continue
        r_sq = sum(abs2, difference(sys.env, pos_i, sys.positions[j]))
        E += sys.pair(idx, j, r_sq)
    end
    return E
end
```

Specialization:
- `Monatomic` → `is_bonded_neighbor` returns `false`, branch eliminated
- `SinglePotential` → `pair(i, j, r_sq)` ignores `i,j`, inlines to `potential(r_sq)`
- `NoCellList` → `_neighbor_indices` returns `1:N`
- `CellList{D,K}` → `_neighbor_indices` iterates neighbor cells

`Val{exclude}` const-propagates reliably with `@inline`. Benchmark to confirm;
fall back to two separate functions only if measurable difference.

This replaces the current 5 separate pair-energy functions.

---

## Named Constructors

```julia
noble_gas(; D=3, N, L, pair_potential, T=Float64)
# → ParticleSystem{..., Monatomic, CacheMonatomic{T}, auto}

polymer_solution(; D=3, num_chains, chain_length, L,
                   pair_potential, bond_potential, bend_potential=NoBend(), T=Float64)
# → ParticleSystem{..., Polymer{TBond,TBend}, CachePolymer{T}, auto}

heteropolymer_solution(; D=3, num_chains, chain_length, species_sequence,
                         pair_table, bond_potential, bend_potential=NoBend(), L, T=Float64)
# → ParticleSystem{..., Polymer{TBond,TBend}, CachePolymer{T}, auto}
```

`auto` = cell list chosen automatically from cutoff and N.

Generic constructor for power users:

```julia
ParticleSystem(; env, positions, molecules, pair, accel=auto)
# Computes molecule_id, monomer_k, initial energies automatically.
```

---

## Move Dispatch

```julia
translate!(sys::ParticleSystem, alg, max_disp)                    # any molecule type
pivot!(sys::ParticleSystem{D,T,E,P,<:Polymer}, alg)               # Polymer only
reptate!(sys::ParticleSystem{D,T,E,P,<:Polymer}, alg)             # Polymer only
```

`pivot!` on a system with `Monatomic` molecules is a compile-time `MethodError`.

### translate! — single function for all variants

```julia
function translate!(sys::ParticleSystem, alg, Δ; chain::Bool=false)
    if chain
        _translate_chain!(sys, alg, Δ)
    else
        _translate_molecule!(sys, alg, Δ)
    end
end
```

`_translate_molecule!` picks a random molecule, then:
- `Monatomic` → single particle move, pair energy only (bond/bend compiled away)
- `Polymer` → single monomer move within the chain, pair + bond + bend energy

`_translate_chain!` moves all particles of a randomly selected molecule.
For `Monatomic` this is equivalent to a single-particle move.

---

## Usage

```julia
# Noble gas
sys = noble_gas(N=500, L=10.0, pair_potential=LennardJones(ε=1.0, σ=1.0))
init!(sys, :random; rng)
alg = Metropolis(rng; β=1.0)
meas = Measurements([:energy => total_energy => Float64[]], interval=100)
for step in 1:100_000
    translate!(sys, alg, 0.2)
    measure!(meas, sys, step)
end

# Homopolymer melt
sys = polymer_solution(num_chains=20, chain_length=50, L=30.0,
    pair_potential=LennardJones(ε=1.0, σ=1.0),
    bond_potential=FENE(k=30.0, r_max=1.5),
    bend_potential=CosineBend(κ=3.0))
init!(sys, :random_walk; rng)
for step in 1:500_000
    translate!(sys, alg, 0.1)
    pivot!(sys, alg)
end

# Diblock A-B copolymer
seq = [k ≤ 25 ? 1 : 2 for k in 1:50]
sys = heteropolymer_solution(num_chains=20, chain_length=50, species_sequence=seq, L=30.0,
    pair_table=PairTable{2}(LennardJones; ε=[1.0 0.5; 0.5 1.0], σ=1.0),
    bond_potential=FENE(k=30.0, r_max=1.5))
init!(sys, :random_walk; rng)
```

---

## Resolved Questions

1. **`Val{exclude}` propagation**: `@inline` ensures const-propagation. Well-established
   Julia pattern. Benchmark to confirm; fall back to two functions only if needed.

2. **`PairTable` owning species**: clean separation. For fixed-N simulations (our scope),
   no sync issues. Grand-canonical would need coordinated updates — future concern.

3. **`_neighbor_indices` return type**: `NoCellList` returns `UnitRange{Int}`,
   `CellList` returns a custom iterator. Both are concrete types known at compile time
   through dispatch on `Accel`. No dynamic dispatch.
