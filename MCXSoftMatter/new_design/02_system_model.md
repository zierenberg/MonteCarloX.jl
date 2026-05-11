# ParticleSystem — Design

## Scope

| In scope | Future | Out of scope |
|---|---|---|
| Noble gas, simple fluids | Lipid membranes | Atomistic proteins |
| Homopolymers | Explicit-solvent mixtures | Rigid molecules |
| Diblock / heteropolymers | `HardWallBox` environment | |

Lattice polymers and HP models → `MCXLatticeMatter`.

---

## Core Type

```julia
mutable struct ParticleSystem{D, T<:AbstractFloat,
                               Env <: AbstractEnvironment{D,T},
                               TPair,
                               Mol} <: AbstractSoftMatterSystem
    env::Env
    positions::Vector{SVector{D,T}}
    species::Vector{Int}    # 1..S; all-ones for homogeneous systems
    pair::TPair             # SinglePotential | PairTable{S}
    molecule::Mol           # Monatomic | LinearChain{TBond,TBend}
    cached_energy::T
end
```

---

## Pair Potentials

```julia
# One species — noble gas, homopolymer
struct SinglePotential{TPot}; potential::TPot; end
@inline (sp::SinglePotential)(si, sj, r_sq) = sp.potential(r_sq)

# S species — diblock, heteropolymer; stack-allocated for S ≤ ~10
struct PairTable{S, TPot}
    table::NTuple{S, NTuple{S, TPot}}
end
@inline (pt::PairTable)(si, sj, r_sq) = pt.table[si][sj](r_sq)
```

`PairTable{1,LJ}` compiles identically to `SinglePotential{LJ}`. Lorentz-Berthelot
mixing rules available as a `PairTable{S}(pot; ε, σ)` constructor convenience.

---

## Molecule Types

```julia
# No bonds — compiler eliminates all molecule branches
struct Monatomic end
@inline is_bonded_neighbor(::Monatomic, i, j) = false
@inline bond_energy(::Monatomic, positions, env) = 0.0
@inline bend_energy(::Monatomic, positions, env) = 0.0

# Bonded linear chains — homo- or heteropolymer
struct LinearChain{TBond, TBend}
    num_chains::Int
    lengths::Vector{Int}     # heterogeneous chain lengths
    offsets::Vector{Int}     # 0-based start index of chain m
    polymer_id::Vector{Int}  # chain index of particle i
    monomer_k::Vector{Int}   # position of particle i within its chain
    bond::TBond
    bend::TBend
end

# 1-2 exclusion only (correct for CG models)
@inline function is_bonded_neighbor(t::LinearChain, i, j)
    t.polymer_id[i] == t.polymer_id[j] &&
    abs(t.monomer_k[i] - t.monomer_k[j]) == 1
end
```

Species live on `ParticleSystem.species`, not in `LinearChain` — the molecule
struct is species-agnostic. Use `PairTable` on `pair` and `molecule.bond` for
heteropolymer interactions.

---

## Pair Energy Inner Loop

```julia
function _local_pair_energy(sys::ParticleSystem{D,T,Env,TPair,Mol},
                             idx::Int) where {D,T,Env,TPair,Mol}
    E = zero(T); pos_i = sys.positions[idx]; si = sys.species[idx]
    @inbounds for j in neighbor_indices(sys.env, idx)
        j == idx && continue
        is_bonded_neighbor(sys.molecule, idx, j) && continue
        r_sq = sum(abs2, difference(sys.env, pos_i, sys.positions[j]))
        E += sys.pair(si, sys.species[j], r_sq)
    end
    return E
end
```

`neighbor_indices` dispatches on `Nbr` inside `env`: cell list or full scan.

---

## Named Constructors

```julia
noble_gas(; D=3, N, L, pair_potential, T=Float64)
# → ParticleSystem{..., PeriodicBox{D,T,auto}, SinglePotential, Monatomic}

polymer_solution(; D=3, num_chains, chain_length, L,
                   pair_potential, bond_potential, bend_potential=NoBend(), T=Float64)
# → ParticleSystem{..., PeriodicBox{D,T,auto}, SinglePotential, LinearChain}

heteropolymer_solution(; D=3, num_chains, chain_length, species_sequence,
                         pair_table, bond_potential, bend_potential=NoBend(), L, T=Float64)
# → ParticleSystem{..., PeriodicBox{D,T,auto}, PairTable{S,...}, LinearChain}
```

---

## Move Dispatch

```julia
translate!(sys::ParticleSystem, alg, max_disp; rng)          # any molecule type
pivot!(sys::ParticleSystem{D,T,E,P,<:LinearChain}, alg; rng) # LinearChain only
reptate!(sys::ParticleSystem{D,T,E,P,<:LinearChain}, alg; rng)
```

`pivot!` on `Monatomic` is a compile-time `MethodError`.

---

## Usage

```julia
# Noble gas
sys = noble_gas(N=500, L=10.0, pair_potential=LennardJones(ε=1.0, σ=1.0))
init!(sys, :random; rng)
alg = Metropolis(rng; β=1.0)
meas = Measurements([:energy => (s -> s.cached_energy) => Float64[]], interval=100)
for step in 1:100_000
    translate!(sys, alg, 0.2; rng)
    measure!(meas, sys, step)
end

# Homopolymer melt
sys = polymer_solution(num_chains=20, chain_length=50, L=30.0,
    pair_potential=LennardJones(ε=1.0, σ=1.0),
    bond_potential=FENE(k=30.0, r_max=1.5),
    bend_potential=CosineBend(κ=3.0))
init!(sys, :random_walk; rng)
for step in 1:500_000
    translate!(sys, alg, 0.1; rng)
    pivot!(sys, alg; rng)
end

# Diblock A-B copolymer
seq = [k ≤ 25 ? 1 : 2 for k in 1:50]
sys = heteropolymer_solution(num_chains=20, chain_length=50, species_sequence=seq, L=30.0,
    pair_table=PairTable{2}(LennardJones; ε=[1.0 0.5; 0.5 1.0], σ=1.0),
    bond_potential=FENE(k=30.0, r_max=1.5))
init!(sys, :random_walk; rng)
```


