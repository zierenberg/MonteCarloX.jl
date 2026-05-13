# MCXLatticeMatter — Future Extensions

No structural redesign needed. `LatticePolymer` is already a single unified type.
Notes below capture potential extensions discussed during the MCXSoftMatter redesign.

---

## 1. Species-dependent couplings (HP model)

Current state: two scalar couplings `J_intra`, `J_inter`.

For the HP (hydrophobic-polar) model, monomers carry species labels and the
contact energy depends on species pairs. Generalise to a coupling table:

```julia
# Current — homogeneous coupling
struct ScalarCoupling{TJ}
    J_intra::TJ
    J_inter::TJ
end

# Future — species-dependent coupling
struct CouplingTable{S, TJ}
    intra::NTuple{S, NTuple{S, TJ}}   # intra[si][sj] for same-polymer contacts
    inter::NTuple{S, NTuple{S, TJ}}   # inter[si][sj] for cross-polymer contacts
end
```

The `site_contacts` function would then return energy directly (not counts),
dispatching on coupling type:

```julia
# ScalarCoupling path (current behavior, no overhead)
@inline function site_energy(sys, site, coupling::ScalarCoupling)
    ni, ne = _count_contacts(sys, site)
    return -coupling.J_intra * ni - coupling.J_inter * ne
end

# CouplingTable path (species-dependent)
@inline function site_energy(sys, site, coupling::CouplingTable)
    si = sys.species[site]
    E = zero(...)
    for nb in sys.neighbors[site]
        sj_owner = sys.state[nb]
        sj_owner == 0 && continue
        sj = sys.species[nb]
        if sj_owner == sys.state[site]
            E -= coupling.intra[si][sj]
        else
            E -= coupling.inter[si][sj]
        end
    end
    return E
end
```

This requires adding `species::Vector{Int}` to `LatticePolymer` (species per site,
not per polymer). For homogeneous systems, could use a sentinel type that always
returns 1 (same pattern as MCXSoftMatter's `SinglePotential`).

### Data structure change

```julia
mutable struct LatticePolymer{D, K, TCoupling}
    polymers::Vector{Vector{SVector{D,Int}}}
    neighbors::Vector{NTuple{K,Int}}
    dims::SVector{D,Int}
    state::Vector{Int}
    coupling::TCoupling          # ScalarCoupling | CouplingTable{S}
    species::Vector{Int}         # species[site] (only used with CouplingTable)
    cached_energy::Float64       # replaces cached_intra + cached_inter
end
```

---

## 2. Lattice gas convenience

A lattice gas is `LatticePolymer` with `length_poly=1`. No structural change needed.
Add a named constructor for clarity:

```julia
lattice_gas(; dims, N, J=1.0)
# → LatticePolymer(; dims, num_poly=N, length_poly=1, J_intra=0.0, J_inter=J)
```

The only performance concern: `polymers::Vector{Vector{SVector{D,Int}}}` creates
N single-element vectors. For large N this is ~80 bytes/particle overhead.
Acceptable for typical lattice gas sizes (L≤100 → N≤10000).

If this becomes a bottleneck, consider a `LatticeGas` type that stores only
`state::Vector{Int}` without the `polymers` array. But defer this until measured.

### Irrelevant updates for lattice gas

`slither!`, `pivot!`, `double_bridge!` are no-ops or meaningless for length-1.
Could add guard clauses or restrict dispatch:

```julia
# Option: guard clause (simplest)
function slither!(sys::LatticePolymer, alg)
    polymer_length(sys, 1) == 1 && return nothing  # no-op for lattice gas
    ...
end
```

---

## 3. Code shared with MCXSoftMatter

The following code is duplicated between MCXSoftMatter and MCXLatticeMatter:

- **Union-find** (`_find!`, `_union!`): 15 lines, identical
- **Cluster utilities** (`largest_cluster_size`, etc.): 10 lines, identical
- **Polymer observables** (`center_of_mass`, `Rg²`, `Ree²`, `gyration_tensor`):
  ~60 lines, same algorithm but different accessors

Decision: accept the duplication. Total is ~85 lines of stable algorithms.
Not worth a shared dependency package. Revisit if a third MCX package appears
with the same pattern.
