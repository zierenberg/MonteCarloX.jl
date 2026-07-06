# Nonreciprocal spin systems (single-species vision cone).
#
# A composed `SpinSystem{model, nr, topo, cache}` with three orthogonal axes:
#   model : IsingModel / BlumeCapelModel(Δ)      — state set, crystal field, proposal
#   nr    : Reciprocal / VisionCone(κ)           — how the local field breaks reciprocity
#   topo  : LatticeTopology                       — neighbor structure (orientation-aware)
#
# The nonreciprocity is a vision cone: spin i couples more strongly (J + κ) to the neighbors ahead
# of its polarization p̂_i = σ_i(x̂+ŷ)/√2. The coupling is asymmetric (J_ij ≠ J_ji), breaking
# detailed balance; there is no global Hamiltonian, only the energy-like diagnostic e = ½ Σ_i E_i.
# κ is a temperature-independent coupling (Garcés & Levis convention), so the local energy is β-free
# and the system plugs into the standard `spin_flip!`/`accept!` path unchanged, with β living
# entirely in the algorithm. Reference: J. Stat. Mech. 2025 043205; cf. arXiv:2606.06981.

#### Spin models ####

"""
    SpinModel

Base type for the spin-model axis: it fixes the local state set, the crystal-field term, and the
single-spin proposal, independent of topology and of the nonreciprocity kind.
"""
abstract type SpinModel end

"""
    IsingModel(J=1)

Two-state model, spins ∈ {-1, 1}, coupling `J`.
"""
struct IsingModel{TJ<:Real} <: SpinModel
    J::TJ
end
IsingModel() = IsingModel(1)

"""
    BlumeCapelModel(J=1, Δ=0)

Three-state model, spins ∈ {-1, 0, 1}, coupling `J`, crystal field `Δ` (energy `+Δ σ²`).
"""
struct BlumeCapelModel{TJ<:Real, TC<:Real} <: SpinModel
    J::TJ
    Δ::TC
end
BlumeCapelModel() = BlumeCapelModel(1, 0)

@inline model_states(::IsingModel) = Int8[-1, 1]
@inline model_states(::BlumeCapelModel) = Int8[-1, 0, 1]

@inline propose_state(rng::AbstractRNG, ::IsingModel, s_old::Int8) = Int8(-s_old)

@inline function propose_state(rng::AbstractRNG, ::BlumeCapelModel, s_old::Int8)
    u = rand(rng, Bool)
    if s_old == Int8(-1)
        return u ? Int8(0) : Int8(1)
    elseif s_old == Int8(0)
        return u ? Int8(-1) : Int8(1)
    else
        return u ? Int8(-1) : Int8(0)
    end
end

@inline onsite_energy(::IsingModel, s::Int8) = 0.0
@inline onsite_energy(m::BlumeCapelModel, s::Int8) = float(m.Δ) * Int(s)^2

#### Nonreciprocity kinds ####

"""
    Nonreciprocity

Base type for the nonreciprocity axis: it determines how the local field is formed from the
neighbors. `Reciprocal` is the identity (equilibrium); `VisionCone` is single-species directional
nonreciprocity. New kinds (e.g. two-species) slot in here without touching models or topology.
"""
abstract type Nonreciprocity end

"""
    Reciprocal()

Identity nonreciprocity: the local field is the plain neighbor sum (equilibrium coupling).
"""
struct Reciprocal <: Nonreciprocity end

"""
    VisionCone(κ)

Single-species vision cone of strength `κ`: a spin couples more strongly (`J + κ`) to the neighbors
ahead of its polarization (half-plane cone, `f=1/2`). `κ` is a temperature-independent coupling
(Garcés & Levis convention); `κ=0` recovers `Reciprocal`.
"""
struct VisionCone{Tκ<:Real} <: Nonreciprocity
    κ::Tκ
end

@inline _kappa(::Reciprocal) = 0.0
@inline _kappa(nr::VisionCone) = nr.κ

#### Topology ####

"""
    LatticeTopology(dims, nbrs)

D-dimensional periodic hypercubic topology. `nbrs[i]` lists the `2D` neighbors of site `i` in the
direction order `-x,+x,-y,+y,…` (odd index = `-dir`, even index = `+dir`), which encodes the
orientation the vision cone needs. Built via [`_build_lattice_neighbors`](@ref).
"""
struct LatticeTopology{D, NN}
    dims::NTuple{D,Int}
    nbrs::Vector{NTuple{NN,Int}}
end

# Directional neighbor sums: `minus` over `-dir` neighbors (odd index), `plus` over `+dir` (even).
@inline function _dir_sums(topo::LatticeTopology{D,NN}, spins, i) where {D,NN}
    nb = @inbounds topo.nbrs[i]
    minus = 0
    plus = 0
    @inbounds for k in 1:NN
        s = Int(spins[nb[k]])
        isodd(k) ? (minus += s) : (plus += s)
    end
    return minus, plus
end

#### System + cache ####

"""
    SpinCache(mag, spin2)

O(1)-updatable running sums shared by all models: total magnetization `Σσ` and `Σσ²`.
"""
mutable struct SpinCache
    mag::Int
    spin2::Int
end

"""
    SpinSystem{M,N,T,C}

Composed spin system with orthogonal `model`, `nr`, and `topo` components plus a mutable `cache`.
Use [`NonreciprocalIsing`](@ref) / [`NonreciprocalBlumeCapel`](@ref), or the general
`SpinSystem(model, nr, dims)` constructor.
"""
mutable struct SpinSystem{M<:SpinModel, N<:Nonreciprocity, T, C} <: AbstractSpinSystem
    spins::Vector{Int8}
    const model::M
    const nr::N
    const topo::T
    const cache::C
end

function SpinSystem(model::SpinModel, nr::Nonreciprocity, dims::AbstractVector{<:Integer})
    d = Tuple(Int.(dims))
    nbrs = _build_lattice_neighbors(d)
    topo = LatticeTopology(d, nbrs)
    sys = SpinSystem(ones(Int8, prod(d)), model, nr, topo, SpinCache(0, 0))
    _recompute_cache!(sys)
    return sys
end

"""
    NonreciprocalIsing(dims; κ, J=1)

Nonreciprocal Ising system on a periodic hypercubic lattice with vision-cone coupling `κ`.
"""
NonreciprocalIsing(dims::AbstractVector{<:Integer}; κ, J=1) =
    SpinSystem(IsingModel(J), VisionCone(κ), dims)

"""
    NonreciprocalBlumeCapel(dims; κ, D=0, J=1)

Nonreciprocal Blume–Capel system on a periodic hypercubic lattice with vision-cone coupling `κ` and
crystal field `D`.
"""
NonreciprocalBlumeCapel(dims::AbstractVector{<:Integer}; κ, D=0, J=1) =
    SpinSystem(BlumeCapelModel(J, D), VisionCone(κ), dims)

function _recompute_cache!(sys::SpinSystem)
    sys.cache.mag = sum(Int, sys.spins)
    sys.cache.spin2 = sum(s -> Int(s)^2, sys.spins)
    return nothing
end

function init!(sys::SpinSystem, type::Symbol; rng=nothing)
    if type == :up
        fill!(sys.spins, Int8(1))
    elseif type == :down
        fill!(sys.spins, Int8(-1))
    elseif type == :zero
        Int8(0) in model_states(sys.model) || error("state 0 is not valid for $(typeof(sys.model))")
        fill!(sys.spins, Int8(0))
    elseif type == :random
        @assert rng !== nothing "Random initialization requires rng"
        states = model_states(sys.model)
        @inbounds for i in eachindex(sys.spins)
            sys.spins[i] = rand(rng, states)
        end
    else
        error("Unknown initialization type: $type")
    end
    _recompute_cache!(sys)
    return sys
end

@inline function MonteCarloX.modify!(sys::SpinSystem, i::Int, s_new::Int8)
    @inbounds s_old = Int(sys.spins[i])
    @inbounds sys.spins[i] = s_new
    sys.cache.mag += Int(s_new) - s_old
    sys.cache.spin2 += Int(s_new)^2 - s_old^2
    return nothing
end

#### Local field ####

"""
    field_components(sys, i, s_ref) -> (full, forward)

Directional neighbor sums for site `i` given reference spin `s_ref`: `full` is the sum over all
neighbors; `forward` is the sum over the neighbors inside `s_ref`'s vision cone (`0` for
`Reciprocal`, and `0` when `s_ref == 0`). The local field is `h = J·full + κ·forward`.
"""
@inline function field_components(sys::SpinSystem{M,Reciprocal}, i, s_ref) where {M}
    minus, plus = _dir_sums(sys.topo, sys.spins, i)
    return (minus + plus, 0)
end

@inline function field_components(sys::SpinSystem{M,<:VisionCone}, i, s_ref) where {M}
    minus, plus = _dir_sums(sys.topo, sys.spins, i)
    forward = s_ref > 0 ? plus : (s_ref < 0 ? minus : 0)
    return (minus + plus, forward)
end

#### Interface: proposal + local energy (drives the generic spin_flip!) ####

@inline propose_state(rng::AbstractRNG, sys::SpinSystem, i) =
    propose_state(rng, sys.model, @inbounds sys.spins[i])

"""
    delta_energy(sys::SpinSystem, i, s_new) -> Float64

Local energy change for setting site `i` to `s_new`, using the vision cone frozen at the current
spin. Temperature-independent (coupling `J + κ`), so the standard `accept!` supplies β.
"""
@inline function delta_energy(sys::SpinSystem, i, s_new::Int8)
    @inbounds s_old = sys.spins[i]
    full, forward = field_components(sys, i, s_old)
    h = float(sys.model.J) * full + _kappa(sys.nr) * forward
    Δs = Int(s_new) - Int(s_old)
    return -Δs * h + (onsite_energy(sys.model, s_new) - onsite_energy(sys.model, s_old))
end

#### Observables ####

@inline magnetization(sys::SpinSystem) = sys.cache.mag

"""
    energy(sys::SpinSystem) -> Float64

Energy-like diagnostic `½ Σ_i E_i` with `E_i = -σ_i Σ_j J_ij σ_j + Δ σ_i²` (vision cone at each
site's own spin, coupling `J + κ` on forward neighbors). There is no global Hamiltonian; at `κ=0`
this reduces to the standard Ising / Blume–Capel energy.
"""
function energy(sys::SpinSystem)
    J = float(sys.model.J)
    κ = _kappa(sys.nr)
    Epair = 0.0
    Eonsite = 0.0
    @inbounds for i in eachindex(sys.spins)
        s = Int(sys.spins[i])
        full, forward = field_components(sys, i, s)
        h = J * full + κ * forward
        Epair += -s * h
        Eonsite += onsite_energy(sys.model, sys.spins[i])
    end
    return Epair / 2 + Eonsite
end

"""
    structure_factor(sys, d) -> Float64

Static structure factor `|Σ_j σ_j e^{i k·r_j}|²` at the smallest wavevector along axis `d`
(`k = 2π/L_d`). These are the `cos`/`sin` Fourier sums used for the correlation length.
"""
function structure_factor(sys::SpinSystem, d::Int)
    dims = sys.topo.dims
    L = dims[d]
    L < 2 && return 0.0
    stride = d == 1 ? 1 : prod(dims[1:d-1])
    k = 2π / L
    c = 0.0
    s = 0.0
    @inbounds for site in eachindex(sys.spins)
        x = ((site - 1) ÷ stride) % L
        σ = Int(sys.spins[site])
        c += σ * cos(k * x)
        s += σ * sin(k * x)
    end
    return c^2 + s^2
end

"""
    correlation_length(sys) -> Float64

Second-moment correlation length `ξ = (1/2sin(π/L)) √(S(0)/S(k_min) − 1)`, averaged over lattice
axes, with `S(0) = (Σσ)²`. Returns `0` when the structure factor vanishes.
"""
function correlation_length(sys::SpinSystem)
    dims = sys.topo.dims
    S0 = float(magnetization(sys))^2
    acc = 0.0
    n = 0
    for d in 1:length(dims)
        L = dims[d]
        L < 2 && continue
        Sk = structure_factor(sys, d)
        Sk <= 0 && continue
        acc += sqrt(max(S0 / Sk - 1, 0.0)) / (2 * sin(π / L))
        n += 1
    end
    return n > 0 ? acc / n : 0.0
end
