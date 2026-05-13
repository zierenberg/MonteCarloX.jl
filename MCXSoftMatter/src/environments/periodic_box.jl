"""
    PeriodicBox{D, T} <: AbstractEnvironment{D, T}

D-dimensional periodic box with side lengths `L` and precomputed `inv_L = 1 ./ L`.
Supports anisotropic boxes (different Lx, Ly, Lz).

Construct from an SVector or a scalar (cubic box):
```julia
PeriodicBox(SVector(10.0, 20.0, 10.0))   # anisotropic
PeriodicBox{3}(10.0)                       # cubic: L = (10, 10, 10)
```
"""
struct PeriodicBox{D, T<:AbstractFloat} <: AbstractEnvironment{D,T}
    L::SVector{D,T}
    inv_L::SVector{D,T}
end

PeriodicBox(L::SVector{D,T}) where {D,T} = PeriodicBox{D,T}(L, inv.(L))
PeriodicBox{D}(L::Real) where D = PeriodicBox(SVector{D}(ntuple(_ -> float(L), Val(D))))

# ── Environment interface ──────────────────────────────────────────────────

@inline function difference(env::PeriodicBox{D}, a::SVector{D,T}, b::SVector{D,T}) where {D,T}
    SVector{D,T}(ntuple(Val(D)) do d
        dx = a[d] - b[d]
        muladd(-env.L[d], round(dx * env.inv_L[d]), dx)
    end)
end

@inline function constrain(env::PeriodicBox{D}, x::SVector{D,T}) where {D,T}
    SVector{D,T}(ntuple(Val(D)) do d
        x[d] - env.L[d] * floor(x[d] * env.inv_L[d])
    end)
end

# Future extensions: Add HardBox with joint constructor Box(L, boundary=:periodic) and Box(L, boundary=:hard) for non-periodic boundaries.