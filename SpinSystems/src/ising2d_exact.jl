using DelimitedFiles: readdlm

const _ISING2D_DOS_CACHE = Dict{Int,Vector{Tuple{Int,Float64}}}()

function _load_ising2d_dos(L::Integer)
    get!(_ISING2D_DOS_CACHE, Int(L)) do
        path = joinpath(@__DIR__, "..", "data", "exact_solutions", "ising2D_$(L)x$(L).csv")
        isfile(path) || error("No exact 2D Ising logDOS for L=$L; add $path")
        raw = readdlm(path, ',', skipstart=1)
        [(Int(raw[i, 1]), Float64(raw[i, 2])) for i in axes(raw, 1)]
    end
end

"""
    logdos_exact_ising2D(L::Integer; format=:binned)

Exact 2D Ising log-density of states for an `L×L` lattice (Beale 1996).

`format` keyword:
- `:binned` (default) — `BinnedObject` on `-2L²:4:2L²`; forbidden energies set to `NaN`
- `:vector` — `Vector{Tuple{Int,Float64}}` at accessible energies only
- `:dict`   — `Dict{Int,Float64}`
"""
function logdos_exact_ising2D(L::Integer; format::Symbol=:binned)
    data = _load_ising2d_dos(L)
    if format === :binned
        bo = BinnedObject(-2L^2:4:2L^2, NaN)
        for (e, v) in data; bo[e] = v; end
        return bo
    end
    format === :vector && return copy(data)
    format === :dict   && return Dict(data)
    error("Unknown format=$format. Use :binned, :vector, or :dict.")
end
logdos_exact_ising2D(; L::Integer, format::Symbol=:binned) = logdos_exact_ising2D(L; format)

""" 
    distribution_exact_ising2D(L::Integer, β::Real)
Exact canonical distribution for 2D Ising at inverse temperature `β`, from the exact DOS.
"""
distribution_exact_ising2D(L::Integer, β::Real) = 
    MonteCarloX.distribution_from_logdos(logdos_exact_ising2D(L; format=:binned), β)
