function translate!(sys::ParticleSystem{D,T}, alg::AbstractMarkovChainMonteCarlo, Δ::T; chain::Bool=false) where {D,T}
    if chain
        _translate_chain!(sys, alg, Δ)
    else
        _translate_monomer!(sys, alg, Δ)
    end
end

# Accept dispatch
# Metropolis with linear logweight: O(1), no energy(sys) call needed
@inline function _accept_energy_change!(alg::AbstractMetropolis, sys, dE)
    accept!(alg, dE)
end
# General IS: needs full energies for non-linear logweights + record_visit!
@inline function _accept_energy_change!(alg::AbstractMarkovChainMonteCarlo, sys, dE)
    E_old = energy(sys)
    accept!(alg, E_old + dE, E_old)
end

#### Monomer translate ####

function _translate_monomer!(sys::ParticleSystem{D,T}, alg::AbstractMarkovChainMonteCarlo, Δ::T) where {D,T}
    rng = alg.rng
    idx = rand(rng, 1:length(sys.positions))

    displacement = SVector{D,T}(ntuple(_ -> Δ * (T(2) * rand(rng, T) - one(T)), Val(D)))
    new_pos = constrain(sys.env, sys.positions[idx] + displacement)

    dE = _monomer_energy_change(sys, idx, new_pos)
    if _accept_energy_change!(alg, sys, _total_dE(dE))
        sys.positions[idx] = new_pos
        update_particle!(sys.cell_list, idx, new_pos)
        _update_cache!(sys.cache, dE)
    end
    return nothing
end

#### Chain translate ####

function _translate_chain!(sys::ParticleSystem{D,T,TEnv,P,<:Polymer}, alg::AbstractMarkovChainMonteCarlo, Δ::T) where {D,T,TEnv,P}
    rng = alg.rng
    n = rand(rng, 1:length(sys.molecules))
    mol = sys.molecules[n]
    M         = mol.length
    start_idx = mol.offset + 1

    displacement = SVector{D,T}(ntuple(_ -> Δ * (T(2) * rand(rng, T) - one(T)), Val(D)))

    dE = _pair_energy_change(sys, start_idx, M, displacement)
    if _accept_energy_change!(alg, sys, dE)
        @inbounds for k in 1:M
            idx     = start_idx + k - 1
            new_pos = constrain(sys.env, sys.positions[idx] + displacement)
            sys.positions[idx] = new_pos
            update_particle!(sys.cell_list, idx, new_pos)
        end
        sys.cache.pair += dE
    end
    return nothing
end
