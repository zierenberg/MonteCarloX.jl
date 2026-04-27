"""
    kawasaki_move!(sys::LatticeGas, alg::AbstractMetropolis)

Perform a Kawasaki swap move: pick a random occupied site and a random empty site,
compute delta energy, and accept/reject via Metropolis criterion.
"""
function kawasaki_move!(sys::LatticeGas, alg::AbstractMetropolis)
    occ_idx = rand(alg.rng, 1:sys.N_particles)
    emp_idx = rand(alg.rng, 1:(sys.N_sites - sys.N_particles))
    occ_site = sys.occupied_sites[occ_idx]
    emp_site = sys.empty_sites[emp_idx]

    ΔE = delta_energy(sys, occ_site, emp_site)

    if accept!(alg, ΔE)
        delta_contacts = round(Int, ΔE / (-sys.J))
        modify!(sys, occ_site, emp_site, delta_contacts)
    end
    return nothing
end

"""
    kawasaki_move!(sys::LatticeGas, alg::AbstractImportanceSampling)

Kawasaki swap with generic importance sampling (e.g. multicanonical).
Uses state-based accept!(alg, E_new, E_old).
"""
function kawasaki_move!(sys::LatticeGas, alg::AbstractImportanceSampling)
    occ_idx = rand(alg.rng, 1:sys.N_particles)
    emp_idx = rand(alg.rng, 1:(sys.N_sites - sys.N_particles))
    occ_site = sys.occupied_sites[occ_idx]
    emp_site = sys.empty_sites[emp_idx]

    ΔE = delta_energy(sys, occ_site, emp_site)
    E_old = energy(sys)
    E_new = E_old + ΔE

    if accept!(alg, E_new, E_old)
        delta_contacts = round(Int, ΔE / (-sys.J))
        modify!(sys, occ_site, emp_site, delta_contacts)
    end
    return nothing
end
