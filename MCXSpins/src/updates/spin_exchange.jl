# ── Kawasaki spin exchange ────────────────────────────────────────────────────
#
# Two-site update through the two-site delta method family (interactions.jl): the change
# record for setting the spins of (i, j) simultaneously. A spin EXCHANGE conserves Σσ
# (spin_sum untouched): Kawasaki dynamics for conserved order parameter.

"""
    spin_exchange!(sys, alg; topology=first(sys.interactions))

One Kawasaki move: swap the spins of a random site and one of its partners, accepted via the
two-site delta family through the algorithm's ensemble and balance. `topology` selects the
pair term whose partner table defines locality. Proposing the unordered pair (i,j) as
i-then-j or j-then-i is symmetric even for varying degree, so detailed balance holds.
Identity swaps (equal spins) are skipped without counting an attempt.
"""
@inline function spin_exchange!(sys::SpinSystem, alg::MetropolisHastingsAlgorithm;
                                topology::PairInteraction=first(sys.interactions))
    spins = sys.spins
    i = pick_site(alg.rng, length(spins))
    nb = @inbounds topology.partners[i]
    j = @inbounds nb[rand(alg.rng, 1:length(nb))]
    @inbounds si = spins[i]
    @inbounds sj = spins[j]
    si == sj && return nothing
    δs = delta(sys.interactions, spins, (i, j), (sj, si))
    ΔE = delta_energy(sys.interactions, δs)
    if _accept_delta!(alg, sys, ΔE)
        commit!(sys.interactions, δs)
        @inbounds spins[i] = sj
        @inbounds spins[j] = si
    end
    return nothing
end
