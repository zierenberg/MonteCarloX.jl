# Interactions-based SpinSystem: cache exactness, finite differences against recomputed
# energies, exact enumerations, and the update families (cluster, Kawasaki, n-fold way,
# heat bath).

using SparseArrays: sparse, SparseMatrixCSC
import Graphs

#### Helpers ####

test_lattice_J(dims, rng; directed=false) = lattice_random_J(Tuple(dims), rng; directed=directed)

"Cache exactness: O(1) caches against a fresh O(N) recompute."
function cache_drift(sys)
    E_cached = MCXSpins.is_hamiltonian(sys) ? energy(sys) : hamiltonian_energy(sys)
    MCXSpins.recompute_all!(sys)
    E_fresh = MCXSpins.is_hamiltonian(sys) ? energy(sys) : hamiltonian_energy(sys)
    return abs(E_cached - E_fresh)
end

"delta_energy against recomputed energy differences over random accepted moves."
function finite_diff_dev(sys, rng, nmoves; proposal...)
    dev = 0.0
    for _ in 1:nmoves
        i = rand(rng, 1:length(sys.spins))
        s_new = propose_state(rng, sys, i; proposal...)
        d = delta_energy(sys, i, s_new)
        E0 = energy(sys; full=true)
        MonteCarloX.modify!(sys, i, s_new)
        E1 = energy(sys; full=true)
        dev = max(dev, abs(E1 - E0 - d))
    end
    return dev
end

"Exact ⟨e⟩/site of an L×L Ising at β by enumeration; M restricts to the Σσ == M sector."
function exact_ising_avg_e(L, β; M=nothing)
    partners = MCXSpins._build_lattice_neighbors((L, L))
    N = L * L
    Es = Float64[]
    for c in 0:(2^N - 1)
        m = 0
        E = 0
        for i in 1:N
            σi = ((c >> (i - 1)) & 1) == 1 ? 1 : -1
            m += σi
            for j in partners[i]
                j > i || continue
                E -= σi * (((c >> (j - 1)) & 1) == 1 ? 1 : -1)
            end
        end
        (M === nothing || m == M) && push!(Es, E)
    end
    w = exp.(-β .* (Es .- minimum(Es)))
    return sum(Es .* w) / sum(w) / N
end

run_flips!(sys, alg, n) = (for _ in 1:n; spin_flip!(sys, alg); end)

#### ΔE consistency across all term combinations ####
#
# (The model zoo these systems were validated against — bit-identical Glauber trajectories
# under shared seeds — was removed after the migration gate passed; finite differences vs
# recomputed energies are the self-contained replacement.)

@testset "SpinSystem finite differences across term combinations" begin
    L = 8
    rng = MersenneTwister(2026)
    hvec = randn(MersenneTwister(55), L * L)                 # site-dependent (random) field
    g = Graphs.SimpleGraphs.grid([L, L]; periodic=false)     # open boundaries: degree varies
    Js = test_lattice_J([L, L], MersenneTwister(31))

    for sys in (IsingSystem([L, L]; J=1.3, h=0.4),
                IsingSystem([L, L]; J=1.3, h=hvec),
                BlumeCapelSystem([L, L]; J=1.1, D=0.7, h=0.3),
                IsingSystem(g; J=1.3, h=0.4),
                BlumeCapelSystem(g; J=1.1, D=0.7, h=0.3),
                IsingSystem(Js; h=0.4),
                BlumeCapelSystem(Js; D=0.7, h=0.3))
        init!(sys, :random, rng=rng)
        @test finite_diff_dev(sys, MersenneTwister(7), 100) < 1e-10
    end
end

#### Cache exactness and finite differences ####

@testset "SpinSystem caches and finite differences" begin
    rng = MersenneTwister(11)

    sys = IsingSystem([8, 8]; J=1.3, h=0.4)
    init!(sys, :random, rng=rng)
    run_flips!(sys, GlauberAlgorithm(MersenneTwister(1); β=0.5), 5000)
    @test cache_drift(sys) == 0                    # Int caches: exactly zero drift

    # Spin(3//2): parametric spin types need no new code.
    s32 = SpinSystem(Spin(3//2), (PairInteraction(1, MCXSpins._init_partners([8, 8])),
                                  CrystalField(0.3)), 64)
    init!(s32, :random, rng=rng)
    @test states(Spin(3//2)) == (Int8(-3), Int8(-1), Int8(1), Int8(3))
    @test finite_diff_dev(s32, MersenneTwister(2), 100) < 1e-10

    xysys = XYSystem([8, 8]; J=1.0)
    @test energy(xysys) ≈ -2.0 * 64                # aligned: E = −J·(#bonds) = −2JN
    init!(xysys, :random, rng=rng)
    @test finite_diff_dev(xysys, MersenneTwister(3), 100; Δθ=0.7) < 1e-9   # Δθ = 0.7 proposal

    hsys = HeisenbergSystem([4, 4, 4]; J=1.0)
    @test energy(hsys) ≈ -3.0 * 64
    init!(hsys, :random, rng=rng)
    @test finite_diff_dev(hsys, MersenneTwister(4), 100) < 1e-9

    ξ = rand(MersenneTwister(12), Int8[-1, 1], 60, 3)
    hop = HopfieldSystem(ξ)
    init!(hop, :random, rng=rng)
    @test finite_diff_dev(hop, MersenneTwister(5), 100) < 1e-9

    ea = EdwardsAndersonSystem([8, 8]; rng=MersenneTwister(3))
    init!(ea, :random, rng=rng)
    run_flips!(ea, GlauberAlgorithm(MersenneTwister(6); β=1.0), 5000)
    @test cache_drift(ea) < 1e-9
end

#### Symmetry trait: Hamiltonian bookkeeping and gates ####

@testset "SpinSystem symmetry trait" begin
    rng = MersenneTwister(21)
    nrbc = VisionConeBlumeCapelSystem([8, 8]; κ=0.5, D=0.5)
    init!(nrbc, :random, rng=rng)
    @test !is_hamiltonian(nrbc)
    @test_throws ErrorException energy(nrbc)
    # hamiltonian part = pair + crystal-field terms (valid muca coordinates); cone excluded
    @test hamiltonian_energy(nrbc) ≈ energy(nrbc.interactions[1]) + energy(nrbc.interactions[3])
    run_flips!(nrbc, GlauberAlgorithm(MersenneTwister(7); β=0.6), 5000)
    @test cache_drift(nrbc) == 0

    # Directed J_ij: dynamics runs, energy refuses.
    Jdir = test_lattice_J([8, 8], MersenneTwister(41); directed=true)
    dir = IsingSystem(Jdir)
    init!(dir, :random, rng=rng)
    alg = GlauberAlgorithm(MersenneTwister(8); β=0.5)
    run_flips!(dir, alg, 3000)
    @test 0 < acceptance_rate(alg) < 1
    @test !is_hamiltonian(dir)
    @test_throws ErrorException energy(dir)
end

#### Partner and geometry queries ####

@testset "SpinSystem partners and geometry" begin
    nbrs = MCXSpins._build_lattice_neighbors((4, 4))
    sys = IsingSystem([4, 4]; J=1.0, h=0.2)
    @test all(i -> sort(collect(partners(sys.interactions[1], i))) == sort(collect(nbrs[i])), 1:16)
    @test partners(sys.interactions[2], 1) == ()          # on-site term: no partners
    @test geometry(sys) == (4, 4)

    msys = IsingSystem(test_lattice_J([4, 4], MersenneTwister(5)); geometry=(4, 4))
    @test all(i -> sort(collect(partners(msys.interactions[1], i))) == sort(collect(nbrs[i])), 1:16)

    # pair + cone terms both list the lattice neighbors — concatenated WITHOUT deduplication
    nr = VisionConeIsingSystem([4, 4]; κ=0.3)
    @test all(i -> sort(unique(collect(partners(nr, i)))) == sort(collect(nbrs[i])), 1:16)

    g = Graphs.SimpleGraphs.grid([4, 4]; periodic=false)
    @test geometry(IsingSystem(g; J=1.0)) isa Graphs.SimpleGraph
    @test geometry(IsingSystem(test_lattice_J([4, 4], MersenneTwister(6)))) === nothing
end

#### init!, full recompute, magnetization types ####

@testset "SpinSystem init! and observables" begin
    bc = BlumeCapelSystem([6, 6]; J=1.1, D=0.7, h=0.3)
    init!(bc, :down)
    @test magnetization(bc) == -36
    init!(bc, :up)
    @test magnetization(bc) == 36
    @test magnetization(bc) == magnetization(bc; full=true)
    @test energy(bc) ≈ energy(bc; full=true)

    xysys = XYSystem([4, 4])
    @test magnetization(xysys) isa ComplexF64
    hsys = HeisenbergSystem([4, 4])
    @test magnetization(hsys) isa SVector{3,Float64}

    # Lattice observables read the passive geometry; uniform configs give ξ = 0 exactly.
    # (Fourier-reference checks live in test_structure_factor.jl / test_correlation.jl.)
    sys = IsingSystem([6, 6])
    @test correlation_length(sys) == 0.0
    init!(sys, :random, rng=MersenneTwister(9))
    @test structure_factor(sys, 1) >= 0.0
    @test correlation_length(sys) >= 0.0
    @test_throws ErrorException structure_factor(HopfieldSystem(rand(MersenneTwister(9), Int8[-1, 1], 16, 1)), 1)
end

#### Cluster updates ####

@testset "SpinSystem cluster updates" begin
    βc = 1 / 2.269
    data = logdos_exact_ising2D(8; format=:vector)
    Ex = Float64.(first.(data))
    lw = last.(data) .- βc .* Ex
    wex = exp.(lw .- maximum(lw))
    e_exact = sum(Ex .* wex) / sum(wex) / 64

    wsys = IsingSystem([8, 8])
    init!(wsys, :random, rng=MersenneTwister(21))
    algw = Wolff(MersenneTwister(22); β=βc)
    for _ in 1:500; cluster_update!(wsys, algw); end
    nw = 8_000
    ew = 0.0
    for _ in 1:nw
        cluster_update!(wsys, algw)
        ew += energy(wsys)
    end
    @test abs(ew / nw / 64 - e_exact) < 0.02
    @test 0 < mean_cluster_size(algw) < 64

    ssys = IsingSystem([8, 8])
    init!(ssys, :random, rng=MersenneTwister(23))
    algs = SwendsenWang(MersenneTwister(24); β=βc)
    for _ in 1:200; cluster_update!(ssys, algs); end
    ns = 4_000
    es = 0.0
    for _ in 1:ns
        cluster_update!(ssys, algs)
        es += energy(ssys)
    end
    @test abs(es / ns / 64 - e_exact) < 0.02
    @test mean_cluster_count(algs) > 1

    # Blume–Capel: σ = 0 sites never join clusters — zero count conserved by pure cluster moves.
    bc = BlumeCapelSystem([8, 8]; J=1, D=0.5)
    init!(bc, :random, rng=MersenneTwister(28))
    nzeros = count(iszero, bc.spins)
    algb = Wolff(MersenneTwister(29); β=0.6)
    for _ in 1:200; cluster_update!(bc, algb); end
    @test count(iszero, bc.spins) == nzeros

    # Gates: field breaks flip symmetry; nonlinear ensembles are refused.
    @test_throws ErrorException cluster_update!(IsingSystem([4, 4]; h=0.3), algw)
    @test_throws ArgumentError Wolff(MersenneTwister(1), FunctionEnsemble(x -> -x^2))
    @test_throws ArgumentError SwendsenWang(MersenneTwister(1), FunctionEnsemble(x -> -x^2))
end

#### Kawasaki spin exchange ####

@testset "SpinSystem spin exchange (Kawasaki)" begin
    # Finite differences + cache commit over random swaps, for several term combinations.
    for sysk in (IsingSystem([6, 6]; J=1.3, h=randn(MersenneTwister(41), 36)),
                 BlumeCapelSystem([6, 6]; J=1.1, D=0.7, h=0.3),
                 XYSystem([6, 6]; J=1.0))
        init!(sysk, :random, rng=MersenneTwister(42))
        rngk = MersenneTwister(43)
        dev = 0.0
        for _ in 1:100
            i = rand(rngk, 1:36)
            nb = sysk.interactions[1].partners[i]
            j = nb[rand(rngk, 1:length(nb))]
            si, sj = sysk.spins[i], sysk.spins[j]
            δs = MCXSpins.delta(sysk.interactions, sysk.spins, (i, j), (sj, si))
            d = delta_energy(sysk.interactions, δs)
            E0 = energy(sysk; full=true)
            MCXSpins.commit!(sysk.interactions, δs)
            sysk.spins[i] = sj
            sysk.spins[j] = si
            E1_cached = energy(sysk)
            E1 = energy(sysk; full=true)
            dev = max(dev, abs(E1 - E0 - d), abs(E1_cached - E1))
        end
        @test dev < 1e-10
    end

    # Fixed-Σσ sector equilibrium vs exact enumeration (3×3, M = 1).
    βk = 0.7
    ek_exact = exact_ising_avg_e(3, βk; M=1)
    ksys = IsingSystem([3, 3])
    set_spins!(ksys, Int8[1, 1, 1, 1, 1, -1, -1, -1, -1])          # Σσ = +1 start
    algk = MetropolisAlgorithm(MersenneTwister(44); β=βk)
    for _ in 1:20_000
        spin_exchange!(ksys, algk)
    end
    nk = 200_000
    ek = 0.0
    for _ in 1:nk
        spin_exchange!(ksys, algk)
        ek += energy(ksys)
    end
    @test abs(ek / nk / 9 - ek_exact) < 0.01
    @test magnetization(ksys) == 1                                  # Σσ exactly conserved

    # Gate: the cone term defines no two-site delta — Kawasaki on a vision-cone system
    # refuses. (Checkerboard start: every neighbor pair is unequal, so the swap is attempted.)
    nr = VisionConeIsingSystem([4, 4]; κ=0.3)
    set_spins!(nr, Int8[(-1)^(x + y) for y in 1:4 for x in 1:4])
    @test_throws ErrorException spin_exchange!(nr, algk)
end

#### n-fold way (rejection-free): SiteEvents + NFoldRates ####

@testset "SpinSystem n-fold way" begin
    βn = 0.4
    en_exact = exact_ising_avg_e(4, βn)
    nsys = IsingSystem([4, 4])
    init!(nsys, :random, rng=MersenneTwister(45))
    src = SiteEvents(nsys, NFoldRates(β=βn))
    alg = Gillespie(MersenneTwister(46))
    advance!(alg, src, 500.0)                                               # thermalize
    acc = zeros(2)                                                          # Σ e·Δt, t_prev
    advance!(alg, src, 40_000.0;
             observe! = (s, event, t) -> (acc[1] += (t - acc[2]) * energy(nsys);
                                          acc[2] = t; nothing))
    @test abs(acc[1] / acc[2] / 16 - en_exact) < 0.025
    @test alg.steps > 1000

    # manual step!/modify! loop: interaction caches stay exact through drawn events
    for _ in 1:5_000
        t, event = step!(alg, src)
        modify!(src, event, t)
    end
    @test energy(nsys) == energy(nsys; full=true)

    @test_throws ErrorException SiteEvents(XYSystem([4, 4]), NFoldRates(β=1.0))  # discrete only
    @test_throws ArgumentError NFoldRates(FunctionEnsemble(x -> -x^2))
end

#### Generic heat bath on composed systems ####

@testset "SpinSystem heat bath (generic states enumeration)" begin
    βh = 0.5
    eh_exact = exact_ising_avg_e(4, βh)
    hsys = IsingSystem([4, 4])
    init!(hsys, :random, rng=MersenneTwister(51))
    hb = HeatBathAlgorithm(MersenneTwister(52); β=βh)
    run_flips!(hsys, hb, 30_000)
    nh = 150_000
    eh = 0.0
    for _ in 1:nh
        spin_flip!(hsys, hb)
        eh += energy(hsys)
    end
    @test abs(eh / nh / 16 - eh_exact) < 0.02

    # 3-state: heat-bath BC agrees with Metropolis BC.
    nbc = 400_000
    bc1 = BlumeCapelSystem([6, 6]; J=1, D=0.5)
    init!(bc1, :random, rng=MersenneTwister(53))
    run_flips!(bc1, HeatBathAlgorithm(MersenneTwister(54); β=0.6), 20_000)
    e1 = 0.0
    hb2 = HeatBathAlgorithm(MersenneTwister(55); β=0.6)
    for _ in 1:nbc
        spin_flip!(bc1, hb2)
        e1 += energy(bc1)
    end
    bc2 = BlumeCapelSystem([6, 6]; J=1, D=0.5)
    init!(bc2, :random, rng=MersenneTwister(56))
    algm = MetropolisAlgorithm(MersenneTwister(57); β=0.6)
    run_flips!(bc2, algm, 20_000)
    e2 = 0.0
    for _ in 1:nbc
        spin_flip!(bc2, algm)
        e2 += energy(bc2)
    end
    @test abs(e1 - e2) / nbc / 36 < 0.02
end
