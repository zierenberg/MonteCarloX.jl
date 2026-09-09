# Exact density of states g(E) of the L×L Ising model (periodic boundaries, J=1),
# written as CSVs consumed by `MCXSpins.logdos_exact_ising2D`
# (`../data/exact_solutions/ising2D_LxL.csv`).
#
# Implements: P. D. Beale, "Exact Distribution of Energies in the Two-Dimensional
# Ising Model", Phys. Rev. Lett. 76, 78 (1996).
#   https://doi.org/10.1103/PhysRevLett.76.78   (local copy: literature/1995_beale_prl.pdf)
#
# Beale casts Kaufman's finite-lattice partition function as a polynomial in the
# low-temperature variable x = e^{-2K},
#     Z_{m,n}(K) = e^{2mnK} Σ_{k=0}^{mn} g_k x^{2k},                        (Eq. 2)
# where g_k counts configurations with energy 4kJ above the two ground states,
# i.e. g(E = 4k - 2mn) = g_k.  All arithmetic below is polynomial arithmetic in x
# following Eqs. (10a)-(12d) verbatim; the only irrational inputs are cos(πk/n),
# so modest BigFloat precision (a few times mn bits) is exact for any L — no
# overflow, and a final round-to-integer check catches any precision loss loudly.
#
# An independent column-transfer-matrix enumeration (exponential in L) verifies
# small lattices, and the L=32 flagship value g_512 ≈ 6.342873169×10^306 printed
# in the Letter is reproduced by `validate_L32()`.
#
# Run:  julia --project=MCXSpins MCXSpins/theory/beale_ising2d_dos.jl
using Printf

#### Polynomial arithmetic (coefficient vectors, index i ↔ x^(i-1)) ####
poly(c...) = BigFloat.(collect(c))
function pmul(a, b)
    c = zeros(BigFloat, length(a) + length(b) - 1)
    for i in eachindex(a), j in eachindex(b)
        c[i + j - 1] = muladd(a[i], b[j], c[i + j - 1])
    end
    c
end
pad(a, len) = vcat(a, zeros(BigFloat, len - length(a)))
padd(a, b) = (len = max(length(a), length(b)); pad(a, len) .+ pad(b, len))
psub(a, b) = padd(a, -b)
ppow(a, k) = k == 0 ? poly(1) : pmul(a, ppow(a, k - 1))
pshift(a, k) = vcat(zeros(BigFloat, k), a)              # multiply by x^k

#### Beale's exact coefficients g_k (Eqs. 10-12; m = n = L, even) ####
"""
    beale_gk(L) -> (g::Vector{BigInt}, δ::Float64)

Exact coefficients `g[k+1] = g_k` of Eq. (2) for the even L×L lattice, and the
largest deviation `δ` of any computed coefficient from an integer (must be ≈ 0;
errors out otherwise, so precision loss can never corrupt the output silently).
"""
function beale_gk(L::Int)
    iseven(L) || error("Beale Eqs. (12) as implemented require even L (got L=$L)")
    m = n = L
    setprecision(BigFloat, 4m * n + 256) do
        β  = poly(0, 2, 0, -2)                          # 2x(1-x²)              (10a)
        β² = pmul(β, β)
        βᵐ = ppow(β, m)

        # c_k² and s_k² of Eqs. (10g)/(10h): degree-4m polynomials for cos(πk/n)
        function ck²sk²(cosθ)
            α = psub(poly(1, 0, 2, 0, 1), cosθ * β)     # (1+x²)² - β cos(πk/n) (10b)
            D = psub(pmul(α, α), β²)                    # α_k² - β²
            αᵖ = [poly(1)]; Dᵖ = [poly(1)]
            for i in 1:m;     push!(αᵖ, pmul(αᵖ[end], α)) end
            for j in 1:m ÷ 2; push!(Dᵖ, pmul(Dᵖ[end], D)) end
            Σ = poly(0)
            for j in 0:m ÷ 2                            # Σ_j m!/((2j)!(m-2j)!) (α²-β²)ʲ α^(m-2j)
                Σ = padd(Σ, binomial(big(m), 2j) * pmul(Dᵖ[j + 1], αᵖ[m - 2j + 1]))
            end
            padd(Σ, βᵐ) ./ big(2)^(m - 1), psub(Σ, βᵐ) ./ big(2)^(m - 1)
        end

        u = ppow(poly(1, -1), m)                        # (1-x)^m
        v = ppow(poly(1, 1), m)                         # (1+x)^m
        c₀ = padd(u, pshift(v, m)); s₀ = psub(u, pshift(v, m))    # (10c), (10d)
        cₙ = padd(v, pshift(u, m)); sₙ = psub(v, pshift(u, m))    # (10e), (10f)

        cosθ(k) = cos(big(π) * k / n)
        Z₁ = poly(1); Z₂ = poly(1)                      # antiperiodic sector   (12a), (12b)
        for k in 0:n ÷ 2 - 1
            c², s² = ck²sk²(cosθ(2k + 1))
            Z₁ = pmul(Z₁, c²); Z₂ = pmul(Z₂, s²)
        end
        Z₃ = pmul(c₀, cₙ); Z₄ = pmul(s₀, sₙ)            # periodic sector       (12c), (12d)
        for k in 1:n ÷ 2 - 1
            c², s² = ck²sk²(cosθ(2k))
            Z₃ = pmul(Z₃, c²); Z₄ = pmul(Z₄, s²)
        end

        P = padd(padd(Z₁, Z₂), padd(Z₃, Z₄)) ./ 2       # = Σ_k g_k x^{2k}      (2), (11)
        length(P) == 2m * n + 1 || error("L=$L: polynomial degree ≠ 2mn")
        δ = Float64(maximum(abs(p - round(p)) for p in P))
        δ < 1e-6 || error("L=$L: coefficient off integer by $δ — raise setprecision")
        all(p -> abs(p) < 1e-6, P[2:2:end]) || error("L=$L: odd powers of x survive")
        [round(BigInt, P[2k + 1]) for k in 0:m * n], δ
    end
end

#### Transfer matrix (independent verifier, exact, O(4^L) — small L only) ####
"exact DOS as (energies, counts) via the column transfer matrix; small L only."
function transfer_dos(L::Int)
    n = 1 << L; mask = n - 1; maxs = 2L^2
    rol(a) = ((a << 1) | (a >> (L - 1))) & mask
    intra = [L - count_ones(a ⊻ rol(a)) for a in 0:n-1]
    e(a, c) = (L - count_ones(a ⊻ c)) + intra[c + 1]
    M = [zeros(BigInt, maxs + 1) for _ in 0:n-1, _ in 0:n-1]
    for a in 0:n-1, c in 0:n-1; M[a + 1, c + 1][e(a, c) + 1] = 1 end
    for _ in 2:L
        N = [zeros(BigInt, maxs + 1) for _ in 0:n-1, _ in 0:n-1]
        for a in 0:n-1, b in 0:n-1
            Mab = M[a + 1, b + 1]
            for c in 0:n-1
                k = e(b, c); Nac = N[a + 1, c + 1]
                @inbounds for s in 0:maxs-k; Nac[s + k + 1] += Mab[s + 1] end
            end
        end
        M = N
    end
    g = reduce((x, y) -> x .+ y, M[a + 1, a + 1] for a in 0:n-1)
    E = [2L^2 - 2s for s in 0:maxs]; keep = g .!= 0
    return E[keep], g[keep]
end

#### generate + verify + write ####
"Beale ≟ transfer matrix, coefficient by coefficient (feasible for L ≲ 8)."
function verify_against_transfer(L::Int)
    g, _ = beale_gk(L)
    exact = Dict(zip(transfer_dos(L)...))
    all(get(exact, 4k - 2L^2, big(0)) == g[k + 1] for k in 0:L^2) ||
        error("L=$L: Beale ≠ transfer matrix")
    println("L=$L  Beale ≡ transfer matrix ✓")
end

function generate(L::Int; dir = joinpath(@__DIR__, "..", "data", "exact_solutions"))
    g, δ = beale_gk(L)
    N = L^2

    # global checks, plus the low-energy tail against the series Eq. (3) (m,n > 5)
    sum(g) == big(2)^N || error("L=$L: Σ g_k ≠ 2^$N")
    g == reverse(g) || error("L=$L: symmetry g_k = g_{mn-k} violated (Eq. 4)")
    tail = (2, 0, 2N, 4N, big(N)^2 + 9N, 4big(N)^2 + 24N)
    all(g[k + 1] == tail[k + 1] for k in 0:5) || error("L=$L: g_0..g_5 ≠ Eq. (3) series")

    path = joinpath(dir, "ising2D_$(L)x$(L).csv")
    setprecision(BigFloat, 4N) do
        open(path, "w") do io
            println(io, "energy,logdos")
            for k in 0:N
                g[k + 1] > 0 && println(io, 4k - 2N, ",", Float64(log(BigFloat(g[k + 1]))))
            end
        end
    end
    @printf "L=%-3d Σg=2^%-4d δ=%.1e ✓  →  %s\n" L N δ basename(path)
end

"reproduce the largest L=32 coefficient printed in the Letter, g_512 ≈ 6.342873169×10^306."
function validate_L32()
    g, δ = beale_gk(32)
    sum(g) == big(2)^1024 || error("L=32: Σ g_k ≠ 2^1024")
    g₅₁₂ = g[513]
    ndigits(g₅₁₂) == 307 || error("L=32: g_512 has $(ndigits(g₅₁₂)) digits, expected 307")
    abs(g₅₁₂ ÷ big(10)^297 - 6342873169) <= 1 || error("L=32: g_512 ≠ 6.342873169×10^306")
    @printf "L=32  g_512 = %s…×10^306 matches Beale (1996), δ=%.1e ✓\n" string(g₅₁₂)[1:10] δ
end

if abspath(PROGRAM_FILE) == @__FILE__
    verify_against_transfer(4)   # independent-method cross-checks
    verify_against_transfer(6)
    generate(8)
    generate(20)
    validate_L32()               # literature check against the value printed in the PRL
end
