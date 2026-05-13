function center_of_mass(sys::ParticleSystem{D,T,TEnv,P,<:Polymer}, n::Int) where {D,T,TEnv,P}
    mol = sys.molecules[n]
    M   = mol.length
    off = mol.offset
    ref = sys.positions[off + 1]
    env = sys.env
    cm  = zero(MVector{D,Float64})
    for k in 1:M
        pos = sys.positions[off + k]
        d = difference(env, pos, ref)
        cm .+= d
    end
    cm ./= M
    for d in 1:D
        cm[d] = cm[d] + ref[d]
    end
    return SVector{D,Float64}(constrain(env, SVector{D,Float64}(cm)))
end

function radius_of_gyration_sq(sys::ParticleSystem{D,T,TEnv,P,<:Polymer}, n::Int) where {D,T,TEnv,P}
    mol = sys.molecules[n]
    M   = mol.length
    off = mol.offset
    cm  = center_of_mass(sys, n)
    env = sys.env
    rg2 = 0.0
    for k in 1:M
        pos = sys.positions[off + k]
        d = difference(env, pos, cm)
        rg2 += sum(abs2, d)
    end
    return rg2 / M
end

function end_to_end_distance_sq(sys::ParticleSystem{D,T,TEnv,P,<:Polymer}, n::Int) where {D,T,TEnv,P}
    mol = sys.molecules[n]
    M   = mol.length
    off = mol.offset
    r1  = sys.positions[off + 1]
    rN  = sys.positions[off + M]
    return Float64(distance_sq(sys.env, r1, rN))
end

function gyration_tensor(sys::ParticleSystem{D,T,TEnv,P,<:Polymer}, n::Int) where {D,T,TEnv,P}
    mol = sys.molecules[n]
    M   = mol.length
    off = mol.offset
    cm  = center_of_mass(sys, n)
    env = sys.env
    G   = zeros(MMatrix{D,D,Float64})
    for k in 1:M
        pos = sys.positions[off + k]
        d = difference(env, pos, cm)
        for i in 1:D, j in 1:D
            G[i,j] += d[i] * d[j]
        end
    end
    G ./= M
    return SMatrix{D,D,Float64}(G)
end
