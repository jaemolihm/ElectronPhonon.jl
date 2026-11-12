# Wannier-basis representation of the spatial symmetry operations, fitted from H(R).
# Included after model.jl (it builds `SymmetryOperators`). No loader fills `model.el_sym` for models
# read with `load_model_from_epw_new`, and the file reader of the removed `epw_data_julia_symmetry.bin`
# (EP #111) has no writer for them, so the operators are recovered from the Hamiltonian itself.

export fit_onsite_symmetry_operators

"""
    fit_onsite_symmetry_operators(model; symmetry = model.symmetry, atol = 1e-6) -> SymmetryOperators

Fit the on-site Wannier-basis representation `D_S` of every spatial operation `S` of `symmetry` from
the Hamiltonian alone, and return it as a `SymmetryOperators` ready for `model.el_sym`. `D_S` is real,
orthogonal and k-independent, and satisfies `H(S_R R) D_S = D_S H(R)` for all lattice vectors `R`,
`S_R = S^(-T)` being the real-space matrix of the reciprocal-space operation `S`. Hence
`H(Sk) = D_S H(k) D_S'` and `u(Sk) = D_S u(k)`. A time-reversal copy of `S` reuses the matrix of `S`.

Envelope (anything outside it is an error, not a fit): spinless real orbitals; every Wannier centre
mapped onto itself (mod lattice) by every operation; symmorphic operations (`τ` a lattice vector).
For each operation `vec(D)` is the smallest eigenvector of `sum_R M_R' M_R`,
`M_R = I ⊗ H(S_R R) - H(R)ᵀ ⊗ I`, made orthogonal by its polar factor after fixing the global phase
to make it real. The sign of each `D_S` is then fixed so the set is a representation,
`D_S D_T = D_(ST)`, by solving the sign cocycle over GF(2); the residual freedom is a sign character
of the group, which leaves every `D ⊗ D` product and `D H D'` unchanged. Errors if the relative
residual `max_R ‖H(S_R R) D - D H(R)‖ / max_R ‖H(R)‖`, the imaginary part of `D` or the
representation closure exceeds `atol`, or if some `S_R R` is not in `model.el_ham.irvec`.
"""
function fit_onsite_symmetry_operators(model; symmetry = model.symmetry, atol = 1e-6)
    nw = model.nw
    irvec = model.el_ham.irvec
    blocks = reshape(model.el_ham.op_r, nw, nw, :)
    irindex = Dict(R => ir for (ir, R) in enumerate(irvec))
    hscale = maximum(ir -> norm(blocks[:, :, ir]), axes(blocks, 3))
    ops = findall(!, symmetry.is_tr)
    Idm = Matrix{ComplexF64}(I, nw, nw)

    isinteger_approx(x) = isapprox(x, round(x); atol = 1e-4)

    # Wannier centres are on-site: each is mapped onto itself modulo a lattice vector
    for isym in ops
        symop = symmetry[isym]
        all(isinteger_approx, symop.τ) || error("operation $isym is not symmorphic (τ = $(symop.τ)): " *
            "an on-site D cannot describe it")
        for c in model.wann_centers
            frac = model.lattice \ (symop.Scart * c - c)
            all(isinteger_approx, frac) || error("operation $isym moves the Wannier centre $c off itself: " *
                "an on-site D cannot describe a site permutation")
        end
    end

    D_raw = Dict{Int, Matrix{Float64}}()
    for isym in ops
        SR = round.(Int, inv(symmetry.S[isym])')
        # Normal matrix of the commutant equations H(S_R R) D = D H(R), summed over R
        normal = zeros(ComplexF64, nw^2, nw^2)
        for (ir, R) in enumerate(irvec)
            ir2 = get(irindex, SR * R, 0)
            ir2 == 0 && error("S_R R = $(SR * R) of operation $isym is not in model.el_ham.irvec")
            M = kron(Idm, blocks[:, :, ir2]) - kron(transpose(blocks[:, :, ir]), Idm)
            normal .+= M' * M
        end
        eig = eigen(Hermitian(normal))
        # A second near-zero mode means the commutant is not one-dimensional (extra or accidental
        # symmetry of H(R)), so the lowest vector need not be the orbital representation
        eig.values[2] > (atol * hscale)^2 * length(irvec) || error("operation $isym: the commutant of " *
            "H(R) is not unique (two smallest normal-matrix eigenvalues $(eig.values[1]), " *
            "$(eig.values[2])); D cannot be fixed from H alone")
        P = eig.vectors[:, 1]
        F = svd(reshape(P, nw, nw))
        P = F.U * F.Vt
        # Fix the global phase so that D is real
        iref = argmax(abs.(P))
        P .*= conj(P[iref]) / abs(P[iref])
        maximum(abs, imag(P)) <= atol || error("operation $isym: fitted D has imaginary part " *
            "$(maximum(abs, imag(P))) > atol = $atol (spinless real orbitals only)")
        F = svd(real(P))
        D = F.U * F.Vt
        resid = maximum(eachindex(irvec)) do ir
            norm(blocks[:, :, irindex[SR * irvec[ir]]] * D - D * blocks[:, :, ir]) / hscale
        end
        resid <= atol || error("operation $isym: H(R) is not symmetric under an on-site D, relative " *
            "residual $resid > atol = $atol")
        D_raw[isym] = D
    end

    # Sign cocycle: D_S D_T = (-1)^b(S,T) D_(ST); solve x_S + x_T + x_ST = b (mod 2) by elimination
    pos = Dict(isym => n for (n, isym) in enumerate(ops))
    op_of = Dict(Tuple(symmetry.S[isym]) => isym for isym in ops)
    rows = BitVector[]
    rhs = Bool[]
    for a in ops, b in ops
        c = get(op_of, Tuple(symmetry.S[a] * symmetry.S[b]), 0)
        c == 0 && error("the spatial operations are not closed under products")
        D_ab = D_raw[a] * D_raw[b]
        plus, minus = norm(D_ab - D_raw[c]), norm(D_ab + D_raw[c])
        min(plus, minus) <= atol || error("D_S D_T = ±D_ST fails by $(min(plus, minus)): the " *
            "operators do not form a (projective) linear representation")
        row = falses(length(ops))
        row[pos[a]] ⊻= true; row[pos[b]] ⊻= true; row[pos[c]] ⊻= true
        push!(rows, row); push!(rhs, minus < plus)
    end
    gf2_rows = BitMatrix(reduce(hcat, rows)')
    rhs = BitVector(rhs)
    pivot_cols = Int[]
    irow = 1
    for icol in eachindex(ops)
        p = findfirst(i -> gf2_rows[i, icol], irow:size(gf2_rows, 1))
        p === nothing && continue
        p += irow - 1
        gf2_rows[[irow, p], :] = gf2_rows[[p, irow], :]
        rhs[[irow, p]] = rhs[[p, irow]]
        for i in axes(gf2_rows, 1)
            if i != irow && gf2_rows[i, icol]
                gf2_rows[i, :] .⊻= gf2_rows[irow, :]
                rhs[i] ⊻= rhs[irow]
            end
        end
        push!(pivot_cols, icol)
        irow += 1
    end
    any(rhs[irow:end]) && error("the sign cocycle is not a coboundary: the representation is projective")
    flip = falses(length(ops))
    for (i, icol) in enumerate(pivot_cols)
        flip[icol] = rhs[i]
    end
    D = Dict(isym => (flip[pos[isym]] ? -1 : 1) .* D_raw[isym] for isym in ops)
    for a in ops, b in ops
        norm(D[a] * D[b] - D[op_of[Tuple(symmetry.S[a] * symmetry.S[b])]]) <= atol ||
            error("D_S D_T != D_ST after fixing signs")
    end

    # One on-site operator per operation of `symmetry`; a time-reversal copy reuses its spatial partner
    operators = map(1:symmetry.nsym) do isym
        S = Tuple(symmetry.S[isym])
        haskey(op_of, S) || error("operation $isym has no spatial partner in `symmetry`")
        WannierObject([Vec3(0, 0, 0)], reshape(ComplexF64.(D[op_of[S]]), nw^2, 1))
    end
    SymmetryOperators(symmetry, operators)
end
