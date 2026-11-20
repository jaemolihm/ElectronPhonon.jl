# Wannier-basis representation of the spatial symmetry operations, fitted from H(R).
# Included after model.jl (it builds `SymmetryOperators`). No loader fills `model.el_sym` for models
# read with `load_model_from_epw_new`, and the file reader of the removed `epw_data_julia_symmetry.bin`
# (EP #111) has no writer for them, so the operators are recovered from the Hamiltonian itself, and
# checked against the e-ph matrix when the model holds one.

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

If `model.epmat` is in memory (or memory-mapped), `D_S` is also checked against it, per spatial operation,
with `W = S_R`, `W_cart` its Cartesian form and the atom map `W τ_κ = τ_π(κ) + L_κ` (`L_κ` a lattice vector):

    g(W Re, W Rp + L_κ)[:, :, π(κ)β] = Σ_α W_cart[β, α] D_S g(Re, Rp)[:, :, κα] D_S'    for all Re, Rp, κ, β,

`g(Re, Rp)[m, n, ν] = ⟨m 0| ∂V/∂u_ν(Rp) |n Re⟩`, `ν = 3(κ-1) + α` (Cartesian displacements, `nmodes == 3 nat`).
For any other `nmodes` every mode is taken as invariant, `g(W Re, W Rp) = D_S g(Re, Rp) D_S'`. A missing
image `W Re` or `W Rp + L_κ` is a zero block, so only a nonzero source entry fails. The relative residual
`max ‖lhs - rhs‖ / max_{Re,Rp} ‖g(Re, Rp)‖` over `(Re, Rp)` blocks (all modes) must be at most `atol`, else an
error naming "epmat". With `model.epmat === nothing` the check is skipped with an `@info` note.
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

    # Check D_S against the e-ph matrix by the law in the docstring
    if model.epmat === nothing
        @info "fit_onsite_symmetry_operators: epmat not loaded; D_S checked against H only"
    else
        (; nmodes) = model
        epmat = model.epmat
        outer_el = model.epmat_outer_momentum == "el"
        irvec_e, irvec_p = outer_el ? (epmat.irvec, epmat.irvec_next) : (epmat.irvec_next, epmat.irvec)
        ne, np = length(irvec_e), length(irvec_p)
        index_e = Dict(R => i for (i, R) in enumerate(irvec_e))
        index_p = Dict(R => i for (i, R) in enumerate(irvec_p))
        nat = length(model.atom_pos)
        cartesian = nmodes == 3 * nat      # Cartesian displacement modes, else every mode is invariant
        modes_per_atom = cartesian ? 3 : nmodes
        natoms_loop = cartesian ? nat : 1
        # g as (nw, nw, nmodes, inner, outer): outer is Re for "el" and Rp for "ph"
        g = reshape(epmat.op_r, nw, nw, nmodes, outer_el ? np : ne, outer_el ? ne : np)
        ie_ip_to_g(ie, ip) = outer_el ? (ip, ie) : (ie, ip)
        frac_atoms = [model.lattice \ (model.alat * τ) for τ in model.atom_pos]
        @views for isym in ops
            symop = symmetry[isym]
            W = round.(Int, inv(symop.S)')
            Wcart = Matrix(symop.Scart)
            atom_image, L = collect(1:nat), fill(Vec3(0, 0, 0), nat)
            if cartesian
                # The real-space rotation of the fit (W in Cartesian coordinates) acts on the modes
                norm(Wcart - model.lattice * W * inv(model.lattice)) <= 1e-8 || error("operation $isym: " *
                    "Scart is not the Cartesian form of S_R, so the mode rotation of the epmat check is undefined")
                for κ in 1:nat
                    frac_image = W * frac_atoms[κ]
                    image = findfirst(κ2 -> all(isinteger_approx, frac_image - frac_atoms[κ2]), 1:nat)
                    image === nothing && error("operation $isym maps atom $κ off every atom: epmat check impossible")
                    atom_image[κ] = image
                    L[κ] = Vec3(round.(Int, frac_image - frac_atoms[image]))
                end
            end
            image_e = [get(index_e, Vec3(W * R), 0) for R in irvec_e]
            image_p = [[get(index_p, Vec3(W * Rp + L[κ]), 0) for Rp in irvec_p] for κ in 1:nat]
            Dop, Dopt = D[isym], collect(D[isym]')
            tasks = map(Iterators.partition(1:ne, cld(ne, 4 * Threads.nthreads()))) do ies
                Threads.@spawn begin
                    T = [zeros(ComplexF64, nw, nw) for _ in 1:modes_per_atom]
                    tmp, diff = zeros(ComplexF64, nw, nw), zeros(ComplexF64, nw, nw)
                    resid_max, gscale_max = 0.0, 0.0
                    for ie in ies, ip in 1:np
                        resid2, gscale2 = 0.0, 0.0
                        for κ in 1:natoms_loop
                            ie2, ip2 = image_e[ie], image_p[κ][ip]
                            off, off2 = cartesian ? (3 * (κ - 1), 3 * (atom_image[κ] - 1)) : (0, 0)
                            # 1. T_α = D g(Re, Rp)[κα] D'
                            for α in 1:modes_per_atom
                                G = g[:, :, off + α, ie_ip_to_g(ie, ip)...]
                                mul!(tmp, Dop, G)
                                mul!(T[α], tmp, Dopt)
                                gscale2 += sum(abs2, G)
                            end
                            # 2. residual of g(W Re, W Rp + L_κ)[π(κ)β] = Σ_α W_cart[β, α] T_α (zero if the image is missing)
                            for β in 1:modes_per_atom
                                if cartesian
                                    diff .= 0
                                    for α in 1:3
                                        diff .+= Wcart[β, α] .* T[α]
                                    end
                                else
                                    diff .= T[β]
                                end
                                if ie2 != 0 && ip2 != 0
                                    diff .-= g[:, :, off2 + β, ie_ip_to_g(ie2, ip2)...]
                                end
                                resid2 += sum(abs2, diff)
                            end
                        end
                        resid_max = max(resid_max, sqrt(resid2))
                        gscale_max = max(gscale_max, sqrt(gscale2))
                    end
                    (resid_max, gscale_max)
                end
            end
            out = fetch.(tasks)
            gscale = maximum(last, out)
            resid_g = gscale == 0 ? 0.0 : maximum(first, out) / gscale     # an all-zero epmat has nothing to violate
            resid_g <= atol || error("operation $isym: D_S does not satisfy the symmetry law of model.epmat, " *
                "relative residual $resid_g > atol = $atol")
        end
    end

    # One on-site operator per operation of `symmetry`; a time-reversal copy reuses its spatial partner
    operators = map(1:symmetry.nsym) do isym
        S = Tuple(symmetry.S[isym])
        haskey(op_of, S) || error("operation $isym has no spatial partner in `symmetry`")
        WannierObject([Vec3(0, 0, 0)], reshape(ComplexF64.(D[op_of[S]]), nw^2, 1))
    end
    SymmetryOperators(symmetry, operators)
end
