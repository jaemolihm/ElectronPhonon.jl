using LinearAlgebra
using StaticArrays
using SparseArrays

export finite_difference_vectors

"""
    finite_difference_vectors(recip_lattice, ngrid, order=1) => (; bvecs, bvecs_cart, wbs)
Choose the b vectors and weights as defined in [1], following the scheme of [2].
[1] N. Marzari and D. Vanderbilt, PRB 56, 12847 (1997)
[2] A. A. Mostofi et al, Phys. Commun. 178 685 (2008)

- `order`: Use higher-order formula with `order`-times more b vectors. Finite-difference error
           scales as ``O(b^(2*order))``.

TODO: Use gcd of ngrid to reduce ngrid.
"""
function finite_difference_vectors(recip_lattice::Mat3{FT}, ngrid; order=1) where {FT}
    nsupcell = 5 # Include b vectors in the [-nsupcell, nsupcell] cell to the search.
    kdist_tol = norm(recip_lattice) * sqrt(eps(FT)) # Difference below kdist_tol are regarded as equal.
    q = [1, 0, 0, 1, 0, 1] # Lower triangular components of I(3)

    # Generate list of b vectors and sort according to distance from origin
    rng = -nsupcell:nsupcell
    bvecs_all = vec([Vec3(i, j, k) ./ ngrid for i in rng, j in rng, k in rng])
    bvecs_cart_all = Ref(recip_lattice) .* bvecs_all
    dist_all = norm.(bvecs_cart_all)

    inds = sortperm(dist_all)
    bvecs_all = bvecs_all[inds]
    bvecs_cart_all = bvecs_cart_all[inds]
    dist_all = dist_all[inds]

    # Group the b vectors by the distance
    ishell_all = zeros(Int, length(bvecs_all))
    for i in 2:length(bvecs_all)
        if dist_all[i] - dist_all[i-1] > kdist_tol
            ishell_all[i] = ishell_all[i-1] + 1
        else
            ishell_all[i] = ishell_all[i-1]
        end
    end
    nshell = maximum(ishell_all)

    found = false
    nshell_used = 0
    Amat = zeros(FT, 6, nshell)
    wbs_shell = zeros(FT, nshell)
    ishell_used = zeros(Bool, nshell)

    # Loop over shells.
    for ishell in 1:nshell
        nshell_used += 1
        Amat[:, nshell_used] .= @views _compute_A(bvecs_cart_all[findall(ishell_all .== ishell)])
        @views A = Amat[:, 1:nshell_used]

        # check whether the new shell is linearly dependent on existing ones
        _, s, _ = svd(A)
        if any(s .< sqrt(eps(FT))) # This shell will not be used.
            nshell_used -= 1
            continue
        end
        ishell_used[ishell] = true
        @info "finite_difference_vectors: using shell $ishell"

        # Try to solve Amat * w = q
        w = A \ q

        if A * w ≈ q
            # b vectors are found. Exit loop.
            found = true
            wbs_shell[ishell_used] .= w
            break
        end
    end

    if ! found
        error("b vector search failed. Maybe the lattice is very skewed.")
    end

    bvecs = empty(bvecs_all)
    wbs = empty(wbs_shell)
    for ishell in 1:nshell
        if ishell_used[ishell]
            bvecs_new = bvecs_all[findall(ishell_all .== ishell)]
            append!(bvecs, bvecs_new)
            append!(wbs, fill(wbs_shell[ishell], length(bvecs_new)))
        end
    end

    # Higher-order finite difference
    if order == 1
        # First-order: keep b and wbs (do nothing)
    elseif order == 2
        # [1 0] / [1 1; 2^2 2^4]
        bvecs = vcat(bvecs, 2. * bvecs)
        wbs = vcat(wbs .* 4/3, wbs .* -1/12)
    elseif order == 3
        # [1 0 0] / [1 1 1; 2^2 2^4 2^6; 3^2 3^4 3^6]
        bvecs = vcat(bvecs, 2. * bvecs, 3. * bvecs)
        wbs = vcat(wbs .* 3/2, wbs .* -3/20, wbs .* 1/90)
    elseif order == 4
        # [1 0 0 0] / [1 1 1 1; 2^2 2^4 2^6 2^8; 3^2 3^4 3^6 3^8; 4^2 4^4 4^6 4^8]
        bvecs = vcat(bvecs, 2. * bvecs, 3. * bvecs, 4. * bvecs)
        wbs = vcat(wbs .* 8/5, wbs .* -1/5, wbs .* 8/315, wbs .* -1/560)
    else
        # Create Vandermonde matrix and compute [1 0 ...] / V.
        V = hcat([(1:order).^(2*n) for n in 1:order]...)
        x = zeros(1, order)
        x[1] = 1
        coeffs = x / V
        bvecs = vcat([bvecs .* n for n in 1:order]...)
        wbs = vcat([wbs .* c for c in coeffs]...)
    end

    bvecs_cart = Ref(recip_lattice) .* bvecs

    # Test completeness relation (Eq. (B1) of Ref. [1])
    @assert sum([b_cart * b_cart' .* wb for (b_cart, wb) in zip(bvecs_cart, wbs)]) ≈ I(3)
    (; bvecs, bvecs_cart, wbs)
end

# A[j] = ∑_b b^a b^b, j = (a, b) = (1, 1), (2, 1), (3, 1), (2, 2), (3, 2), (3, 3)
function _compute_A(bvecs_cart)
    A = sum(b * b' for b in bvecs_cart)
    SVector(A[1, 1], A[2, 1], A[3, 1], A[2, 2], A[3, 2], A[3, 3])
end

"""
Construct a vector of sparse matrices `∇` that maps a quantity defined on `el` to its
covariant derivative:
``(∇ᵅ * f)[ik] = ∑_b wb * bᵅ * m[ib, ik]' * f[ikb] * m[ib, ik] - i[ξ[ik], f[ik]]``,
where `ikb` is the index of `k + b` and `ξ` the position matrix in the eigenstate gauge.
Note that `ξ` is not the Berry connection: it does not include the Hamiltonian derivative term.
If using symmetry (`el_sym !== nothing`), ∇ is calculated for `el`, which is on the full k grid,
not `el_irr`, which is on the irreducible k grid.

- `el`: QMEStates for the unfolded (full) k point grid.
- `ik_to_ikirr_isym`: Mapping from ik on the full k grid to (ikirr, isym).
- ∇[1], ∇[2], ∇[3] correspond to Cartesian x, y, z directions.
- `hdf_group`: If given, write data for the sparse matrix to file. If not given, return the
sparse matrix itself.
"""
function compute_covariant_derivative_matrix(el_irr::EPW.QMEStates{FT}, el_irr_states, bvec_data,
        el_sym, el, ik_to_ikirr_isym; hdf_group=nothing, fourier_mode="gridopt") where FT

    nw = first(el_irr_states).nw

    if el_sym !== nothing
        # Using symmetry. el_irr and el_irr_states are on the irreducible k grid, while el
        # is on the full k grid. ik_to_ikirr_isym is the mapping from the full grid to the
        # irreducible grid.

        # Compute symmetry gauge matrix. For Sk = S_isym * k, the eigenstate at Sk is
        # U(Sk) = S_isym(k) * U(k). (Here, k is a point in the irreducible grid.)
        smat_all = zeros(Complex{FT}, nw, nw, el.kpts.n)
        @views for ik in 1:el.kpts.n
            ikirr, isym = ik_to_ikirr_isym[ik]
            # TODO: Optimize by skipping if symop is identity
            get_fourier!(smat_all[:, :, ik], el_sym.operators[isym], el_irr.kpts.vectors[ikirr], mode=fourier_mode)
        end
    end

    kpts = el.kpts

    # FIXME: el.nband instead of rng_maxdoes not work because rng can be outside of 1:el.nband
    # rng_max is a dirty fix...
    rng_max = maximum(x -> x.rng[end], el_irr_states)
    mmat = zeros(Complex{FT}, rng_max, rng_max)
    u_k  = zeros(Complex{FT}, nw, rng_max)
    u_kb = zeros(Complex{FT}, nw, rng_max)

    # FIXME: Simplify
    nband_ignore = el.nband_ignore

    sp_i = Int[]
    sp_j = Int[]
    sp_vals = [Complex{FT}[] for _ in 1:3]

    # 1. Derivative term
    for ik in 1:kpts.n
        xk = kpts.vectors[ik]
        ikirr = ik_to_ikirr_isym[ik][1]
        rng_k = el_irr_states[ikirr].rng
        if el_sym !== nothing
            @views mul!(u_k[:, rng_k], smat_all[:, :, ik], el_irr_states[ikirr].u)
        else
            u_k[:, rng_k] .= el_irr_states[ikirr].u
        end

        for (b, b_cart, wb) in zip(bvec_data...)
            xkb = xk + b
            ikb = xk_to_ik(xkb, kpts)
            ikb === nothing && continue

            ikbirr = ik_to_ikirr_isym[ikb][1]
            rng_kb = el_irr_states[ikbirr].rng
            if el_sym !== nothing
                @views mul!(u_kb[:, rng_kb], smat_all[:, :, ikb], el_irr_states[ikbirr].u)
            else
                u_kb[:, rng_kb] .= el_irr_states[ikbirr].u
            end

            # Compute overlap matrix: mmat = U(k+b)' * U(k)
            @views mul!(mmat[rng_kb, rng_k], u_kb[:, rng_kb]', u_k[:, rng_k])

            for ib2 in rng_k, ib1 in rng_k
                ind_i = get_1d_index(el, ib1 + nband_ignore, ib2 + nband_ignore, ik)
                ind_i == 0 && continue
                for jb2 in rng_kb, jb1 in rng_kb
                    # ∇ᵅf[ik, ib1, ib2] += mkb'[ib1, jb1] * f[ikb, jb1, jb2] * mkb[jb2, ib2] * wb * bᵅ
                    ind_f = get_1d_index(el, jb1 + nband_ignore, jb2 + nband_ignore, ikb)
                    ind_f == 0 && continue
                    push!(sp_i, ind_i)
                    push!(sp_j, ind_f)
                    coeff = mmat[jb1, ib1]' * mmat[jb2, ib2] * wb
                    for idir in 1:3
                        push!(sp_vals[idir], coeff * b_cart[idir])
                    end
                end
            end
        end
    end

    # 2. position matrix term
    # (∇ * f)[ib1, ib2] += - im * (ξ[ib1, ib3] * f[ib3, ib2] - f[ib1, ib3] * ξ[ib3, ib2])
    for ik in 1:kpts.n
        ikirr, isym = ik_to_ikirr_isym[ik]
        rng = el_irr_states[ikirr].rng
        rbar = el_irr_states[ikirr].rbar

        for ib1 in rng, ib2 in rng
            ind_i = get_1d_index(el, ib1 + nband_ignore, ib2 + nband_ignore, ik)
            ind_i == 0 && continue
            for ib3 in rng
                ind_f = get_1d_index(el, ib3 + nband_ignore, ib2 + nband_ignore, ik)
                if ind_f != 0
                    push!(sp_i, ind_i)
                    push!(sp_j, ind_f)
                    if el_sym === nothing
                        rbar_mel = rbar[ib1, ib3]
                    else
                        rbar_mel = el_sym.symmetry[isym].Scart * rbar[ib1, ib3]
                        rbar_mel = el_sym.symmetry[isym].is_tr ? conj(rbar_mel) : rbar_mel
                    end
                    for idir in 1:3
                        push!(sp_vals[idir], -im * rbar_mel[idir])
                    end
                end

                ind_f = get_1d_index(el, ib1 + nband_ignore, ib3 + nband_ignore, ik)
                if ind_f != 0
                    push!(sp_i, ind_i)
                    push!(sp_j, ind_f)
                    if el_sym === nothing
                        rbar_mel = rbar[ib3, ib2]
                    else
                        rbar_mel = el_sym.symmetry[isym].Scart * rbar[ib3, ib2]
                        rbar_mel = el_sym.symmetry[isym].is_tr ? conj(rbar_mel) : rbar_mel
                    end
                    for idir in 1:3
                        push!(sp_vals[idir], im * rbar_mel[idir])
                    end
                end
            end
        end
    end

    if hdf_group === nothing
        # Construct and return ∇ as a sparse matrix
        ∇ = [dropzeros!(sparse(sp_i, sp_j, sp_vals[1], el.n, el.n)),
              dropzeros!(sparse(sp_i, sp_j, sp_vals[2], el.n, el.n)),
              dropzeros!(sparse(sp_i, sp_j, sp_vals[3], el.n, el.n))]
        return ∇
    else
        # Write data needed to construct ∇ to file
        hdf_group["n"] = el.n
        hdf_group["I"] = sp_i
        hdf_group["J"] = sp_j
        hdf_group["V1"] = sp_vals[1]
        hdf_group["V2"] = sp_vals[2]
        hdf_group["V3"] = sp_vals[3]
        return [sparse([], [], Complex{FT}[])] # dummy output for return type stability
    end
end

"""
    compute_covariant_derivative_matrix(el_irr::EPW.QMEStates, el_irr_states, bvec_data; kwargs...)
Run without any symmetry.
"""
function compute_covariant_derivative_matrix(el_irr::EPW.QMEStates, el_irr_states, bvec_data;
                                             kwargs...)
    # Without symmetry
    el = el_irr
    ik_to_ikirr_isym = [(ik, 0) for ik in 1:el.kpts.n]
    compute_covariant_derivative_matrix(el_irr, el_irr_states, bvec_data, nothing, el,
                                        ik_to_ikirr_isym; kwargs...)
end

function load_covariant_derivative_matrix(f, ::Type{FT}=Float64) where FT
    n = read(f, "n")::Int
    sp_i = read(f, "I")::Vector{Int}
    sp_j = read(f, "J")::Vector{Int}
    sp_vals = [read(f, "V$i")::Vector{Complex{FT}} for i in 1:3]
    ∇ = [dropzeros!(sparse(sp_i, sp_j, sp_val, n, n)) for sp_val in sp_vals]
    ∇
end