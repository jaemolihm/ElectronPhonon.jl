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
        # Second-order: include b and 2b, with weights 4/3 and -1/12
        # [1 0] / [1 1; 2^2 2^4]
        bvecs = vcat(bvecs, 2. * bvecs)
        wbs = vcat(wbs .* 4/3, wbs .* -1/12)
    elseif order == 3
        # Third-order: include b, 2b and 3b, with weights 3/2, -3/20, and 1/90
        # [1 0 0] / [1 1 1; 2^2 2^4 2^6; 3^2 3^4 3^6]
        bvecs = vcat(bvecs, 2. * bvecs, 3. * bvecs)
        wbs = vcat(wbs .* 3/2, wbs .* -3/20, wbs .* 1/90)
    else
        error("Only order 1, 2 and 3 are implemented, but got $order.")
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
- ∇[1], ∇[2], ∇[3] correspond to Cartesian x, y, z directions.
- `hdf_group`: If given, write data for the sparse matrix to file. If not given, return the
sparse matrix itself.
"""
function compute_covariant_derivative_matrix(el, el_k_save, bvec_data, hdf_group=nothing)
    indmap = EPW.states_index_map(el)
    kpts = el.kpts

    # FIXME: el.nband instead of rng_maxdoes not work because rng can be outside of 1:el.nband
    # rng_max is a dirty fix...
    rng_max = maximum(x -> x.rng[end], el_k_save)
    mmat = zeros(ComplexF64, rng_max, rng_max)

    sp_i = Int[]
    sp_j = Int[]
    sp_vals = [ComplexF64[] for _ in 1:3]

    # 1. Derivative term
    for ind_i in 1:el.n
        ik = el.ik[ind_i]
        ib1 = el.ib1[ind_i]
        ib2 = el.ib2[ind_i]
        xk = kpts.vectors[ik]
        rng_k = el_k_save[ik].rng

        for (b, b_cart, wb) in zip(bvec_data...)
            xkb = xk .+ b
            ikb = xk_to_ik(xkb, kpts)
            ikb === nothing && continue
            rng_kb = el_k_save[ikb].rng

            # Compute overlap matrix: mmat = U(k+b)' * U(k)
            mmat_rng = @views mmat[rng_kb, rng_k]
            mul!(mmat_rng, get_u(el_k_save[ikb])', get_u(el_k_save[ik]))

            for jb1 in rng_kb, jb2 in rng_kb
                # ∇ᵅf[ik, ib1, ib2] += mkb'[ib1, jb1] * f[ikb, jb1, jb2] * mkb[jb2, ib2] * wb * bᵅ
                ind_f = get(indmap, EPW.CI(jb1, jb2, ikb), -1)
                ind_f == -1 && continue
                push!(sp_i, ind_i)
                push!(sp_j, ind_f)
                coeff = mmat[jb1, ib1]' * mmat[jb2, ib2] * wb
                for idir in 1:3
                    push!(sp_vals[idir], coeff * b_cart[idir])
                end
            end
        end
    end

    # 2. position matrix term
    # (∇ * f)[ib1, ib2] += - im * (ξ[ib1, ib3] * f[ib3, ib2] - f[ib1, ib3] * ξ[ib3, ib2])
    for ik in 1:kpts.n
        rng = el_k_save[ik].rng
        rbar = el_k_save[ik].rbar

        for ib1 in rng, ib2 in rng
            ind_i = get(indmap, EPW.CI(ib1, ib2, ik), -1)
            ind_i == -1 && continue
            for ib3 in rng
                ind_f = get(indmap, EPW.CI(ib3, ib2, ik), -1)
                if ind_f != -1
                    push!(sp_i, ind_i)
                    push!(sp_j, ind_f)
                    for idir in 1:3
                        push!(sp_vals[idir], -im * rbar[ib1, ib3][idir])
                    end
                end

                ind_f = get(indmap, EPW.CI(ib1, ib3, ik), -1)
                if ind_f != -1
                    push!(sp_i, ind_i)
                    push!(sp_j, ind_f)
                    for idir in 1:3
                        push!(sp_vals[idir], im * rbar[ib3, ib2][idir])
                    end
                end
            end
        end
    end


    if hdf_group === nothing
        # Construct and return ∇ as a sparse matrix
        ∇ = [dropzeros!(sparse(sp_i, sp_j, sp_val, el.n, el.n)) for sp_val in sp_vals]
        return ∇
    else
        # Write data needed to construct ∇ to file
        hdf_group["n"] = el.n
        hdf_group["I"] = sp_i
        hdf_group["J"] = sp_j
        hdf_group["V1"] = sp_vals[1]
        hdf_group["V2"] = sp_vals[2]
        hdf_group["V3"] = sp_vals[3]
        return
    end
end

function load_covariant_derivative_matrix(f)
    n = read(f, "n")::Int
    sp_i = read(f, "I")::Vector{Int}
    sp_j = read(f, "J")::Vector{Int}
    sp_vals = [read(f, "V$i")::Vector{ComplexF64} for i in 1:3]
    ∇ = [dropzeros!(sparse(sp_i, sp_j, sp_val, n, n)) for sp_val in sp_vals]
    ∇
end