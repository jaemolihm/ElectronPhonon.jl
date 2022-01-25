using LinearAlgebra
using SparseArrays

export finite_difference_vectors

"""
    finite_difference_vectors(recip_lattice, ngrid)
Compute finite difference vectors and weights following the scheme of [1].
[1] N. Marzari and D. Vanderbilt, PRB 56, 12847 (1997)]
Output `bvecs` is in crystal coordinates.
TODO: Implement. (Currently hardcoded for fcc lattice)
"""
function finite_difference_vectors(recip_lattice, ngrid)
    bvecs = [Vec3(x) ./ ngrid for x in [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1), (0, 0, -1), (0, -1, 0), (-1, 0, 0), (-1, -1, -1)]]
    wbs = [59.08152174582381 * (ngrid[1] / 20)^2 for b in bvecs]
    bvecs_cart = Ref(recip_lattice) .* bvecs
    # Test completeness relation (Eq. (B1) of Ref. [1])
    @assert sum([b_cart * b_cart' .* wb for (b_cart, wb) in zip(bvecs_cart, wbs)]) ≈ I(3)
    (; bvecs, bvecs_cart, wbs)
end


"""
Construct a vector of sparse matrices `∇` that maps a quantity defined on `el` to its
covariant derivative:
``(∇ᵅ * f)[ik] = ∑_b wb * bᵅ * m[ib, ik]' * f[ikb] * m[ib, ik] - i[ξ[ik], f[ik]]``,
where `ikb` is the index of `k + b` and `ξ` the position matrix in the eigenstate gauge.
Note that `ξ` is not the Berry connection: it does not include the Hamiltonian derivative term.
- `hdf_group`: If given, write data for the sparse matrix to file. If not given, return the
sparse matrix itself.
"""
function compute_covariant_derivative_matrix(el, el_k_save, bvec_data, hdf_group=nothing)
    indmap = EPW.states_index_map(el)
    kpts = el.kpts

    mmat = zeros(ComplexF64, el.nband, el.nband)

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

    # TODO: 2. position matrix term


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