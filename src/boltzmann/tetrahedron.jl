"""
Routines for tetrahedron integration
"""

const KLIST = [SVector{3,Int}(-1, -1, -1),
               SVector{3,Int}(+1, -1, -1),
               SVector{3,Int}(-1, +1, -1),
               SVector{3,Int}(+1, +1, -1),
               SVector{3,Int}(-1, -1, +1),
               SVector{3,Int}(+1, -1, +1),
               SVector{3,Int}(-1, +1, +1),
               SVector{3,Int}(+1, +1, +1)]

export delta_tetrahedron
export delta_parallelepiped
export delta_tetrahedron_weights

@inline function delta_tetrahedron(etarget, e1234)
    """
    Calcultate val = int_{tetrahedron} d^3k delta(etarget - e(k)) / vol
    where e(k) is a linear function, e1, ..., e4 are the values of e(k) at
    the four vertices of the tetrahedron, and vol is the volume of the tetrahedron.

    Ref: Eqs. (7-10) of P. B. Allen, Phys. Stat. Sol. (b) 120, 629 (1983).
    Note in case B of Eq. (8), we use an equivalent formula which is more stable
    when (e2 - e1) is small.

    # Input
    - etarget: energy where the delta function becomes nonzero
    - e1234: Array containing energy at the four vertices of the tetrahedron
    """
    e1, e2, e3, e4 = sort(e1234)
    if e1 < etarget <= e2
        c1 = 3 / ((e2 - e1) * (e3 - e1) * (e4 - e1))
        return (etarget - e1)^2 * c1
    elseif e2 < etarget <= e3
        e31 = e3 - e1
        e41 = e4 - e1
        c2 = 3 / (e41 * e31)
        c3 = 3 * (e4 + e3 - e2 - e1) / (e41 * e31 * (e4 - e2) * (e3 -e2))
        return (2*etarget - e1 - e2) * c2 - (etarget - e2)^2 * c3
    elseif e3 < etarget <= e4
        c4 = 3 / ((e4 - e1) * (e4 - e2) * (e4 - e3))
        return (e4 - etarget)^2 * c4
    else
        # etarget <= e1 or e4 < etarget.
        return zero(etarget)
    end
end


@inline function delta_parallelepiped(etarget::T, e0, v0, L) where {T}
    """
    Calcultate val = int_{parallelepiped} d^3k delta(`etarget` - e(k)) / volume
    where e(k) = `e0` + k ⋅ `v0`. The parallelepiped region is [-L/2, L/2]^3.
    Divide the parallelepiped into six tetrahedra and use tetrahedron integration.
    """
    if norm(v0) < 1E-10
        # Velocity is too small. Tetrahedron method does not work.
        return zero(T), 0
    end
    L_div_2 = L ./ 2
    v0_L = v0 .* L_div_2
    e1 = e0 - v0_L[1] - v0_L[2] - v0_L[3]
    e2 = e0 + v0_L[1] - v0_L[2] - v0_L[3]
    e3 = e0 - v0_L[1] + v0_L[2] - v0_L[3]
    e4 = e0 + v0_L[1] + v0_L[2] - v0_L[3]
    e5 = e0 - v0_L[1] - v0_L[2] + v0_L[3]
    e6 = e0 + v0_L[1] - v0_L[2] + v0_L[3]
    e7 = e0 - v0_L[1] + v0_L[2] + v0_L[3]
    e8 = e0 + v0_L[1] + v0_L[2] + v0_L[3]
    val = zero(T)
    val += delta_tetrahedron(etarget, SVector{4,T}(e1, e2, e4, e5))
    val += delta_tetrahedron(etarget, SVector{4,T}(e1, e3, e4, e5))
    val += delta_tetrahedron(etarget, SVector{4,T}(e2, e4, e6, e5))
    val += delta_tetrahedron(etarget, SVector{4,T}(e3, e4, e7, e5))
    val += delta_tetrahedron(etarget, SVector{4,T}(e4, e7, e8, e5))
    val += delta_tetrahedron(etarget, SVector{4,T}(e4, e6, e8, e5))
    return val / 6
end

"""
Calcultate val = int_{parallelepiped} d^3k delta(`etarget` - e(k)) / volume
where `e(k) = e0 + k ⋅ v0`. The parallelepiped region is [-L/2, L/2]^3.
Divide the parallelepiped into six tetrahedra and use tetrahedron integration.
# Sampling
Sample points on the `e(k) = etarget` plane, where the spacing is relative to the edge
lengths. Sample around `nsample_1d` points along each dimension, and at most
`(nsample_1d+1)^2` points are sampled. Sampled points are stored in the array `ksamples`.
"""
@inline function delta_parallelepiped_sampling!(etarget::T, e0, v0, L, nsample_1d, ksamples) where {T}
    if norm(v0) < 1E-10
        # Velocity is too small. Tetrahedron method does not work.
        return zero(T), 0
    end

    L_div_2 = L ./ 2
    e1 = e0 + (-v0[1] * L_div_2[1] - v0[2] * L_div_2[2] - v0[3] * L_div_2[3])
    e2 = e0 + (+v0[1] * L_div_2[1] - v0[2] * L_div_2[2] - v0[3] * L_div_2[3])
    e3 = e0 + (-v0[1] * L_div_2[1] + v0[2] * L_div_2[2] - v0[3] * L_div_2[3])
    e4 = e0 + (+v0[1] * L_div_2[1] + v0[2] * L_div_2[2] - v0[3] * L_div_2[3])
    e5 = e0 + (-v0[1] * L_div_2[1] - v0[2] * L_div_2[2] + v0[3] * L_div_2[3])
    e6 = e0 + (+v0[1] * L_div_2[1] - v0[2] * L_div_2[2] + v0[3] * L_div_2[3])
    e7 = e0 + (-v0[1] * L_div_2[1] + v0[2] * L_div_2[2] + v0[3] * L_div_2[3])
    e8 = e0 + (+v0[1] * L_div_2[1] + v0[2] * L_div_2[2] + v0[3] * L_div_2[3])
    val = zero(T)
    val += delta_tetrahedron(etarget, SVector{4,T}(e1, e2, e4, e5))
    val += delta_tetrahedron(etarget, SVector{4,T}(e1, e3, e4, e5))
    val += delta_tetrahedron(etarget, SVector{4,T}(e2, e4, e6, e5))
    val += delta_tetrahedron(etarget, SVector{4,T}(e3, e4, e7, e5))
    val += delta_tetrahedron(etarget, SVector{4,T}(e4, e7, e8, e5))
    val += delta_tetrahedron(etarget, SVector{4,T}(e4, e6, e8, e5))

    # TODO: If |v0| = 0, skip sampling

    # Find one point on the e(k0) = etarget plane
    e = (e1, e2, e3, e4, e5, e6, e7, e8)
    imin = argmin(e)
    imax = argmax(e)
    emin = e[imin]
    emax = e[imax]
    # Trivial case: etarget outside the energy range
    if etarget < emin || emax < etarget
        return zero(T), 0
    end

    kmin, kmax = KLIST[imin], KLIST[imax]
    ratio = (etarget - emin) / (emax - emin)
    k0 = (kmin + (kmax - kmin) * ratio) .* L_div_2

    # Find two orthogonal unit vectors on the plane
    i = argmin(abs.(v0))
    if i == 1
        u0 = Vec3{T}(1, 0, 0)
    elseif i == 2
        u0 = Vec3{T}(0, 1, 0)
    else
        u0 = Vec3{T}(0, 0, 1)
    end
    u1 = cross(v0, u0)
    u2 = cross(v0, u1)

    # Normalize unit vectors according to the step size
    u1 = u1 / norm(u1 ./ L) / nsample_1d
    u2 = u2 / norm(u2 ./ L) / nsample_1d

    # k vectors on the e(k) = etarget plane is k0 + n1 * u1 + n2 * u2.
    # Sample k vectors on the plane and inside the cube.

    nmax = floor(Int, sqrt(3) * nsample_1d)
    nsamples = 0
    for n1 in -nmax:nmax
        was_inside = false
        k = k0 + n1 * u1 - nmax * u2
        for n2 in -nmax:nmax
            k += u2
            if all(abs.(k) .<= L_div_2)
                was_inside = true
                nsamples += 1
                ksamples[:, nsamples] .= k
            elseif was_inside
                # Was inside the cube, but moved out. The remaining points are all outside.
                break
            end
        end
    end

    return val / 6, nsamples
end


function delta_tetrahedron_weights(etarget, e1234)
    """
    Calcultate weights for evaluating I = int_{tetrahedron} d^3k f(k) delta(etarget - e(k)) / vol
    where e(k) is a linear function, e1, ..., e4 are the values of e(k) at
    the four vertices of the tetrahedron, and vol is the volume of the tetrahedron.
    With the output `w1234`, `I = sum_{i=1}^{4} f(k_i) w_i` holds.

    Ref: Appendix B of Blöchl et al, PRB 16 232 (1994). Since the reference is for step functions,
    we take the derivative with respect to e_F in all equations in the reference.

    # Input
    - etarget: energy where the delta function becomes nonzero
    - e1234: Array containing energy at the four vertices of the tetrahedron
    # Output
    - w1234: Array containing weights at the four vertices
    """
    inds = sortperm(e1234)
    e1, e2, e3, e4 = e1234[inds]
    if e1 < etarget <= e2
        c11 = (etarget - e1)^2 / ((e2 - e1) * (e3 - e1) * (e4 - e1))
        c12 = (etarget - e1) * c11
        w2 = c12 / (e2 - e1)
        w3 = c12 / (e3 - e1)
        w4 = c12 / (e4 - e1)
        w1 = 3 * c11 - (w2 + w3 + w4)
    elseif e2 < etarget <= e3
        # local variables
        # eei = etarget - ei
        ee1 = etarget - e1
        ee2 = etarget - e2
        ee3 = etarget - e3
        ee4 = etarget - e4
        # eij = ei - ej
        e41 = e4 - e1
        e31 = e3 - e1
        e42 = e4 - e2
        e32 = e3 - e2
        # equation (B11)
        c1  = ee1^2 / (4 * e41 * e31)
        dc1 = ee1 / (2 * e41 * e31)
        # equation (B12)
        c2  = - ee1 * ee2 * ee3 / (4 * e41 * e32 * e31 )
        dc2 = ( - ee2 * ee3 - ee1 * ee3 - ee1 * ee2 ) / (4 * e41 * e32 * e31)
        # equation (B13)
        c3  = - ee2^2 * ee4 / (4 * e42 * e32 * e41)
        dc3 = - (2 * ee2 * ee4 + ee2^2) / (4 * e42 * e32 * e41)

        # weights: derivatives of equations (B7-B10)
        w1 = dc1 - ((dc1 + dc2) * ee3 + c1 + c2) / e31 - ((dc1 + dc2 + dc3) * ee4 + c1 + c2 + c3) / e41
        w2 = dc1 + dc2 + dc3 - ((dc2 + dc3) * ee3 + c2 + c3) / e32 - (dc3 * ee4 + c3) / e42
        w3 = ((dc1 + dc2 ) * ee1 + c1 + c2) / e31 + ((dc2 + dc3) * ee2 + c2 + c3) / e32
        w4 = ((dc1 + dc2 + dc3) * ee1 + c1 + c2 + c3) / e41 + (dc3 * ee2 + c3) / e42
    elseif e3 < etarget <= e4
        c31 = (e4 - etarget)^2 / ((e4 - e1) * (e4 - e2) * (e4 - e3))
        c32 = (e4 - etarget) * c31
        w1 = c32 / (e4 - e1)
        w2 = c32 / (e4 - e2)
        w3 = c32 / (e4 - e3)
        w4 = 3 * c31 - (w1 + w2 + w3)
    else
        w1, w2, w3, w4 = zero(etarget), zero(etarget), zero(etarget), zero(etarget)
    end
    w1234 = SVector(w1, w2, w3, w4)
    # w1234 are the weights in the sorted index. Need to invert the indices to original index.
    w1234[sortperm(inds)]
end

# TODO: Blöchl correction (Eq. (22))
# TODO: Add tests (e.g. sum of weights == total weight)