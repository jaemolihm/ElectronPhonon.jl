using IterativeSolvers

"""
Reset GMRES solver iterable with given `x` and `b`.
See https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl/issues/166
and https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl/issues/276
for related discussions.
"""
function reset_gmres_iterable!(it::IterativeSolvers.GMRESIterable, x, b;
                               reltol::Real = zero(real(eltype(b))),
                               abstol::Real = sqrt(eps(real(eltype(b)))),
                               initially_zero::Bool = false,)
    it.x .= x
    it.b .= b
    if ! initially_zero
        it.mv_products += 1
    end
    it.residual.current = IterativeSolvers.init!(it.arnoldi, it.x, it.b, it.Pl, it.Ax;
                        initially_zero)
    IterativeSolvers.init_residual!(it.residual, it.residual.current)
    it.tol = max(reltol * it.residual.current, abstol)
    it.β = it.residual.current
end