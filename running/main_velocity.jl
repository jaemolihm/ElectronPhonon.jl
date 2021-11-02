using PrettyPrint
using StaticArrays
using BenchmarkTools
using NPZ
using PyCall
using Profile
using LinearAlgebra
using Base.Threads
using Distributed
using Revise
BLAS.set_num_threads(1)

using Plots
push!(LOAD_PATH, "/home/jmlim/julia_epw/EPW.jl")
using EPW
using EPW.Diagonalize


folder = "/home/jmlim/julia_epw/silicon_nk6"
model = load_model(folder)

kpoints = generate_kvec_grid(1, 1, 10000)

nw = model.nw
hks = [zeros(ComplexF64, nw, nw) for i=1:nthreads()]
vks = [zeros(ComplexF64, nw, nw, 3) for i=1:nthreads()]
ek_save = zeros(Float64, nw, kpoints.n)
uk_save = zeros(ComplexF64, nw, nw, kpoints.n)
veldiagk_save = zeros(Float64, 3, nw, kpoints.n)

Threads.@threads :static for ik in 1:kpoints.n
# @time for ik in 1:kpoints.n
    hk = hks[Threads.threadid()]
    vk = vks[Threads.threadid()]
    xk = kpoints.vectors[ik]
    get_fourier!(hk, model.el_ham, xk, mode="normal")
    @views ek_save[:, ik] = solve_eigen_el!(uk_save[:, :, ik], hk)

    get_fourier!(vk, model.el_ham_R, xk, mode="normal")
    for idir in 1:3
        veldiagk_save[idir, :, ik] .= real(diag(uk_save[:,:,ik]' * vk[:,:,idir] * uk_save[:,:,ik]))
    end
end # ik

xk = [k[3] for k in kpoints.vectors]

de = (ek_save[1, 3:end] - ek_save[1, 1:end-2]) ./ ((xk[3] - xk[1]) * (2π/10.3490))

plot(xk, ek_save[1, :])
plot(xk[2:end-1], de)
plot!(xk, vec([-1, 1, -1]' * veldiagk_save[:, 1, :]))


function test(a)
    a .+= 1
    return
end
a = 3:5
@show a
test(a)
