# Benchmark: electron-phonon Wannier->Bloch interpolation, per-(k,q) vs list-batched, CPU vs GPU.
#
#   RR_to_kR : Fourier of the (large) e-ph operator over R_el + rotation by uk, for nk k-points
#   kR_to_kq : Fourier over R_ep + rotation by ukq, u_ph, for nq q-points (fixed k)
#
# Compares one call per point (a batch of ONE — the batched drivers are list-only, there is no
# single-point entry point) against one call for the whole list
# (get_eph_RR_to_kR_batched! / get_eph_kR_to_kq_batched!).
#
# Run with both ElectronPhonon (this gpu branch) and CUDA in the environment:
#   julia --project=<env> benchmark/bench_eph_gpu.jl

using ElectronPhonon
using CUDA
using LinearAlgebra
using ElectronPhonon: WannierObject, Vec3, to_device,
    get_eph_RR_to_kR_batched!, get_eph_kR_to_kq_batched!
using Printf

const PB_FOLDER = "/mnt/home/jlihm/ceph/superconductivity/Pb/tutorial/1_epw/"
model = ElectronPhonon.load_model_from_epw_new(PB_FOLDER, "temp", "pb"; epmat_outer_momentum="el")
nw = model.nw; nmodes = model.nmodes; nr_ep = length(model.epmat.irvec_next); nband = nw
@printf "Model: nw=%d, nmodes=%d, nr_ep=%d   CUDA: %s\n\n" nw nmodes nr_ep CUDA.functional()

nk = 64; nq = 64
ks = [Vec3(rand(3)...) for _ in 1:nk]
qs = [Vec3(rand(3)...) for _ in 1:nq]
uks  = cat([Matrix(qr(rand(ComplexF64, nw, nw)).Q) for _ in 1:nk]...; dims=3)
ukqs = cat([Matrix(qr(rand(ComplexF64, nw, nw)).Q) for _ in 1:nq]...; dims=3)
uphs = cat([rand(ComplexF64, nmodes, nmodes) for _ in 1:nq]...; dims=3)

cput(f) = (f(); minimum(@elapsed(f()) for _ in 1:3))
gput(f) = (CUDA.@sync f(); minimum(CUDA.@elapsed(CUDA.@sync f()) for _ in 1:3))

# ---- RR_to_kR over nk k-points ----
epmat_c   = model.epmat
epmat_cit = get_interpolator(epmat_c; fourier_mode="batched", batch_size=1)   # per-k Fourier
epmat_ck  = get_interpolator(epmat_c; fourier_mode="batched", batch_size=nk)
epmat_g   = to_device(ElectronPhonon.gpu_backend(), epmat_c)
epmat_git = get_interpolator(epmat_g; fourier_mode="batched", batch_size=1)
epmat_gk  = get_interpolator(epmat_g; fourier_mode="batched", batch_size=nk)
uks_g = CuArray(uks)

# `ep_ekpR_all` is (ndata, nr_ep, nk) — one k per trailing slice (see get_eph_RR_to_kR_batched!).
ep_all_c = zeros(ComplexF64, nw*nband*nmodes, nr_ep, nk)
ep_all_g = CUDA.zeros(ComplexF64, nw*nband*nmodes, nr_ep, nk)
ep_one_c = zeros(ComplexF64, nw*nband*nmodes, nr_ep, 1)
ep_one_g = CUDA.zeros(ComplexF64, nw*nband*nmodes, nr_ep, 1)

rr_perk!(out, itp, U) = for ik in 1:nk
    get_eph_RR_to_kR_batched!(out, itp, view(ks, ik:ik), @view U[:, :, ik:ik])
end

t = (cput(()->rr_perk!(ep_one_c, epmat_cit, uks)),  gput(()->rr_perk!(ep_one_g, epmat_git, uks_g)),
     cput(()->get_eph_RR_to_kR_batched!(ep_all_c, epmat_ck, ks, uks)),
     gput(()->get_eph_RR_to_kR_batched!(ep_all_g, epmat_gk, ks, uks_g)))
@printf "RR_to_kR (%d k)   per-k:  CPU %6.2f  GPU %6.2f ms  |  batched:  CPU %6.2f  GPU %6.2f ms\n" nk (t.*1e3)...

# ---- kR_to_kq over nq q-points (fixed k = ks[1]) ----
obj_k1_c = WannierObject(model.epmat.irvec_next, ep_all_c[:, :, 1])
obj_k1_g = to_device(ElectronPhonon.gpu_backend(), WannierObject(model.epmat.irvec_next, Array(ep_all_g)[:, :, 1]))
itp_c_perq = get_interpolator(obj_k1_c; fourier_mode="batched", batch_size=1)   # per-q Fourier
itp_g_perq = get_interpolator(obj_k1_g; fourier_mode="batched", batch_size=1)
itp_c1 = get_interpolator(obj_k1_c; fourier_mode="batched", batch_size=nq)
itp_g1 = get_interpolator(obj_k1_g; fourier_mode="batched", batch_size=nq)
uphs_g = CuArray(uphs); ukqs_g = CuArray(ukqs)
ep4c = zeros(ComplexF64, nw, nw, nmodes, nq); ep4g = CUDA.zeros(ComplexF64, nw, nw, nmodes, nq)

# All interpolators are built once, outside the timed closures (matching the RR_to_kR section).
kq_perq!(itp, EP, UPH, UKQ) = for iq in 1:nq
    get_eph_kR_to_kq_batched!(view(EP, :, :, :, iq:iq), itp, view(qs, iq:iq),
                              @view(UPH[:, :, iq:iq]), @view(UKQ[:, :, iq:iq]))
end

t = (cput(()->kq_perq!(itp_c_perq, ep4c, uphs, ukqs)),
     gput(()->kq_perq!(itp_g_perq, ep4g, uphs_g, ukqs_g)),
     cput(()->get_eph_kR_to_kq_batched!(ep4c, itp_c1, qs, uphs, ukqs)),
     gput(()->get_eph_kR_to_kq_batched!(ep4g, itp_g1, qs, uphs_g, ukqs_g)))
@printf "kR_to_kq (%d q)   per-q:  CPU %6.2f  GPU %6.2f ms  |  batched:  CPU %6.2f  GPU %6.2f ms\n" nq (t.*1e3)...

# NOTE: batching collapses thousands of per-point kernel launches into a few large ones.
# RR_to_kR moves the large e-ph operator (nw^2*nmodes*nr_ep x nr_el), a clear GPU win once
# batched. kR_to_kq's matrices are tiny for Pb (nw=4), so CPU is competitive there; the GPU
# pulls ahead for larger nw / nband / nmodes.
