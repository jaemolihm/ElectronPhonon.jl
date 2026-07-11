using Base.Threads: nthreads

"""
    EphOuterQLoopBuffers

Pre-allocated buffers for `run_eph_outer_q` that can be reused across multiple calls
to avoid repeated allocation of interpolator channels, `ElPhData` buffers, and `GridOpt` objects.

Create once and pass via the `eph_buffers` keyword argument to `run_eph_outer_q`.

    EphOuterQLoopBuffers(model; nchunks_threads=nthreads(), precompute_el_kq=false,
                         fourier_mode="gridopt", nband_max=model.nw)
"""
struct EphOuterQLoopBuffers{FT, IT <: AbstractWannierInterpolator}
    epdatas::Channel{ElPhData{FT}}
    ep_eRpq_obj::HostWannierObject{FT}
    ep_eRpqs::Channel{AbstractWannierInterpolator{FT}}
    epmat::IT
    ham_threads::Union{Nothing, Channel{AbstractWannierInterpolator{FT}}}
    vel_threads::Union{Nothing, Channel{AbstractWannierInterpolator{FT}}}
end

function EphOuterQLoopBuffers(model::Model{FT};
        nchunks_threads = nthreads(),
        precompute_el_kq = false,
        fourier_mode = "gridopt",
        nband_max = model.nw,
    ) where {FT}

    (; nw, nmodes) = model
    nbuffers = min(nthreads(), nchunks_threads)
    threads = nbuffers > 1

    epdatas = Channel{ElPhData{FT}}(nbuffers)
    foreach(1:nbuffers) do _
        put!(epdatas, ElPhData{FT}(nw, nmodes, nband_max))
    end

    ep_eRpq_obj = get_next_wannier_object(model.epmat)
    ep_eRpqs = get_interpolator_channel(ep_eRpq_obj; fourier_mode, nbuffers)

    epmat = get_interpolator(model.epmat; fourier_mode, threads)

    if !precompute_el_kq
        ham_threads = get_interpolator_channel(model.el_ham; fourier_mode, nbuffers)
        vel_threads = if model.el_velocity_mode === :Direct
            get_interpolator_channel(model.el_vel; fourier_mode, nbuffers)
        else
            get_interpolator_channel(model.el_ham_R; fourier_mode, nbuffers)
        end
    else
        ham_threads = nothing
        vel_threads = nothing
    end

    EphOuterQLoopBuffers(epdatas, ep_eRpq_obj, ep_eRpqs, epmat, ham_threads, vel_threads)
end
