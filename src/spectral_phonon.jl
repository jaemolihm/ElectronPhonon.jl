
# Spectral function of phonons

using Base.Threads

export PhononSpectralParams
export PhononSpectralData
export compute_phonon_spectral!

Base.@kwdef struct PhononSpectralParams{T <: Real}
    Tlist::Vector{T} # Temperature
    μ::T # Fermi level
    smearing::T # Smearing parameter for delta function
    degeneracy::T # degeneracy of electron bands (e.g. spin degeneracy)
    ωlist::Vector{T} # Frequency to calculate the spectral function
end

# Data and buffers for phonon spectral function
Base.@kwdef struct PhononSpectralData{T <: Real}
    selfen::Array{Complex{T}, 4}
    selfen_static::Array{Complex{T}, 3}
    specfun::Array{Complex{T}, 4}
    buffer::Vector{Complex{T}}
end

function PhononSpectralData(params::PhononSpectralParams, nmodes, nq)
    nω = length(params.ωlist)
    nT = length(params.Tlist)
    T = typeof(params).parameters[1]
    PhononSpectralData{T}(
        selfen=zeros(Complex{T}, nω, nmodes, nT, nq),
        selfen_static=zeros(Complex{T}, nmodes, nT, nq),
        specfun=zeros(Complex{T}, nω, nmodes, nT, nq),
        buffer=zeros(Complex{T}, nω),
    )
end

"""
    function compute_phonon_spectral!(phspec::PhononSpectralData, epdata,
        params::PhononSpectralParams, iq)
Compute phonon spectral function for given k and q point data in epdata.
Implements Eq.(145) of RMP 89, 015003 (2017) (except for using g instead of g^b)
Currently, only the diagonal self-energy is implemented.
TODO: off-diagonal self-energy
"""
@timing "spectral_ph" function compute_phonon_spectral!(phspec::PhononSpectralData, epdata,
        params::PhononSpectralParams, iq)
    f_k = epdata.el_k.occupation
    f_kq = epdata.el_kq.occupation

    buffer = phspec.buffer
    ωlist = params.ωlist
    μ = params.μ
    η = params.smearing
    inv_η = 1 / η

    # Calculate static and dynamic phonon self-energy
    @inbounds for ib in epdata.el_k.rng, jb in epdata.el_kq.rng
        δe = epdata.el_kq.e[jb] - epdata.el_k.e[ib]
        update_buffer = true # Need to update buffer because ib or jb changed.

        for (iT, T) in enumerate(params.Tlist)
            set_occupation!(epdata.el_k, μ, T)
            set_occupation!(epdata.el_kq, μ, T)

            delta_occ = f_kq[jb] - f_k[ib]
            if abs(delta_occ) < 1E-10
                continue
            end

            # Calculation of buffer is the most costly step. Reuse buffer for different
            # temperatures. Update buffer only if ib or jb has changed.
            # This block is inside the iT loop to avoid calculating buffer if delta_occ
            # is zero for all temperatures.
            if update_buffer
                for (iω, ω) in enumerate(ωlist)
                    # Option 1: Lorentizan
                    # buffer[iω] = 1 / (δe - ω - im * η)

                    # Option 2: real part is the same, imaginary part is replaced with
                    # Gaussian, using Im[1/(x-iη)] ≈ π δ(x) ≈ π * gaussian(x/η) / η.
                    buffer[iω] = (δe - ω) / ((δe - ω)^2 + η^2)
                    buffer[iω] += im * π * gaussian((δe - ω) * inv_η) * inv_η
                end
                update_buffer = false
            end

            # selfen_static = numerator * real(1 / (δe + im * η))
            # selfen(ω) = numerator / (δe - ω + im * η)
            for imode in 1:epdata.nmodes
                if epdata.ph.e[imode] < omega_acoustic
                    continue
                end

                numerator = epdata.g2[jb, ib, imode] * epdata.wtk * delta_occ
                phspec.selfen_static[imode, iT, iq] += numerator * δe / (δe^2 + η^2)
                for (iω, ω) in enumerate(ωlist)
                    phspec.selfen[iω, imode, iT, iq] += numerator * buffer[iω]
                end
            end
        end # mode
    end # temperature
end


"""
Calculate phonon Green function from the phonon energies and the self-energies.
Implements Eq.(136) of RMP 89, 015003 (2017)
Currently, only the diagonal self-energy is implemented.
TODO: off-diagonal self-energy
"""
function calculate_phonon_green(ωlist, energy, selfen)
    @. selfen = complex(real(selfen), -abs(imag(selfen)))

    nω, nmodes, nT, nq = size(selfen)
    @assert size(energy) == (nmodes, nq)
    @assert size(ωlist) == (nω,)
    green = zero(selfen)
    for iq in 1:nq
        for iT in 1:nT
            for imode in 1:nmodes
                e = energy[imode, iq]
                @inbounds for iω in 1:nω
                    green[iω, imode, iT, iq] = (2 * e
                        / (ωlist[iω]^2 - e^2 - 2 * e * selfen[iω, imode, iT, iq]))
                end
            end
        end
    end
    green
end