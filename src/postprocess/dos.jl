export compute_dos
export plot_dos

# TODO: Tetrahedron

"""
    compute_dos(model, nks; η = 10.0 * unit_to_aru(:meV)) => (; elist, dos, pdos)

Compute the density of state ``DOS(e) = 1/Nk ∑_{n,k} δ(e - e_{nk})``
and WF-projected density of state ``PDOS(e) = 1/Nk ∑_{n,k} |U_{i,nk}⟩|^2 δ(e - e_{nk})``.
We use Gaussian smearing of width `η`.
"""
function compute_dos(model, nks; η = 100.0 * unit_to_aru(:meV), elist = nothing)
    kpts = kpoints_grid(nks)
    els = compute_electron_states(model, kpts, ["eigenvector"])

    if elist === nothing
        emax = maximum(maximum(el.e) for el in els)
        emin = minimum(minimum(el.e) for el in els)
        elist = range(emin - 10η, emax + 10η, step = η / 2)
    end

    dos = zeros(length(elist))
    pdos = zeros(length(elist), model.nw)

    dos_nk = zeros(length(elist))

    @views for (ik, el) in enumerate(els)
        for n in el.rng
            @. dos_nk = gaussian((elist - el.e[n]) / η) / η
            @. dos += dos_nk * kpts.weights[ik]
            for iw in 1:model.nw
                @. pdos[:, iw] += dos_nk * kpts.weights[ik] * abs(el.u[iw, n])^2
            end
        end
    end

    (; elist, dos, pdos)
end

function plot_dos(model, nks; pdos_inds = 1:model.nw, η = 100.0 * unit_to_aru(:meV), elist = nothing, close_fig=true)
    elist, dos, pdos = compute_dos(model, nks; η, elist)

    # Plot band structure
    fig, ax = PyPlot.subplots(1, 1)
    ax.plot(elist./ unit_to_aru(:eV), dos .* unit_to_aru(:eV); c = "k", label = "DOS")

    for i in pdos_inds
        ax.plot(elist ./ unit_to_aru(:eV), pdos[:, i] .* unit_to_aru(:eV); label = "PDOS $i")
    end
    
    ax.legend()
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("DOS (1 / eV)")
    display(fig)
    close_fig && close(fig)
    (; fig, ax, elist, dos, pdos)
end
