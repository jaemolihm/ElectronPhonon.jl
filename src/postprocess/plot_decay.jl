import PyPlot

export plot_decay
export plot_decay_eph

function plot_decay(obj::AbstractWannierObject, lattice, ax=PyPlot.gca(); logscale=true, display_fig=true, close_fig=true)
    absR = norm.(Ref(lattice) .* obj.irvec)
    norms = real.(sqrt.(vec(sum(x -> abs2(x), obj.op_r; dims=1))))
    ax.plot(absR, norms, "o")
    ax.set_xlabel("R (Å)")
    ax.set_ylabel("norm of op(R)")
    if logscale
        # drop very small values
        if any(norms .> 1e-10)
            ax.set_ylim([minimum(norms[norms .> 1e-10]) / 2, maximum(norms) * 2])
        end
        ax.set_yscale("log")
    end
    fig = PyPlot.gcf()
    display_fig && display(fig)
    close_fig && close(fig)
    fig
end

function plot_decay(model::Model)
    operators = [:el_ham, :el_ham_R, :el_pos, :el_vel, :ph_dyn, :ph_dyn_R]
    fig, plotaxes = PyPlot.subplots(2, 3, figsize=(8, 5))
    for (ax, key) in zip(vec(plotaxes), operators)
        ax.set_title(String(key))
        getfield(model, key) === nothing && continue
        plot_decay(getfield(model, key), model.lattice, ax, display_fig=false, close_fig=false)
    end
    fig.tight_layout(pad=0.0)
    display(fig)
    close(fig)
    fig
end

function plot_decay_eph(model::Model)
    # epmat with outer momentum changed
    nR1 = length(model.epmat.irvec_next)
    nR2 = length(model.epmat.irvec)
    n = div(model.epmat.ndata, nR1)
    data = Base.ReshapedArray(model.epmat.op_r, (n, nR1, nR2), ())
    tmp = PermutedDimsArray(data, (1, 3, 2))
    data_new = Base.ReshapedArray(tmp, (n * nR2, nR1), ())
    epmat_new = ElectronPhonon.WannierObject(model.epmat.irvec_next, data_new)

    if model.epmat_outer_momentum == "el"
        labels = ["epmat(R_e)", "epmat(R_p)"]
    else
        labels = ["epmat(R_p)", "epmat(R_e)"]
    end

    fig, plotaxes = PyPlot.subplots(1, 2, figsize=(8, 3))
    for (ax, obj, key) in zip(vec(plotaxes), [model.epmat, epmat_new], labels)
        ax.set_title(String(key))
        plot_decay(obj, model.lattice, ax, display_fig=false, close_fig=false)
    end
    fig.tight_layout(pad=0.0)
    display(fig)
    close(fig)
    fig
end
