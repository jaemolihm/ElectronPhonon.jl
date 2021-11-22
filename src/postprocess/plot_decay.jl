import PyPlot

export plot_decay

function plot_decay(obj::AbstractWannierObject, lattice, ax=PyPlot.gca(); logscale=true, display_fig=true, close_fig=true)
    absR = norm.(Ref(lattice) .* obj.irvec)
    norms = real.(sqrt.(vec(sum(x -> abs2(x), obj.op_r; dims=1))))
    ax.plot(absR, norms, "o")
    ax.set_xlabel("R (Å)")
    ax.set_ylabel("norm of op(R)")
    if logscale
        # drop very small values
        ax.set_ylim([minimum(norms[norms .> 1e-10]) / 2, maximum(norms) * 2])
        ax.set_yscale("log")
    end
    fig = PyPlot.gcf()
    display_fig && display(fig)
    close_fig && close(fig)
    fig
end

function plot_decay(model::ModelEPW)
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