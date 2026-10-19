# Helpers for tests on the flat tile of a real outer-k run: a collector calculator, host copies of a tile, and
# the item order written from the segment table alone (the layout contract), independent of the accessors.
using ElectronPhonon

mutable struct _TileCollector <: ElectronPhonon.AbstractCalculator end
ElectronPhonon.supports(::_TileCollector, ::Type{<:ElectronPhonon.LoopTag}) = true
ElectronPhonon.setup_calculator!(c::_TileCollector, backend, els_k, els_kq, phs; kwargs...) = c

# Host copies of what a tile hands out, and the independent reference of the scatters from them.
function _tile_host(tile)
    seg = tile.seg_host
    (; ep = Array(tile.ep), e = Array(tile.phs.e), imap_k = Array(tile.imap_k), imap_kq = Array(tile.imap_kq),
       ikq = Array(tile.ikq), ik = tile.ik, nm = tile.phs.nmodes, seg)
end

# `[(l, m, n, ν, j)]` of every item, the loops of the layout contract: classes in order, pairs, then `ν`,
# `n`, `m` (fastest).
function _items_by_loops(seg, nm)
    items = NTuple{5, Int}[]
    for c in axes(seg, 2), jj in 1:seg[ElectronPhonon.SEG_NPAIRS, c], ν in 1:nm, n in 1:seg[ElectronPhonon.SEG_BK, c],
            m in 1:seg[ElectronPhonon.SEG_BKQ, c]
        push!(items, (length(items) + 1, m, n, ν, seg[ElectronPhonon.SEG_PAIR, c] + jj))
    end
    items
end

function _scatter_reference(h, n_i, n_f, i0, ni_stride)
    g2 = zeros(h.nm, ni_stride, n_f); re = zeros(h.nm, ni_stride, n_f); im_out = zeros(h.nm, ni_stride, n_f)
    for (l, m, n, ν, j) in _items_by_loops(h.seg, h.nm)
        i, f = h.imap_k[n, h.ik], h.imap_kq[m, h.ikq[j]]
        (i > 0 && f > 0) || continue
        g2[ν, i - i0, f] = abs2(h.ep[l]) / (2 * h.e[ν, j])
        re[ν, i - i0, f] = real(h.ep[l])
        im_out[ν, i - i0, f] = imag(h.ep[l])
    end
    (g2, re, im_out)
end

