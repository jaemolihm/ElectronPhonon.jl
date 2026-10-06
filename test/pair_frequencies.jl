# The phonon frequency `[ν, i, f]` of every state pair of a `G2Calculator` / `EPElementCalculator`,
# gathered from its phonon table and q index. A contract entry's `reference` (a NamedTuple) holds
# the reference loop's own per-pair frequencies as `ωq`.
pair_ω(c) = ElectronPhonon.gather_pair_table!(zeros(eltype(c.ωph), c.nmodes, c.el_i.n, c.el_f.n),
    c.ωph, c.iq_kk, c.el_i.iks, c.el_f.iks)
pair_ω(ref::NamedTuple) = ref.ωq
