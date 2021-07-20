set style data dots
set nokey
set xrange [0: 5.06617]
set yrange [-10.14954 : 29.62070]
set arrow from  1.73843, -10.14954 to  1.73843,  29.62070 nohead
set arrow from  2.60765, -10.14954 to  2.60765,  29.62070 nohead
set arrow from  3.22228, -10.14954 to  3.22228,  29.62070 nohead
set xtics ("G"  0.00000,"X"  1.73843,"W"  2.60765,"K"  3.22228,"G"  5.06617)
 plot "bn_band.dat"
