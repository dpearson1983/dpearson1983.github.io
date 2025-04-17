set term pngcairo color enhanced font "CMUSansSerif,48" size 2880,2160 background "#404040"

set border lw 2 lc rgb "white"
set xtics tc rgb "white"
set ytics tc rgb "white"

set xlabel "{/CMUSansSerif-Italics m}" tc rgb "white"
set ylabel "{/CMUSansSerif-Italics b}" tc rgb "white"

set xrange [-1.4 to -0.7]
set yrange [3 to 6.5]

set mxtics 5
set mytics 5

# set output "test_plot.png"
#
# plot "trial.dat" using 1:2 every 10 notitle with points pt 6 ps 3.0 lw 4 lc rgb "green", "accepted.dat" using 1:2 every 10 notitle with points pt 7 ps 3.0 lc 1
#
# unset output

do for [i=1:10000] {
    set output sprintf("frames/metropolis_hastings_%05.0f.png", i)

    plot "accepted.dat" using 1:2 every 1::1::i notitle with points pt 7 ps 3.0, "" using 1:2 every 1::i::i notitle with points pt 7 ps 3.0 lc rgb "#57ABDC", "trial.dat" using 1:2 every 1::i::i notitle with points pt 6 ps 3.0 lw 4 lc rgb "green"

    unset output
}
