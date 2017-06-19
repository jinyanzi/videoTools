#!/bin/bash

# run the file first

if [ $# -lt 1 ];then
	echo "No video provided"
	exit
fi

vname=$(basename $1)
vname=${vname%.*}
outname=$vname.txt
echo "Produce mean intensity plot for "$1

if [[ ! -f $outname ]];then
	make meanIntensity
	./meanIntensity $1 "$outname"
fi

plotCommand="set title 'Mean Intensity of $vname';set xlabel 'Frame index'; set ylabel 'Intensity'; set term png; set output '$vname"_meanIntensity.png"'; plot '$outname' u 1:2 with linespoints title 'Left', '' u 1:3 with linespoints title 'right';"

echo $plotCommand | gnuplot
