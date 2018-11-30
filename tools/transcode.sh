#!/bin/bash

INPUTDIR=$1
OUTPUTDIR=$2

mkdir $OUTPUTDIR


for file in $INPUTDIR/*; do
    filename=$(basename -- "$file")
    stem="${filename%.*}"
    echo "Processing $filename"
    ffmpeg -i $file -map v:0 -c:v libx264 -crf 18 -pix_fmt yuv420p -g 5 -profile:v high $OUTPUTDIR/$stem.mp4
done
