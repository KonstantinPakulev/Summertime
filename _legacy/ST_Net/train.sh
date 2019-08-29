#!/usr/bin/env bash

time="$(date "+%m_%d_%H_%M_%S")"
save="$(pwd)/runs/"${time}

mkdir $save
mkdir $save"/log"
mkdir $save"/model"
mkdir $save"/image"

echo "Saved to ""$save"""

stdout=$save"/out.out"
stderr=$save"/out.err"

rm -f $stdout
rm -f $stderr

bsub -q normal -J k.pakulev -gpu "num=1" -o $stdout -e $stderr python /Vol0/user/k.pakulev/Summertime/legacy/ST_Net/train.py --save=$save