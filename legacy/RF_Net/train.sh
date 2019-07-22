#!/usr/bin/env bash

# see config for patch loss options
use_pl=${1:-True}
time="$(date "+%m_%d_%H_%M")"
save="$(pwd)/runs/"$time""

mkdir $save
mkdir ""$save"/log"
mkdir ""$save"/model"
mkdir ""$save"/image"

echo "Saved to "$save""

stdout=$save"/out.out"
stderr=$save"/out.err"

rm -f $stdout
rm -f $stderr

bsub -q normal -J k.pakulev -gpu "num=1:mode=exclusive_process" -o $stdout -e $stderr python ~/Summertime/RF_Net/train.py --save=$save --det-step=1 --des-step=2 --use-pl=$use_pl

