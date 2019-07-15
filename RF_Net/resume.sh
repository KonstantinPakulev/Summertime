#!/usr/bin/env bash
root="$(pwd)/runs"

time=$1
model=$2

save=""$root"/"$time""
resume=""$save"/model/"$model""

echo "Resume "$resume""

stdout=$save"/out.out"
stderr=$save"/out.err"

rm -f $stdout
rm -f $stderr

bsub -q normal -J k.pakulev -gpu "num=1:mode=exclusive_process" -o $stdout -e $stderr python ~/Summertime/RF_Net/train.py --save=$save --det-step=1 --des-step=2 --resume=$resume

