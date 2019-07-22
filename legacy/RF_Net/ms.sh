#!/usr/bin/env bash

root="/Vol0/user/k.pakulev/Summertime/RF_Net/runs"
date="07_15_20_00"
model="e004_NN_0.332_NNT_0.447_NNDR_0.823_MeanMS_0.534.pth.tar"

save=""$root"/"$date""
resume=""$save"/model/"$model""

stdout=$save"/test_out.out"
stderr=$save"/test_out.err"

rm -f $stdout
rm -f $stderr

bsub -q normal -J k.pakulev -gpu "num=1:mode=exclusive_process" -o $stdout -e $stderr python ~/Summertime/RF_Net/ms.py --data=view --resume=$resume
bsub -q normal -J k.pakulev -gpu "num=1:mode=exclusive_process" -o $stdout -e $stderr python ~/Summertime/RF_Net/ms.py --data=illu --resume=$resume
bsub -q normal -J k.pakulev -gpu "num=1:mode=exclusive_process" -o $stdout -e $stderr python ~/Summertime/RF_Net/ms.py --data=ef --resume=$resume