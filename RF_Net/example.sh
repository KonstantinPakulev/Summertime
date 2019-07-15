#!/usr/bin/env bash

root="/Vol0/user/k.pakulev/Summertime/RF_Net/runs"
date="07_15_20_00"
model="e004_NN_0.332_NNT_0.447_NNDR_0.823_MeanMS_0.534.pth.tar"
img_path="/Vol0/user/k.pakulev/Summertime/RF_Net/material/img2.png@/Vol0/user/k.pakulev/Summertime/RF_Net/material/img3.png"

save=""$root"/"$date""
resume=""$save"/model/"$model""

stdout=$save"/example_out.out"
stderr=$save"/example_out.err"

rm -f $stdout
rm -f $stderr

bsub -q normal -J k.pakulev -gpu "num=1:mode=exclusive_process" -o $stdout -e $stderr python ~/Summertime/RF_Net/example.py --imgpath=$img_path --resume=$resume