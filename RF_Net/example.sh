#!/usr/bin/env bash

root="/Vol0/user/k.pakulev/Summertime/RF_Net/runs"
date="07_15_20_00"
model="e089_NN_0.418_NNT_0.636_NNDR_0.867_MeanMS_0.640.pth.tar"
img_path="/Vol0/user/k.pakulev/Summertime/RF_Net/data/hpatch_v_sequence/v_adam"

save=""$root"/"$date""
resume=""$save"/model/"$model""

mkdir ""$save"/material"

echo "Saved to "$save""

stdout=$save"/example_out.out"
stderr=$save"/example_out.err"

rm -f $stdout
rm -f $stderr

bsub -q normal -J k.pakulev -gpu "num=1:mode=exclusive_process" -o $stdout -e $stderr python ~/Summertime/RF_Net/example.py --save=$save --imgpath=$img_path --resume=$resume