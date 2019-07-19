#!/usr/bin/env bash

root="/Vol0/user/k.pakulev/Summertime/ST_Net/runs"
date="07_19_19_30"
model="e021.pth.tar"
img_path="/Vol0/user/k.pakulev/Summertime/ST_Net/data/hpatch_v_sequence/v_adam"

save=""$root"/"$date""
resume=""$save"/model/"$model""

mkdir ""$save"/material"

echo "Saved to "$save""

stdout=$save"/example_out.out"
stderr=$save"/example_out.err"

rm -f $stdout
rm -f $stderr

bsub -q normal -J k.pakulev -gpu "num=1:mode=exclusive_process" -o $stdout -e $stderr python ~/Summertime/ST_Net/example.py --save=$save --imgpath=$img_path --resume=$resume