#!/usr/bin/env bash

save="/Vol0/user/k.pakulev/Summertime/legacy/RF_Net/runs/08_11_18_04"
resume="/Vol0/user/k.pakulev/Summertime/legacy/RF_Net/runs/08_11_18_04/model/e142_NN_0.453_NNT_0.651_NNDR_0.844_MeanMS_0.649.pth.tar"

stdout=$save"/test_out.out"
stderr=$save"/test_out.err"

rm -f $stdout
rm -f $stderr

bsub -q normal -J k.pakulev -gpu "num=1:mode=exclusive_process" -o $stdout -e $stderr python ~/Summertime/legacy/RF_Net/ms.py --data=view --resume=$resume
