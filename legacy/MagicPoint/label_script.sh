# !/bin/sh
rm -f out_l.out
rm -f out_l.err
bsub -q normal -J k.pakulev -gpu "num=1" -o out.out -e out.err python ~/Summertime/MagicPoint/label_tum.py