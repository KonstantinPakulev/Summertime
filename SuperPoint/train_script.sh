rm -f out.out
rm -f out.err
bsub -q normal -J k.pakulev -gpu "num=1:mode=exclusive_process" -o out.out -e out.err python ~/Summertime/SuperPoint/train.py