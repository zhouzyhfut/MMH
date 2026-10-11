cd /CLIPtrans
export MMH_DATA_ROOT=/16T/leizhibo/Dataset/guangyuan_3mod
python train.py --mode Triplet --lr 0.001 --gpu 0
python train.py --mode Triplet --lr 0.001 --gpu 1
