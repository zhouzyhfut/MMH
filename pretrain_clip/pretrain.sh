cd /pretrain_clip
export MMH_PRETRAIN_DATA_ROOT=/16T/leizhibo/Dataset/guangyuan_3mod
python pretrain.py --lr 0.001 --dataset regdb --gpu 0