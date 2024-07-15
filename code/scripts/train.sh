cfg=$1
batch_size=64

state_epoch=1 
pretrained_model_path='./saved_models/data/model_save_file'
log_dir='new'

multi_gpus=False
mixed_precision=True

nodes=1
num_workers=8
master_port=11266
stamp=gpu${nodes}MP_${mixed_precision}

python /content/GALIP/code/src/train.py \
    --cfg $cfg \
    --log_dir $log_dir \
    --batch_size $batch_size \
    --state_epoch $state_epoch \
    --num_workers $num_workers \
    --pretrained_model_path $pretrained_model_path \
