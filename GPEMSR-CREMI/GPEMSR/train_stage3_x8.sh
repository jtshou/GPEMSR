python -m torch.distributed.launch --nproc_per_node=2 --master_port=4432 train_stage3.py -opt option/train_stage3_x8.yml