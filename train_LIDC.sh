EPOCH_TIME = 150
OUT_DIR  = './model_out'
CFG = 'configs/swin_tiny_patch4_window7_224_lite.yaml'
DATA_DIR = 'data/LIDC'
LEARNING_RATE = 0.01
IMG_SIZE = 224
BATCH_SIZE = 24
echo "start train model"
python train.py --dataset LIDC --cfg 'configs/swin_tiny_patch4_window7_224_lite.yaml' --root_path 'data/LIDC' --max_epochs 150 --output_dir './model_out' --img_size 224 --base_lr 0.01 --batch_size 24
