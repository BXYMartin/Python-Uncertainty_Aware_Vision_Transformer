# Uncertainty-Aware Vision Transformers for Medical Image Segmentation
The codes is publicly available on Github [Python-Uncertainty_Aware_Vision_Transformer](https://github.com/BXYMartin/Python-Uncertainty_Aware_Vision_Transformer). 

Code from the repository is implemented based on the original implementation of [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet). The training and testing framework is inherited. Main contribution of this work includes codes for the sampling, uncertainty-aware skip-connections module design, performance evaluation, uncertainty visualization, LIDC dataset definition & preprocessing, model structural changes, out-of-distribution samples creation & prediction and model computational complexity analysis.

## 1. Download pre-trained swin transformer model (Swin-T)
* [Get pre-trained model in this link] (https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY?usp=sharing): Put pretrained Swin-T into folder "pretrained_ckpt/"

## 2. Prepare data

- The Synapse datasets we used are provided by TransUnet's authors. Please go to ["./datasets/README.md"](datasets/README.md) for details, or please send an Email to jienengchen01 AT gmail.com to request the preprocessed data. If you would like to use the preprocessed data, please use it for research purposes and do not redistribute it (following the TransUnet's License).

- The LIDC dataset is acquired from the author of Hirarchical Probabilistic Unet in this link [Google Cloud Storage](https://console.cloud.google.com/storage/browser/hpunet-data/lidc_crops) or refer to their repo [Hirarchical Probabilistic Unet](https://github.com/deepmind/deepmind-research/tree/master/hierarchical_probabilistic_unet).

For both datasets, we provide the pre-computed index file for train/test/eval splitting.

## 3. Environment

- Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

## 4. Usage

- Run the train script on synapse dataset. The batch size we used is 24. If you do not have enough GPU memory, the bacth size can be reduced to 12 or 6 to save memory.

- Train

```bash
python train.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --root_path your DATA_DIR --max_epochs 150 --output_dir your OUT_DIR  --img_size 224 --base_lr 0.05 --batch_size 24
```

- Test 

```bash
python test.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path your DATA_DIR --output_dir your OUT_DIR --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24
```

- Sample
```bash
python sample.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path your DATA_DIR --output_dir your OUT_DIR --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24
```

- Out-of-distribution: run with random patches
```bash
python patch.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path your DATA_DIR --output_dir your OUT_DIR --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24
```


- Out-of-distribution: run with gaussian blurs
```bash
python ood.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path your DATA_DIR --output_dir your OUT_DIR --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24
```


- Out-of-distribution: run with real tumors
```bash
python tumor.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path your DATA_DIR --output_dir your OUT_DIR --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24
```

- Uncertainty visualization

```bash
python visualize.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path your DATA_DIR --output_dir your OUT_DIR --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24
```

- Computational complexity

```bash
python flops.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path your DATA_DIR --output_dir your OUT_DIR --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24
```

## References
* [Hierarchical Probabilistic Unet](https://github.com/deepmind/deepmind-research/tree/master/hierarchical_probabilistic_unet)
* [SwinUnet](https://github.com/HuCaoFighting/Swin-Unet)
* [TransUnet](https://github.com/Beckschen/TransUNet)
* [SwinTransformer](https://github.com/microsoft/Swin-Transformer)
