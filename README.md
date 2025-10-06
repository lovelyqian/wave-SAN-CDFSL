# 1 Wave-SAN
Repository for the paper: Wave-SAN: Wavelet based Style Augmentation Network for Cross-Domain Few-Shot Learning


[[Paper](https://arxiv.org/abs/2203.07656)]

<img width="749" alt="image" src="https://github.com/lovelyqian/wave-SAN-CDFSL/assets/49612387/64d6af8b-885c-4deb-a619-e43377df8351">


# 2 Setup 
## 2.1 conda env & code
```
# conda env
conda create --name py36 python=3.6
conda activate py36
conda install pytorch torchvision -c pytorch
pip3 install scipy>=1.3.2
pip3 install tensorboardX>=1.4
pip3 install h5py>=2.9.0

# code
git clone https://github.com/lovelyqian/wave-SAN-CDFSL
cd wave-SAN-CDFSL
```

## 2.2 datasets
We use the mini-Imagenet as the single source dataset, and use cub, cars, places, plantae, ChestX, ISIC, EuroSAT, and CropDisease as novel target datasets. 

:(Note results of ChestX, ISIC, EuroSAT, and CropDisease are not reported in the paper. For your convenience, we keep these datasets in the code. 

For the mini-Imagenet, cub, cars, places, and plantae, we refer to the [FWT](https://github.com/hytseng0509/CrossDomainFewShot) repo.

For the ChestX, ISIC, EuroSAT, and CropDisease, we refer to the [BS-CDFSL](https://github.com/IBM/cdfsl-benchmark) repo.


# 3 Wave-SAN
## 3.1 pretrain 
recommend: use the [pretrained ckp](https://github.com/lovelyqian/wave-SAN-CDFSL/tree/main/output/checkpoints/baseline) by [FWT](https://github.com/hytseng0509/CrossDomainFewShot). 

Or, you can pretrain it: 
```
python3 network_train.py --dataset miniImagenet --stage pretrain --name your-exp-pretrain --train_aug --stop_epoch 400 --save_freq 100
```


## 3.2 meta-train
take 5-way 1-shot as an example: 

```
python3 network_train.py --dataset miniImagenet --stage metatrain --name your-exp-name --train_aug --warmup baseline --n_shot 1 --stop_epoch 200 --save_freq 100
```

the `--warmup` can be replaced by `your-exp-pretrain`

after the meta-train is done, the script will automatically perform the inference.  

our meta-trained ckps can be found in [wave-SAN 1shot](https://github.com/lovelyqian/wave-SAN-CDFSL/tree/main/output/checkpoints/GNN-waveSAN-1shot) and [wave-SAN 5shot](https://github.com/lovelyqian/wave-SAN-CDFSL/tree/main/output/checkpoints/GNN-waveSAN-5shot)

## 3.3 testing
if you wish to test on specific target dataset with specific model:

```
python3 test_function.py --dataset cub --name your-exp-name --n_shot 1
```

# 4 Citing
If you find our paper or this code useful for your research, please considering cite us (●°u°●)」:
```

@article{fu2022wave,
  title={Wave-SAN: Wavelet based Style Augmentation Network for Cross-Domain Few-Shot Learning},
  author={Fu, Yuqian and Xie, Yu and Fu, Yanwei and Chen, Jingjing and Jiang, Yu-Gang},
  journal={arXiv preprint},
  year={2022}
}
```

Also, we have published StyleAdv(CVPR23) which outperforms wave-SAN by generating both "virtual" and "hard" styles via adversarial style attack. [[Paper](https://arxiv.org/pdf/2302.09309)], [[Code](https://github.com/lovelyqian/StyleAdv-CDFSL)], [[Presentation Video on Bilibili](https://www.bilibili.com/video/BV1th4y1s78H/?spm_id_from=333.999.0.0&vd_source=668a0bb77d7d7b855bde68ecea1232e7)]

```
@inproceedings{fu2023styleadv,
  title={StyleAdv: Meta Style Adversarial Training for Cross-Domain Few-Shot Learning},
  author={Fu, Yuqian and Xie, Yu and Fu, Yanwei and Jiang, Yu-Gang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={24575--24584},
  year={2023}
}
```

We also have works [meta-FDMixup](https://github.com/lovelyqian/Meta-FDMixup),  [Me-D2N](https://github.com/lovelyqian/ME-D2N_for_CDFSL), [TGDM](https://arxiv.org/abs/2210.05392) which tackles CD-FSL with few labeled target examples. 


# 5 Acknowledge
Our code is built upon the implementation of [FWT](https://github.com/hytseng0509/CrossDomainFewShot) and [ATA](https://github.com/Haoqing-Wang/CDFSL-ATA). Thanks for their work.
