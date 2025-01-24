# U-Net

参考：  
https://github.com/milesial/Pytorch-UNet  
https://github.com/Jack-Cherish/Deep-Learning/tree/master/Pytorch-Seg/lesson-2  
https://mp.weixin.qq.com/s/KI-9z7FBjfoWfZK3PEPXJA  
https://mp.weixin.qq.com/s/6tZVUbyEjLVewM8vGK9Zhw  
https://mp.weixin.qq.com/s/7FY77k3xtK-UyfoXpFXgBQ  
https://github.com/usuyama/pytorch-unet  
  
1、代码clone Pytorch-UNet项目：  
git clone https://github.com/milesial/Pytorch-UNet.git  
cd Pytorch-UNet  
  
2、数据集下载：  
bash scripts/download_data.sh  
Downloading train_hq.zip to /home/haoyu/work/code/DiveIntoAlgo/cv/U-Net/Pytorch-UNet  
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 804M/804M [00:44<00:00, 18.9MB/s]  
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 804M/804M [00:44<00:00, 19.0MB/s]  
Archive:  train_hq.zip  
   creating: train_hq/  
  inflating: train_hq/a1038b207299_14.jpg    
  inflating: train_hq/0d3adbbc9a8b_06.jpg    
... ...  
... ...  
Downloading train_masks.zip to /home/haoyu/work/code/DiveIntoAlgo/cv/U-Net/Pytorch-UNet  
 96%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊      | 28.0M/29.1M [00:03<00:00, 17.2MB/s]  
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 29.1M/29.1M [00:03<00:00, 8.52MB/s]  
Archive:  train_masks.zip  
   creating: train_masks/  
  inflating: train_masks/00087a6bd4dc_01_mask.gif    
 extracting: train_masks/00087a6bd4dc_02_mask.gif    
... ...  
... ...  
  
如果你的数据集下载遇到问题，也可以直接到Kaggle下载：  
https://www.kaggle.com/competitions/carvana-image-masking-challenge/data  
只要下载train_hq.zip和train_masks.zip，解压出来分别对应原始图片和mask图片，各有5088张，一一对应，分别放到data/imgs和data/masks目录下。  
  
3、训练  
python train.py  
  
4、推理  
python predict.py -i input.jpg -o output_mask.jpg -m checkpoints/checkpoint_epoch3.pth  
  
### License  
  
MIT
