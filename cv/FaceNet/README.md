# FaceNet

参考：  
https://github.com/bubbliiiing/facenet-pytorch  
https://blog.csdn.net/weixin_44791964/article/details/108220265  
  
1、源码下载  
git clone https://github.com/bubbliiiing/facenet-pytorch.git  
  
2、数据集下载  
训练用的CASIA-WebFaces数据集以及评估用的LFW数据集可以在百度网盘下载。  
链接: https://pan.baidu.com/s/1qMxFR8H_ih0xmY-rKgRejw  
提取码: bcrq  
下载下来的datasets.zip即CASIA-WebFaces数据集，lfw.zip即LFW数据集。  
将这2个压缩文件放到facenet-pytorch项目根目录下，解压后会把数据集放到datasets和lfw目录下。  
  
3、使用预训练权重预测  
python predict.py  
  
4、训练  
python train.py  
  
开始训练前，执行python txt_annotation.py命令，生成cls_train.txt，并打上如下patch：  
```python
diff --git a/train.py b/train.py
index 54e525f..2b0d077 100644
--- a/train.py
+++ b/train.py
@@ -62,7 +62,7 @@ if __name__ == "__main__":
     #   mobilenet
     #   inception_resnetv1
     #--------------------------------------------------------#
-    backbone        = "mobilenet"
+    backbone        = "inception_resnetv1"
     #----------------------------------------------------------------------------------------------------------------------------#
     #   权值文件的下载请看README，可以通过网盘下载。
     #   模型的 预训练权重 比较重要的部分是 主干特征提取网络的权值部分，用于进行特征提取。
@@ -76,7 +76,7 @@ if __name__ == "__main__":
     #   如果想要让模型从主干的预训练权值开始训练，则设置model_path = ''，pretrain = True，此时仅加载主干。
     #   如果想要让模型从0开始训练，则设置model_path = ''，pretrain = Fasle，此时从0开始训练。
     #----------------------------------------------------------------------------------------------------------------------------#  
-    model_path      = "model_data/facenet_mobilenet.pth"
+    model_path      = ""
     #----------------------------------------------------------------------------------------------------------------------------#
     #   是否使用主干网络的预训练权重，此处使用的是主干的权重，因此是在模型构建的时候进行加载的。
     #   如果设置了model_path，则主干的权值无需加载，pretrained的值无意义。
diff --git a/utils/dataloader.py b/utils/dataloader.py
index 471102f..a62f03e 100644
--- a/utils/dataloader.py
+++ b/utils/dataloader.py
@@ -161,7 +161,7 @@ class LFWDataset(datasets.ImageFolder):
             for line in f.readlines()[1:]:
                 pair = line.strip().split()
                 pairs.append(pair)
-        return np.array(pairs)
+        return np.array(pairs, dtype=object)^M
 
     def get_lfw_paths(self,lfw_dir,file_ext="jpg"):
 
diff --git a/utils/utils_metrics.py b/utils/utils_metrics.py
index 110fc2b..94794e4 100644
--- a/utils/utils_metrics.py
+++ b/utils/utils_metrics.py
@@ -69,7 +69,14 @@ def calculate_val(thresholds, distances, labels, far_target=1e-3, nrof_folds=10)
         for threshold_idx, threshold in enumerate(thresholds):
             _, far_train[threshold_idx] = calculate_val_far(threshold, distances[train_set], labels[train_set])
         if np.max(far_train)>=far_target:
-            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
+            # f = interpolate.interp1d(far_train, thresholds, kind='slinear')^M
+^M
+            # 去除 far_train 中的重复值^M
+            unique_far_train, unique_indices = np.unique(far_train, return_index=True)^M
+            unique_thresholds = thresholds[unique_indices]^M
+            # 使用去重后的数据进行插值^M
+            f = interpolate.interp1d(unique_far_train, unique_thresholds, kind='slinear')^M
+^M
             threshold = f(far_target)
         else:
             threshold = 0.0
```
  
### License  
  
MIT
