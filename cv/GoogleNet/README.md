# GoogleNet

GoogleNet（也称为 Inception v1）是由Google团队在2014年提出的一种深度卷积神经网络架构，它在当年的ImageNet图像分类竞赛中取得了优异的成绩。GoogleNet的核心创新是提出了Inception模块，通过多尺度卷积和降维技术，显著减少了参数数量并提高了计算效率。  
  
GoogleNet的核心特点  
Inception 模块：  
&emsp;Inception模块通过并行使用不同尺寸的卷积核（如 1x1、3x3、5x5）和池化操作，捕捉多尺度的特征。  
&emsp;使用1x1卷积进行降维，减少计算量。  
全局平均池化：  
&emsp;使用全局平均池化（Global Average Pooling）代替全连接层，减少了参数数量。  
辅助分类器：  
&emsp;在网络中间层添加辅助分类器，帮助梯度传播并缓解梯度消失问题。  
深度和宽度：  
&emsp;GoogleNet是一个既深又宽的网络，共有22层（包括卷积层和全连接层）。  
  
GoogleNet的架构  
初始层：普通的卷积层和池化层。  
Inception模块：多个Inception模块堆叠而成。  
辅助分类器：两个辅助分类器用于中间监督。  
全局平均池化：代替全连接层，输出分类结果。  
   
```python
python GoogleNet.py 
Epoch [1/10], Train Loss: 1.4146, Train Acc: 47.88%, Val Loss: 1.1981, Val Acc: 57.27%
Epoch [2/10], Train Loss: 0.9528, Train Acc: 66.26%, Val Loss: 0.8868, Val Acc: 70.05%
Epoch [3/10], Train Loss: 0.7164, Train Acc: 75.09%, Val Loss: 0.6402, Val Acc: 77.62%
Epoch [4/10], Train Loss: 0.5751, Train Acc: 80.15%, Val Loss: 0.6644, Val Acc: 77.92%
Epoch [5/10], Train Loss: 0.4869, Train Acc: 83.07%, Val Loss: 0.6749, Val Acc: 76.05%
Epoch [6/10], Train Loss: 0.4091, Train Acc: 85.98%, Val Loss: 0.6723, Val Acc: 78.18%
Epoch [7/10], Train Loss: 0.3534, Train Acc: 87.87%, Val Loss: 0.5511, Val Acc: 82.19%
Epoch [8/10], Train Loss: 0.2933, Train Acc: 89.85%, Val Loss: 0.5029, Val Acc: 84.21%
Epoch [9/10], Train Loss: 0.2557, Train Acc: 91.08%, Val Loss: 0.5842, Val Acc: 82.20%
Epoch [10/10], Train Loss: 0.2129, Train Acc: 92.53%, Val Loss: 0.5722, Val Acc: 82.64%
```
  
### License  
  
MIT
