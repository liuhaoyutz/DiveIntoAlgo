# GoogleNet

ResNet（Residual Network）是由微软研究院的Kaiming He等人在2015年提出的一种深度卷积神经网络架构。ResNet的核心创新是引入了残差学习（Residual Learning），通过 跳跃连接（Skip Connection）解决了深层网络中的梯度消失和退化问题，使得训练非常深的网络（如 100 层以上）成为可能。ResNet在ImageNet竞赛中取得了突破性成绩，并成为深度学习领域的经典模型之一。  
  
ResNet的核心思想  
残差学习：  
传统的神经网络直接学习目标映射H(x)，而ResNet学习的是残差映射F(x)=H(x)−x。  
通过跳跃连接，将输入x直接加到输出上，即H(x)=F(x)+x。  
  
跳跃连接：  
跳跃连接将输入直接传递到后面的层，避免了梯度消失问题，同时使得网络更容易优化。  
  
深度扩展：  
ResNet可以扩展到非常深的层次（如 ResNet-50、ResNet-101、ResNet-152），而不会出现性能退化。  
  
ResNet的架构  
ResNet的基本构建块是残差块（Residual Block），分为两种：  
普通残差块：用于较浅的ResNet（如 ResNet-18、ResNet-34）。  
瓶颈残差块：用于较深的 ResNet（如 ResNet-50、ResNet-101、ResNet-152），通过1x1卷积减少计算量。  
  
ResNet的不同版本（如 ResNet-18、ResNet-34、ResNet-50 等）是根据网络的深度来命名的。  
ResNet-18的"18"表示该网络的总层数为18层。具体包括：  
1个初始卷积层：7x7 卷积，步幅为 2，输出通道数为 64。  
1个最大池化层：3x3 池化，步幅为 2。  
4个残差块组：每个残差块组包含2个残差块，每个残差块包含2个卷积层。因此4个殘差块组共有16层。  
1个全局平均池化层：将特征图的空间维度降为 1x1。  
1个全连接层：用于分类任务。  
  
```python
python ResNet-18.py 
Epoch [1/10], Train Loss: 1.4719, Train Acc: 45.73%, Val Loss: 1.4131, Val Acc: 50.99%
Epoch [2/10], Train Loss: 0.8858, Train Acc: 68.82%, Val Loss: 0.9514, Val Acc: 66.29%
Epoch [3/10], Train Loss: 0.6420, Train Acc: 77.51%, Val Loss: 0.6415, Val Acc: 77.84%
Epoch [4/10], Train Loss: 0.5065, Train Acc: 82.55%, Val Loss: 0.6006, Val Acc: 80.32%
Epoch [5/10], Train Loss: 0.4124, Train Acc: 85.46%, Val Loss: 0.6374, Val Acc: 78.38%
Epoch [6/10], Train Loss: 0.3255, Train Acc: 88.75%, Val Loss: 0.6080, Val Acc: 80.84%
Epoch [7/10], Train Loss: 0.2459, Train Acc: 91.36%, Val Loss: 0.5141, Val Acc: 83.35%
Epoch [8/10], Train Loss: 0.1826, Train Acc: 93.54%, Val Loss: 0.6727, Val Acc: 80.97%
Epoch [9/10], Train Loss: 0.1254, Train Acc: 95.54%, Val Loss: 0.5908, Val Acc: 83.96%
Epoch [10/10], Train Loss: 0.0966, Train Acc: 96.54%, Val Loss: 0.6928, Val Acc: 82.30%
```
  
### License  
  
MIT
