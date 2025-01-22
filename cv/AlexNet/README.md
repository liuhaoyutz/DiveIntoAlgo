# AlexNet

AlexNet是由Alex Krizhevsky、Ilya Sutskever和Geoffrey E. Hinton在2012年提出的卷积神经网络，它在ImageNet大规模视觉识别挑战赛（ILSVRC）中取得了显著的成绩，极大地推动了深度学习的发展。AlexNet的成功在于其引入了几项重要的创新和技术改进，这些使得它比当时的其他模型更加有效。  
  
AlexNet的特点  
更深的架构：相比于之前的CNN，AlexNet拥有更多的层，包括五个卷积层和三个全连接层。  
ReLU激活函数：首次大规模使用ReLU（Rectified Linear Unit）作为激活函数，这加速了训练过程，并且缓解了梯度消失问题。  
数据增强：通过图像翻转和平移等技术来增加训练样本的数量和多样性。  
Dropout：用于防止过拟合的技术，在训练过程中随机丢弃一部分神经元。  
多GPU训练：利用两个NVIDIA GTX 580 GPUs来并行处理不同的卷积层，以加快计算速度。  
大数据集：在ImageNet数据集上进行了训练，该数据集包含超过1百万张标记图片，涵盖了1000个类别。  
  
AlexNet的结构  
输入层：227x227像素的RGB图像。  
卷积层C1：96个11x11的过滤器，步长为4，无填充。    
最大池化层P1：3x3窗口，步长为2。  
卷积层C2：256个5x5的过滤器，步长为1，填充为2。  
最大池化层P2：3x3窗口，步长为2。  
卷积层C3：384个3x3的过滤器，步长为1，填充为1。  
卷积层C4：384个3x3的过滤器，步长为1，填充为1。  
卷积层C5：256个3x3的过滤器，步长为1，填充为1。  
最大池化层 P3：3x3窗口，步长为2。  
全连接层FC6：4096个神经元。  
Dropout层D1：丢弃概率为0.5。  
全连接层FC7：4096个神经元。  
Dropout层D2：丢弃概率为0.5。  
输出层FC8：1000个神经元（对应于ImageNet的1000类别）。  
  
关键点说明  
数据集：这里我们使用的是CIFAR-10数据集而不是ImageNet，因为后者的数据量非常大。CIFAR-10包含10个类别的彩色图像，每个图像是32x32像素。  
模型调整：为了适应CIFAR-10的小尺寸图像，我们调整了AlexNet的某些参数，比如卷积层中的通道数和全连接层的大小。对于ImageNet数据集，你应该恢复到原始的AlexNet配置。  
训练技巧：采用了数据增强（如随机水平翻转和平移裁剪）以及标准化处理来提高泛化能力。同时使用了Adam优化器和Dropout技术来加速收敛并防止过拟合。  

```python
Epoch [1/10], Step [100/782], Loss: 1.9036
Epoch [1/10], Step [200/782], Loss: 1.6331
Epoch [1/10], Step [300/782], Loss: 1.7845
Epoch [1/10], Step [400/782], Loss: 1.6011
Epoch [1/10], Step [500/782], Loss: 1.6529
Epoch [1/10], Step [600/782], Loss: 1.6390
Epoch [1/10], Step [700/782], Loss: 1.3466
Epoch [2/10], Step [100/782], Loss: 1.5495
Epoch [2/10], Step [200/782], Loss: 1.8218
Epoch [2/10], Step [300/782], Loss: 1.4356
Epoch [2/10], Step [400/782], Loss: 1.7253
Epoch [2/10], Step [500/782], Loss: 1.5406
Epoch [2/10], Step [600/782], Loss: 1.5129
Epoch [2/10], Step [700/782], Loss: 1.4190
Epoch [3/10], Step [100/782], Loss: 1.3684
Epoch [3/10], Step [200/782], Loss: 1.4501
Epoch [3/10], Step [300/782], Loss: 1.3071
Epoch [3/10], Step [400/782], Loss: 1.4815
Epoch [3/10], Step [500/782], Loss: 1.2729
Epoch [3/10], Step [600/782], Loss: 1.4897
Epoch [3/10], Step [700/782], Loss: 1.5567
Epoch [4/10], Step [100/782], Loss: 1.5215
Epoch [4/10], Step [200/782], Loss: 1.2774
Epoch [4/10], Step [300/782], Loss: 1.1204
Epoch [4/10], Step [400/782], Loss: 1.3671
Epoch [4/10], Step [500/782], Loss: 1.2078
Epoch [4/10], Step [600/782], Loss: 1.4011
Epoch [4/10], Step [700/782], Loss: 1.1908
Epoch [5/10], Step [100/782], Loss: 1.1191
Epoch [5/10], Step [200/782], Loss: 1.1307
Epoch [5/10], Step [300/782], Loss: 1.0523
Epoch [5/10], Step [400/782], Loss: 1.3328
Epoch [5/10], Step [500/782], Loss: 1.1945
Epoch [5/10], Step [600/782], Loss: 1.3299
Epoch [5/10], Step [700/782], Loss: 1.0201
Epoch [6/10], Step [100/782], Loss: 1.0904
Epoch [6/10], Step [200/782], Loss: 1.1063
Epoch [6/10], Step [300/782], Loss: 0.9578
Epoch [6/10], Step [400/782], Loss: 1.0598
Epoch [6/10], Step [500/782], Loss: 1.2457
Epoch [6/10], Step [600/782], Loss: 0.9023
Epoch [6/10], Step [700/782], Loss: 0.9005
Epoch [7/10], Step [100/782], Loss: 1.0185
Epoch [7/10], Step [200/782], Loss: 0.9327
Epoch [7/10], Step [300/782], Loss: 1.2201
Epoch [7/10], Step [400/782], Loss: 1.0597
Epoch [7/10], Step [500/782], Loss: 1.1373
Epoch [7/10], Step [600/782], Loss: 1.0691
Epoch [7/10], Step [700/782], Loss: 0.9217
Epoch [8/10], Step [100/782], Loss: 0.9362
Epoch [8/10], Step [200/782], Loss: 1.2076
Epoch [8/10], Step [300/782], Loss: 0.8203
Epoch [8/10], Step [400/782], Loss: 1.0272
Epoch [8/10], Step [500/782], Loss: 0.9555
Epoch [8/10], Step [600/782], Loss: 1.1203
Epoch [8/10], Step [700/782], Loss: 1.0967
Epoch [9/10], Step [100/782], Loss: 1.0204
Epoch [9/10], Step [200/782], Loss: 1.1209
Epoch [9/10], Step [300/782], Loss: 1.0944
Epoch [9/10], Step [400/782], Loss: 0.8406
Epoch [9/10], Step [500/782], Loss: 0.9575
Epoch [9/10], Step [600/782], Loss: 0.9944
Epoch [9/10], Step [700/782], Loss: 0.7835
Epoch [10/10], Step [100/782], Loss: 1.0724
Epoch [10/10], Step [200/782], Loss: 0.7809
Epoch [10/10], Step [300/782], Loss: 1.0928
Epoch [10/10], Step [400/782], Loss: 0.7482
Epoch [10/10], Step [500/782], Loss: 1.0029
Epoch [10/10], Step [600/782], Loss: 0.7931
Epoch [10/10], Step [700/782], Loss: 1.1611

```
### License  
  
MIT
