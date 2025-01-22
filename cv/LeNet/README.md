# LeNet

LeNet-5是由Yann LeCun等人在1998年发表的一篇论文"Gradient-Based Learning Applied to Document Recognition"中提出的一个卷积神经网络CNN）。它是早期成功的卷积神经网络之一，最初是为了解决手写字符识别问题而设计的。
LeNet-5中的5表示该模型是Yann LeCun及其同事开发的一系列卷积神经网络中的第五个版本。

输入层
输入图像：LeNet-5的原始设计是针对32x32像素的灰度图像，例如来自USPS数据集的手写数字图片。
LeNet-5的输入图像尺寸被设定为32x32像素的原因主要与模型架构的设计和当时的数据集有关。
1、LeNet-5最初是针对USPSUnited States Postal Service）数据集设计的，该数据集中的图像尺寸恰好是16x16或者经过处理后变为32x32。
2、LeNet-5包含多个卷积层和池化层，这些层会逐步减少特征图的空间维度。以32x32的输入为例，经过一系列的卷积操作和2x2的平均池化之后，仍然能够保持足够的空间分辨率来提取有意义的特征。如果输入尺寸过小，比如直接使用原始的28x28（如MNIST），那么在网络的深层可能会导致特征图变得太小，从而损失太多信息。

卷积层 (Layer C1)
类型：卷积层
过滤器数量：6
过滤器尺寸：5x5
激活函数：Tanh
输出特征图尺寸：28x28（因为没有填充，且步长为 1）

下采样层 (Layer S2)
类型：平均池化层（也称为下采样层）
池化窗口尺寸：2x2
步长：2
激活函数：Tanh
输出特征图尺寸：14x14

卷积层 (Layer C3)
类型：卷积层
过滤器数量：16
过滤器尺寸：5x5
激活函数：Tanh
输出特征图尺寸：10x10（同样无填充，步长为 1）

下采样层 (Layer S4)
类型：平均池化层
池化窗口尺寸：2x2
步长：2
激活函数：Tanh
输出特征图尺寸：5x5

卷积层 (Layer C5)
类型：卷积层
过滤器数量：120
过滤器尺寸：5x5
激活函数：Tanh
输出特征图尺寸：1x1（全连接层的替代品）

全连接层 (Layer F6)
类型：全连接层
神经元数量：84
激活函数：Tanh

输出层
类型：全连接层
神经元数量：根据分类任务确定（对于MNIST数据集来说是10个类别）
激活函数：一般不使用非线性激活函数，直接输出用于分类的概率分布（通常结合softmax函数一起使用）。

```python
Epoch [1/5], Step [100/938], Loss: 0.3957
Epoch [1/5], Step [200/938], Loss: 0.1374
Epoch [1/5], Step [300/938], Loss: 0.2662
Epoch [1/5], Step [400/938], Loss: 0.0838
Epoch [1/5], Step [500/938], Loss: 0.1088
Epoch [1/5], Step [600/938], Loss: 0.0528
Epoch [1/5], Step [700/938], Loss: 0.1292
Epoch [1/5], Step [800/938], Loss: 0.1348
Epoch [1/5], Step [900/938], Loss: 0.0639
Epoch [2/5], Step [100/938], Loss: 0.0538
Epoch [2/5], Step [200/938], Loss: 0.1276
Epoch [2/5], Step [300/938], Loss: 0.0240
Epoch [2/5], Step [400/938], Loss: 0.0326
Epoch [2/5], Step [500/938], Loss: 0.0307
Epoch [2/5], Step [600/938], Loss: 0.1131
Epoch [2/5], Step [700/938], Loss: 0.0971
Epoch [2/5], Step [800/938], Loss: 0.1571
Epoch [2/5], Step [900/938], Loss: 0.0444
Epoch [3/5], Step [100/938], Loss: 0.0302
Epoch [3/5], Step [200/938], Loss: 0.0465
Epoch [3/5], Step [300/938], Loss: 0.0864
Epoch [3/5], Step [400/938], Loss: 0.0289
Epoch [3/5], Step [500/938], Loss: 0.0171
Epoch [3/5], Step [600/938], Loss: 0.0529
Epoch [3/5], Step [700/938], Loss: 0.0969
Epoch [3/5], Step [800/938], Loss: 0.0560
Epoch [3/5], Step [900/938], Loss: 0.0704
Epoch [4/5], Step [100/938], Loss: 0.0219
Epoch [4/5], Step [200/938], Loss: 0.0101
Epoch [4/5], Step [300/938], Loss: 0.0126
Epoch [4/5], Step [400/938], Loss: 0.1124
Epoch [4/5], Step [500/938], Loss: 0.0454
Epoch [4/5], Step [600/938], Loss: 0.0257
Epoch [4/5], Step [700/938], Loss: 0.0039
Epoch [4/5], Step [800/938], Loss: 0.0092
Epoch [4/5], Step [900/938], Loss: 0.0565
Epoch [5/5], Step [100/938], Loss: 0.0083
Epoch [5/5], Step [200/938], Loss: 0.0119
Epoch [5/5], Step [300/938], Loss: 0.0340
Epoch [5/5], Step [400/938], Loss: 0.0036
Epoch [5/5], Step [500/938], Loss: 0.0079
Epoch [5/5], Step [600/938], Loss: 0.0054
Epoch [5/5], Step [700/938], Loss: 0.0009
Epoch [5/5], Step [800/938], Loss: 0.0108
Epoch [5/5], Step [900/938], Loss: 0.0363
Test Accuracy of the model on the 10000 test images: 98.34%
```
### License  
  
MIT
