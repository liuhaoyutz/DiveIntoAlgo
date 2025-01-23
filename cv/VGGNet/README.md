# VGGNet

VGGNet（Visual Geometry Group Network）是由牛津大学的Visual Geometry Group提出的一种卷积神经网络架构，首次亮相于2014年的ILSVRC（ImageNet Large Scale Visual Recognition Challenge）。VGGNet的主要特点是使用了非常小且一致的卷积核（3x3），并通过堆叠多个这样的卷积层来构建深层结构。这种设计不仅简化了模型的配置，还使得它能够捕捉到图像中的复杂特征模式。  
  
VGGNet的特点  
简单统一的架构：所有卷积层都使用3x3的小卷积核，并且步长为1，填充为1。这确保了每次卷积后特征图的尺寸不变。  
最大池化层：每个卷积块后面跟着一个2x2的最大池化层，步长为2，用于减少空间维度。  
更深的网络：通过增加卷积层的数量，VGGNet可以达到更深的层次结构，如VGG16和VGG19分别有16和19层（包括全连接层）。  
全连接层：在卷积层之后，通常会有几个全连接层，最后是输出层。  
高准确率：尽管参数数量较多，但在ImageNet数据集上取得了很好的分类效果。  
  
VGG16结构：  
Input Layer: 224x224 RGB images.  
Conv Block 1: Two 3x3 conv layers (64 filters), followed by a 2x2 max pooling layer.  
Conv Block 2: Two 3x3 conv layers (128 filters), followed by a 2x2 max pooling layer.  
Conv Block 3: Three 3x3 conv layers (256 filters), followed by a 2x2 max pooling layer.  
Conv Block 4: Three 3x3 conv layers (512 filters), followed by a 2x2 max pooling layer.  
Conv Block 5: Three 3x3 conv layers (512 filters), followed by a 2x2 max pooling layer.  
Fully Connected Layers: Three fully connected layers, each with 4096 units, and the final output layer with 1000 units for ImageNet classification.  
  
VGG16-v1.py，正确率只有10%，因为CIFAR10是10分类，所以相当于什么都没学到。做如下改进：  
学习率调度器：optim.lr_scheduler.StepLR 在每 10 个 epoch 后将学习率乘以 0.1，有助于加速收敛并防止过拟合。  
正确的数据预处理：使用了适合CIFAR-10数据集的均值和标准差进行归一化，这通常能显著改善模型性能。  
简化模型：减少了卷积层中的通道数，使得模型更适合小规模数据集，同时降低了计算复杂度。  
增强正则化：增加了Dropout层的比例，并在优化器中加入了L2正则化（通过 weight_decay 参数），以防止过拟合。  
固定随机种子：设置了随机种子以保证实验结果的可重复性。  
监控训练过程：每个 epoch 结束后评估测试集上的准确率，并输出结果，帮助监控训练进展。  
  
VGG12-v2.py，正确率还是只有10%。可尝试如下检查：  
1、检查数据加载  
确保训练和测试数据正确加载且标签无误。通过可视化一些训练样本及其标签来验证这一点。  
```python
import matplotlib.pyplot as plt

def imshow(img):
    img = img / 2 + 0.5     # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# Show images
imshow(torchvision.utils.make_grid(images))
# Print labels
print(' '.join(f'{train_dataset.classes[labels[j]]:5s}' for j in range(batch_size)))
```
  
2、初始化权重  
确保模型的权重被正确初始化。虽然PyTorch默认会使用Kaiming初始化（适用于ReLU激活函数），但可以显式地进行初始化以确保一致性。  
```python
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

model.apply(init_weights)
```
  
3、检查损失函数  
确认使用的损失函数适合当前任务。对于多分类任务，CrossEntropyLoss是合适的选择。确保没有对输入进行额外的softmax操作，因为CrossEntropyLoss内部已经包含了log_softmax。  
  
4、检查模型是否正在学习  
为了确认模型是否正在学习，在训练开始前打印一次测试集上的准确率，并在每个epoch结束时再次打印。理想情况下，应该看到准确率逐步提高。  
  
5、梯度检查  
如果所有参数的梯度都为 0，会导致模型无法更新参数，从而无法学习。这种情况通常被称为梯度消失（Gradient Vanishing），是深度学习中常见的问题之一。  
```python
for name, param in model.named_parameters():
    if param.grad is None:
        print(f'No gradient for {name}')
    elif torch.all(param.grad == 0):
        print(f'All zero gradients for {name}')
```
  
6、调试模式  
将训练批次大小设置为非常小的值（如2或4），并只训练几个batch来快速观察模型的行为。这可以帮助更快地识别潜在的问题。  
  
7、使用预训练模型  
考虑使用一个已经在ImageNet上预训练过的VGG模型，并在此基础上进行微调（fine-tuning）。这样可以利用预训练模型已经学到的特征，从而可能更快地达到较好的性能。  
```python
from torchvision import models

# Load a pretrained VGG16 model and modify the classifier for CIFAR-10
vgg16_pretrained = models.vgg16(pretrained=True)
num_ftrs = vgg16_pretrained.classifier[6].in_features
vgg16_pretrained.classifier[6] = nn.Linear(num_ftrs, 10)

# Freeze all layers except the last one
for param in vgg16_pretrained.features.parameters():
    param.requires_grad = False

model = vgg16_pretrained.to(device)
```
  
8、减少正则化  
如果怀疑过度正则化可能是问题所在，尝试暂时移除或减少Dropout和L2正则化的强度，看看是否有所改善。  
  
9、增加训练时间  
有时候，模型需要更多的时间才能收敛到一个好的解。可以尝试增加训练的epoch数量，同时注意监控过拟合的迹象。  
  
VGG16-v3.py的正确率仍然为10%。  
  
梯度消失或爆炸  
问题：梯度消失或爆炸可能导致模型无法训练。  
怎样检查梯度消失或爆炸问题：打印模型参数的梯度值，如果梯度值为非常小（如1e-7或更小），则可能是梯度消失。如果梯度值非常大（如 1e+10 或更大），则可能是梯度爆炸。  
```python
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name} gradient norm: {torch.norm(param.grad)}")
```
VGG16-v4.py有如下打印信息，可以确认发生了梯度消失。  
```python
features.0.weight gradient norm: 0.0
features.0.bias gradient norm: 0.0
features.2.weight gradient norm: 0.0
features.2.bias gradient norm: 0.0
features.5.weight gradient norm: 0.0
features.5.bias gradient norm: 0.0
features.7.weight gradient norm: 0.0
features.7.bias gradient norm: 0.0
features.10.weight gradient norm: 0.0
features.10.bias gradient norm: 0.0
features.12.weight gradient norm: 0.0
features.12.bias gradient norm: 0.0
features.14.weight gradient norm: 0.0
features.14.bias gradient norm: 0.0
features.17.weight gradient norm: 0.0
features.17.bias gradient norm: 0.0
features.19.weight gradient norm: 0.0
features.19.bias gradient norm: 0.0
features.21.weight gradient norm: 0.0
features.21.bias gradient norm: 0.0
features.24.weight gradient norm: 0.0
features.24.bias gradient norm: 0.0
features.26.weight gradient norm: 0.0
features.26.bias gradient norm: 0.0
features.28.weight gradient norm: 0.0
features.28.bias gradient norm: 0.0
classifier.0.weight gradient norm: 0.0
classifier.0.bias gradient norm: 0.0041320486925542355
classifier.3.weight gradient norm: 0.001241662772372365
classifier.3.bias gradient norm: 0.08040982484817505
classifier.6.weight gradient norm: 0.0706406831741333
classifier.6.bias gradient norm: 0.18065650761127472
Epoch [1/10], Train Loss: 2.3035, Train Acc: 10.18%, Val Loss: 2.3028, Val Acc: 10.00%
```
  
解决方法：  
使用梯度裁剪（Gradient Clipping）解决梯度爆炸：  
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
添加批量归一化（Batch Normalization）层解决梯度消失：  
```python
nn.BatchNorm2d(num_features)
```
给前面的v1, v2, v3版本加上BatchNorm层，正确率能达到87%附近。    
  
  
梯度消失和爆炸的区别  
梯度消失：梯度值非常小（接近0）；损失值几乎不下降；参数值几乎不更新；可能原因包括网络过深，不适合的激活函数；解决方法包括使用ReLU，殘差连接，BatchNorm。  
梯度爆炸：梯度值非常大（NaN)；损失值变得非常大或NaN；参数值变得非常大或NaN；可能的原因包括学习率过高，权重初始化不当，网络过深。解决方法包括梯度裁减、降低学习率，权重初始化。  
  
```python
python VGG16-v3-BatchNorm.py 
Files already downloaded and verified
Initial Test Accuracy: 9.92%
Epoch [1/30], Step [100/782], Loss: 2.1977
Epoch [1/30], Step [200/782], Loss: 2.1060
Epoch [1/30], Step [300/782], Loss: 2.0838
Epoch [1/30], Step [400/782], Loss: 1.8230
Epoch [1/30], Step [500/782], Loss: 1.7246
Epoch [1/30], Step [600/782], Loss: 1.7195
Epoch [1/30], Step [700/782], Loss: 1.6867
Epoch [1/30], Test Accuracy: 37.26%
Epoch [2/30], Step [100/782], Loss: 1.6851
Epoch [2/30], Step [200/782], Loss: 1.4996
Epoch [2/30], Step [300/782], Loss: 1.4564
Epoch [2/30], Step [400/782], Loss: 1.5675
Epoch [2/30], Step [500/782], Loss: 1.6169
Epoch [2/30], Step [600/782], Loss: 1.3897
Epoch [2/30], Step [700/782], Loss: 1.4412
Epoch [2/30], Test Accuracy: 52.40%
Epoch [3/30], Step [100/782], Loss: 1.5973
Epoch [3/30], Step [200/782], Loss: 1.3951
Epoch [3/30], Step [300/782], Loss: 1.3967
Epoch [3/30], Step [400/782], Loss: 1.1926
Epoch [3/30], Step [500/782], Loss: 1.2492
Epoch [3/30], Step [600/782], Loss: 1.1684
Epoch [3/30], Step [700/782], Loss: 1.2648
Epoch [3/30], Test Accuracy: 57.41%
Epoch [4/30], Step [100/782], Loss: 1.0868
Epoch [4/30], Step [200/782], Loss: 1.0486
Epoch [4/30], Step [300/782], Loss: 0.9606
Epoch [4/30], Step [400/782], Loss: 1.0201
Epoch [4/30], Step [500/782], Loss: 1.1518
Epoch [4/30], Step [600/782], Loss: 1.0622
Epoch [4/30], Step [700/782], Loss: 0.9644
Epoch [4/30], Test Accuracy: 67.01%
Epoch [5/30], Step [100/782], Loss: 1.1401
Epoch [5/30], Step [200/782], Loss: 1.1586
Epoch [5/30], Step [300/782], Loss: 0.9179
Epoch [5/30], Step [400/782], Loss: 0.8312
Epoch [5/30], Step [500/782], Loss: 0.9715
Epoch [5/30], Step [600/782], Loss: 0.9917
Epoch [5/30], Step [700/782], Loss: 0.9151
Epoch [5/30], Test Accuracy: 63.99%
Epoch [6/30], Step [100/782], Loss: 0.8340
Epoch [6/30], Step [200/782], Loss: 1.0805
Epoch [6/30], Step [300/782], Loss: 0.8913
Epoch [6/30], Step [400/782], Loss: 1.0911
Epoch [6/30], Step [500/782], Loss: 0.7581
Epoch [6/30], Step [600/782], Loss: 0.9960
Epoch [6/30], Step [700/782], Loss: 0.9643
Epoch [6/30], Test Accuracy: 75.08%
Epoch [7/30], Step [100/782], Loss: 0.8228
Epoch [7/30], Step [200/782], Loss: 0.6252
Epoch [7/30], Step [300/782], Loss: 0.7553
Epoch [7/30], Step [400/782], Loss: 0.5359
Epoch [7/30], Step [500/782], Loss: 0.8588
Epoch [7/30], Step [600/782], Loss: 0.7501
Epoch [7/30], Step [700/782], Loss: 0.9714
Epoch [7/30], Test Accuracy: 74.99%
Epoch [8/30], Step [100/782], Loss: 0.5600
Epoch [8/30], Step [200/782], Loss: 0.8886
Epoch [8/30], Step [300/782], Loss: 0.5628
Epoch [8/30], Step [400/782], Loss: 0.8341
Epoch [8/30], Step [500/782], Loss: 0.7908
Epoch [8/30], Step [600/782], Loss: 1.0174
Epoch [8/30], Step [700/782], Loss: 0.7565
Epoch [8/30], Test Accuracy: 74.50%
Epoch [9/30], Step [100/782], Loss: 0.5155
Epoch [9/30], Step [200/782], Loss: 0.6299
Epoch [9/30], Step [300/782], Loss: 0.6544
Epoch [9/30], Step [400/782], Loss: 0.7326
Epoch [9/30], Step [500/782], Loss: 0.6231
Epoch [9/30], Step [600/782], Loss: 0.6398
Epoch [9/30], Step [700/782], Loss: 0.7711
Epoch [9/30], Test Accuracy: 78.46%
Epoch [10/30], Step [100/782], Loss: 0.9950
Epoch [10/30], Step [200/782], Loss: 0.6390
Epoch [10/30], Step [300/782], Loss: 0.6156
Epoch [10/30], Step [400/782], Loss: 0.7185
Epoch [10/30], Step [500/782], Loss: 0.6150
Epoch [10/30], Step [600/782], Loss: 0.4741
Epoch [10/30], Step [700/782], Loss: 0.6784
Epoch [10/30], Test Accuracy: 78.80%
Epoch [11/30], Step [100/782], Loss: 0.4733
Epoch [11/30], Step [200/782], Loss: 0.4956
Epoch [11/30], Step [300/782], Loss: 0.5810
Epoch [11/30], Step [400/782], Loss: 0.3979
Epoch [11/30], Step [500/782], Loss: 0.5999
Epoch [11/30], Step [600/782], Loss: 0.3147
Epoch [11/30], Step [700/782], Loss: 0.6404
Epoch [11/30], Test Accuracy: 84.53%
Epoch [12/30], Step [100/782], Loss: 0.3798
Epoch [12/30], Step [200/782], Loss: 0.5333
Epoch [12/30], Step [300/782], Loss: 0.5573
Epoch [12/30], Step [400/782], Loss: 0.4907
Epoch [12/30], Step [500/782], Loss: 0.4845
Epoch [12/30], Step [600/782], Loss: 0.3987
Epoch [12/30], Step [700/782], Loss: 0.5462
Epoch [12/30], Test Accuracy: 85.19%
Epoch [13/30], Step [100/782], Loss: 0.5307
Epoch [13/30], Step [200/782], Loss: 0.4790
Epoch [13/30], Step [300/782], Loss: 0.3920
Epoch [13/30], Step [400/782], Loss: 0.5628
Epoch [13/30], Step [500/782], Loss: 0.3839
Epoch [13/30], Step [600/782], Loss: 0.6262
Epoch [13/30], Step [700/782], Loss: 0.5002
Epoch [13/30], Test Accuracy: 85.69%
Epoch [14/30], Step [100/782], Loss: 0.5638
Epoch [14/30], Step [200/782], Loss: 0.3101
Epoch [14/30], Step [300/782], Loss: 0.3707
Epoch [14/30], Step [400/782], Loss: 0.3424
Epoch [14/30], Step [500/782], Loss: 0.4777
Epoch [14/30], Step [600/782], Loss: 0.3617
Epoch [14/30], Step [700/782], Loss: 0.2913
Epoch [14/30], Test Accuracy: 85.64%
Epoch [15/30], Step [100/782], Loss: 0.4613
Epoch [15/30], Step [200/782], Loss: 0.2807
Epoch [15/30], Step [300/782], Loss: 0.4562
Epoch [15/30], Step [400/782], Loss: 0.3625
Epoch [15/30], Step [500/782], Loss: 0.2704
Epoch [15/30], Step [600/782], Loss: 0.4174
Epoch [15/30], Step [700/782], Loss: 0.3537
Epoch [15/30], Test Accuracy: 86.00%
Epoch [16/30], Step [100/782], Loss: 0.3799
Epoch [16/30], Step [200/782], Loss: 0.4473
Epoch [16/30], Step [300/782], Loss: 0.3287
Epoch [16/30], Step [400/782], Loss: 0.4727
Epoch [16/30], Step [500/782], Loss: 0.3359
Epoch [16/30], Step [600/782], Loss: 0.4831
Epoch [16/30], Step [700/782], Loss: 0.3702
Epoch [16/30], Test Accuracy: 86.10%
Epoch [17/30], Step [100/782], Loss: 0.6049
Epoch [17/30], Step [200/782], Loss: 0.3980
Epoch [17/30], Step [300/782], Loss: 0.2941
Epoch [17/30], Step [400/782], Loss: 0.4337
Epoch [17/30], Step [500/782], Loss: 0.3871
Epoch [17/30], Step [600/782], Loss: 0.6392
Epoch [17/30], Step [700/782], Loss: 0.2934
Epoch [17/30], Test Accuracy: 86.42%
Epoch [18/30], Step [100/782], Loss: 0.3363
Epoch [18/30], Step [200/782], Loss: 0.2612
Epoch [18/30], Step [300/782], Loss: 0.2907
Epoch [18/30], Step [400/782], Loss: 0.5233
Epoch [18/30], Step [500/782], Loss: 0.3667
Epoch [18/30], Step [600/782], Loss: 0.4853
Epoch [18/30], Step [700/782], Loss: 0.6439
Epoch [18/30], Test Accuracy: 86.75%
Epoch [19/30], Step [100/782], Loss: 0.5749
Epoch [19/30], Step [200/782], Loss: 0.2861
Epoch [19/30], Step [300/782], Loss: 0.2706
Epoch [19/30], Step [400/782], Loss: 0.3475
Epoch [19/30], Step [500/782], Loss: 0.3737
Epoch [19/30], Step [600/782], Loss: 0.3910
Epoch [19/30], Step [700/782], Loss: 0.3155
Epoch [19/30], Test Accuracy: 86.74%
Epoch [20/30], Step [100/782], Loss: 0.5933
Epoch [20/30], Step [200/782], Loss: 0.2406
Epoch [20/30], Step [300/782], Loss: 0.3280
Epoch [20/30], Step [400/782], Loss: 0.5722
Epoch [20/30], Step [500/782], Loss: 0.4346
Epoch [20/30], Step [600/782], Loss: 0.3783
Epoch [20/30], Step [700/782], Loss: 0.2588
Epoch [20/30], Test Accuracy: 86.85%
Epoch [21/30], Step [100/782], Loss: 0.2736
Epoch [21/30], Step [200/782], Loss: 0.3290
Epoch [21/30], Step [300/782], Loss: 0.2888
Epoch [21/30], Step [400/782], Loss: 0.3452
Epoch [21/30], Step [500/782], Loss: 0.2570
Epoch [21/30], Step [600/782], Loss: 0.1799
Epoch [21/30], Step [700/782], Loss: 0.1383
Epoch [21/30], Test Accuracy: 87.17%
Epoch [22/30], Step [100/782], Loss: 0.4101
Epoch [22/30], Step [200/782], Loss: 0.4502
Epoch [22/30], Step [300/782], Loss: 0.4552
Epoch [22/30], Step [400/782], Loss: 0.1731
Epoch [22/30], Step [500/782], Loss: 0.3058
Epoch [22/30], Step [600/782], Loss: 0.2139
Epoch [22/30], Step [700/782], Loss: 0.2705
Epoch [22/30], Test Accuracy: 87.20%
Epoch [23/30], Step [100/782], Loss: 0.3097
Epoch [23/30], Step [200/782], Loss: 0.2099
Epoch [23/30], Step [300/782], Loss: 0.2964
Epoch [23/30], Step [400/782], Loss: 0.3397
Epoch [23/30], Step [500/782], Loss: 0.4152
Epoch [23/30], Step [600/782], Loss: 0.2532
Epoch [23/30], Step [700/782], Loss: 0.2364
Epoch [23/30], Test Accuracy: 87.48%
Epoch [24/30], Step [100/782], Loss: 0.4809
Epoch [24/30], Step [200/782], Loss: 0.3236
Epoch [24/30], Step [300/782], Loss: 0.4044
Epoch [24/30], Step [400/782], Loss: 0.1287
Epoch [24/30], Step [500/782], Loss: 0.3298
Epoch [24/30], Step [600/782], Loss: 0.3248
Epoch [24/30], Step [700/782], Loss: 0.3475
Epoch [24/30], Test Accuracy: 87.35%
Epoch [25/30], Step [100/782], Loss: 0.2573
Epoch [25/30], Step [200/782], Loss: 0.2602
Epoch [25/30], Step [300/782], Loss: 0.3215
Epoch [25/30], Step [400/782], Loss: 0.2824
Epoch [25/30], Step [500/782], Loss: 0.4428
Epoch [25/30], Step [600/782], Loss: 0.1914
Epoch [25/30], Step [700/782], Loss: 0.4130
Epoch [25/30], Test Accuracy: 87.36%
Epoch [26/30], Step [100/782], Loss: 0.4236
Epoch [26/30], Step [200/782], Loss: 0.3034
Epoch [26/30], Step [300/782], Loss: 0.2851
Epoch [26/30], Step [400/782], Loss: 0.3226
Epoch [26/30], Step [500/782], Loss: 0.6354
Epoch [26/30], Step [600/782], Loss: 0.2743
Epoch [26/30], Step [700/782], Loss: 0.1565
Epoch [26/30], Test Accuracy: 87.46%
Epoch [27/30], Step [100/782], Loss: 0.3124
Epoch [27/30], Step [200/782], Loss: 0.3879
Epoch [27/30], Step [300/782], Loss: 0.3715
Epoch [27/30], Step [400/782], Loss: 0.2114
Epoch [27/30], Step [500/782], Loss: 0.1574
Epoch [27/30], Step [600/782], Loss: 0.5745
Epoch [27/30], Step [700/782], Loss: 0.2518
Epoch [27/30], Test Accuracy: 87.39%
Epoch [28/30], Step [100/782], Loss: 0.2253
Epoch [28/30], Step [200/782], Loss: 0.2596
Epoch [28/30], Step [300/782], Loss: 0.3870
Epoch [28/30], Step [400/782], Loss: 0.2430
Epoch [28/30], Step [500/782], Loss: 0.3230
Epoch [28/30], Step [600/782], Loss: 0.6046
Epoch [28/30], Step [700/782], Loss: 0.3574
Epoch [28/30], Test Accuracy: 87.24%
Epoch [29/30], Step [100/782], Loss: 0.3675
Epoch [29/30], Step [200/782], Loss: 0.4251
Epoch [29/30], Step [300/782], Loss: 0.4544
Epoch [29/30], Step [400/782], Loss: 0.3488
Epoch [29/30], Step [500/782], Loss: 0.3157
Epoch [29/30], Step [600/782], Loss: 0.2399
Epoch [29/30], Step [700/782], Loss: 0.3056
Epoch [29/30], Test Accuracy: 87.62%
Epoch [30/30], Step [100/782], Loss: 0.1995
Epoch [30/30], Step [200/782], Loss: 0.2439
Epoch [30/30], Step [300/782], Loss: 0.3172
Epoch [30/30], Step [400/782], Loss: 0.3713
Epoch [30/30], Step [500/782], Loss: 0.2855
Epoch [30/30], Step [600/782], Loss: 0.3071
Epoch [30/30], Step [700/782], Loss: 0.2135
Epoch [30/30], Test Accuracy: 87.62%
```
  
### License  
  
MIT
