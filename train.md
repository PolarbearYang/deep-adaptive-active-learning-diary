# 训练配置

以resnet18+cifar10为例，默认`cifar10.sh`的内容如下：

```shell
CUDA_VISIBLE_DEVICES=6 python main.py --save-freq 100 \
    --model resnet18_cifar \
    --dataset cifar10 \
    --strategy EntropySampling \
    --num-init-labels 1000 \
    --rounds 20 \
    --num-query 1000 \
    --updating \
    --n-epoch 200 \
    --batch-size 64 \
    --lr 0.001 \
    --momentum 0.9 \
    --plot-mode tsne \
    --plot-using-predefined \
    --plot-preload-path pretrained/resnet18_cifar_cifar10_tsne_embedded.npy
```

其中的必要参数为：

- `--model`: 使用的模型名称，名称必须全部小写。目前可用的模型及其使用场景将在下面列出。
- `--dataset`: 使用的数据集名称，名称必须全部小写。目前可用的是`cifar10`,`cifar100`,`mnist`,`fashionmnist`,`svhn`,`mixedgaussian`。
  - `mixedgaussian`是由三个中心不同，方差相同的分布生成的二维点集，用于做3-分类。
- `--strategy`: 使用的策略名称。目前可用的主动学习策略名称将在下面列出。
- `--num-init-labels`: 初始样本标注量。
- `--rounds`: 总共进行多少active rounds。
- `--num-query`: 每个active round查询的样本数量。
- `--updating`: 每个active round是使用update方式还是retraining方式。有此选项则载入上一个轮次的参数进行训练；否则重新开始训练。
- `--n-epoch`: 每个active round训练多少个epoch。
- `--batch-size`: batch size。
- `--lr`: 初始学习率。
- `--momentum`: optimizer中的momentum参数。

可选参数为：

- `--work-dir`: 指定工作路径。该路径存储所有运行过程中产生的信息，包括log、参数、分析图像等。
  - 若未事先指定，文件夹名称为{模型名}+{数据集名}+{运行开始时间戳}+{ID}
  - 若事先指定且文件夹不存在/文件夹下没有resume信息，则从该文件夹下从头开始训练
  - 若事先指定且文件夹下有恢复信息，则读取该文件夹中的训练信息并继续训练（用于程序意外中断的情况）
- `--plot-mode`: 使用哪种方式进行降维聚类可视化。可使用`tsne`, `umap`, `raw`。
  - `raw`不进行任何降维手段，只能用于二维点构成的数据集。
- `--plot-using-predefined`: 若使用此参数，则我们使用预定义的降维坐标可视化数据集。必须和`--plot-preload-path`载入的坐标搭配使用。
- `--plot-preload-path`: 事先制定的数据点降维后的二维坐标路径。必须以`*.npy`格式存储（不要带pickle）。

## 训练结果存储的架构

我们将训练得到的参数、log信息以及figure存储在工作路径中（默认存储在task文件夹下），以在CIFAR10上训练的ResNet18为例：

```
resnet18_cifar_cifar10_2021-08-18-10-50_19321c36
├── active_round_0
├── active_round_1
│   ├── figures
│   │   ├── train
│   │   ├── eval
│   │   ├── test
│   ├── active_round_1-label_num_2000-epoch_100.pth
│   ├── active_round_1-label_num_2000-epoch_200.pth
│   ├── ...
├── ...
├── active_round_20
├── 20210818_105051.log
├── 20210818_105051.log.json
├── acc_num_labels.png
├── latest.pth
├── ...
```

其中：
- `latest.pth`中存储所有resume的必要信息。
- `log`及`log.json`中存储所有分析用的必要信息，包含每个iter及epoch中的loss和acc，以及每个round中标注集合查询集。
- `acc_num_labels.png`仅在训练完成后出现，表示acc随样本标注量增加的曲线。
- `active_round_{num}`文件夹下存储训练模型参数以及画出的所有图像，数据集中的训练集/验证集/测试集会各有一组图像。
  - 所有图像都集中在`figure`子文件夹下。为了节约时间，降维后的样本坐标也会以`*.npy`格式存储在该文件夹下。

## 目前可用的模型名称

|model|Input size|notes|
|---|---|---|
|resnet18_cifar|32 x 32|专用于CIFAR系列数据集的ResNet18|
|resnet34_cifar|32 x 32|同上|
|resnet50_cifar|32 x 32|同上|
|resnet101_cifar|32 x 32|同上|
|resnet152_cifar|32 x 32|同上|
|resnet18|224 x 224|torchvision中的原模型|
|resnet34|224 x 224|同上|
|resnet50|224 x 224|同上|
|resnet101|224 x 224|同上|
|resnet152|224 x 224|同上|
|resnext50_32x4d|224 x 224|同上|
|resnext101_32x8d|224 x 224|同上|
|wide_resnet50_2|224 x 224|同上|
|wide_resnet101_2|224 x 224|同上|
|toynet|1 x 2|用于对二维点集进行分类的简单架构|

## 目前可用的主动学习策略名称

|strategy|notes|
|---|---|
|RandomSampling||
|LeastConfidence||
|MarginSampling||
|EntropySampling||
|LeastConfidenceDropout||
|MarginSamplingDropout||
|EntropySamplingDropout||
|KMeansSampling||
|KCenterGreedy||
|BALDDropout||
|CoreSet||
|AdversarialBIM||
|AdversarialDeepFool||
