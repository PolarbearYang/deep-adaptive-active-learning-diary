# Getting Started

该文档提供了Deep Adaptive Active Learning框架的基本用法。

## 目录结构

请在代码根目录下创建一个`data`文件夹以存储使用的数据集。
若目录结构不同，这需要在`dataset`文件夹下的对应代码中调整路径。
我们目前使用的数据集有CIFAR10, CIFAR100, MNIST, FashionMNIST, SVHN。
若数据集不存在，程序会自动下载并提取这些数据集。

```
deep-adaptive-active-learning
├── architectures
├── datasets
├── plotter
├── query_strategies
├── utils
├── data
│   ├── cifar10
│   │   ├── cifar-10-batches-py
│   ├── cifar100
│   │   ├── cifar-100-python
│   ├── MNIST
│   │   ├── processed
│   │   ├── raw
│   ├── SVHN
│   │   ├── train_32x32.mat
│   │   ├── test_32x32.mat
│   │   ├── extra_32x32.mat

```

## 训练一个模型

我们分别为CIFAR10, CIFAR100, SVHN, MNIST, FashionMNIST五个数据集各提供了一个脚本
（分别是`cifar10.sh`,`cifar100.sh`,`svhn.sh`,`mnist.sh`,`fashionmnist`），
均默认使用entropy sampling策略。目前均只支持单GPU训练。

使用方式以cifar10为例：

```shell
bash cifar10.sh
```

bash文件中需要指定的参数解释如下文档所示:

- [训练配置](train.md)

## 画图说明

画图相关的功能都封装在`plotter`文件夹下，每个active round会进行一次作图。
目前正在考虑

- [作图功能说明](plotter/plotter.md)

## 教程文件

若要执行另外一些个性化功能，我们提供了如下教程：

- [加入新的数据集](tutorials/new_dataset.md)
- [加入新的模型](tutorials/new_model.md)
- [设计新的策略](tutorials/new_strategy.md)
