# Tutorial 2: 加入新的模型

只需简单包装一个类，它能够根据参数返回一个模型架构即可。

- 在`architectures`文件夹下添加该方法（指返回模型架构的类，而不是模型架构本身，主要是在该类中封装预训练参数或一些与其他组件名称相关的参数以方便调用），并在`__init__.py`文件中import该类并在`__all__`中添加该方法对应的类名称。以后在调用该数据集时，只需输入该类的名称即可。
- 该方法对应的文件中务必添加`from .builder import MODELS`
- 包装该数据集并返回以上五大信息的类前面务必添加`@MODELS.register_module()`

一个样例如下：

```python
import torch
import torch.nn as nn
from .builder import MODELS


class ToyNet(nn.Module):
    """A Deep Net designed for datasets with two-dimensional data.
       Used for toy datasets like 2D mixed Gaussian.

        Args:
            num_classes: (int)
                Number of classes in the used dataset.
            drop_rate: (float)
                The Dropout rate in all `torch.nn.Dropout` modules.

    """

    def __init__(self, num_classes: int = 3, drop_rate=0.) -> None:
        super(ToyNet, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 20),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(20, 10),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(10, num_classes)

    def forward(self, x: torch.Tensor, with_feature=True) -> torch.Tensor:
        feature = self.classifier(x)
        feature = feature.view(feature.size(0), -1)
        out = self.fc(feature)
        if with_feature:
            return out, feature
        else:
            return out


@MODELS.register_module()
class toynet(object):
    """The class for returning ToyNet architecture"""
    def __init__(self):
        pass

    def __call__(self, **kwargs) -> ToyNet:
        return ToyNet(**kwargs)

```
