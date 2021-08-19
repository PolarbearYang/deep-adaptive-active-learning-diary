# Tutorial 1: 加入新的数据集

定制数据集需要包含的信息有：

- 每个label对应的类名称
- 对应于该数据集的transform
- 包装该数据集的Dataset类（我们在文件中将其写为各种handler）
- 一个包装该数据集的类，该类别能够返回：
  - 数据集的train/eval/test split的raw data及对应标签
  - 类别数量
  - 每个label映射到类名的字典
  - 用于包装该数据集的Dataset类
  - 封装了训练transform和测试transform的类
    - 参考`datasets.base_transform.base_transform_wrapper`，其能够返回`ord_train_transform`与`ord_eval_transform`这两个属性，对应两种transform（这也得根据模型决定，之后考虑把该类分离出来）

为了方便处理，我们不直接使用torch中默认的数据集封装器，而是直接取出raw data再用我们自己的类进行处理，这样可以方便制作动态数据集。

## 添加数据集

假定我们已经写好了以上类：

- 在`datasets`文件夹下添加该方法，并在`__init__.py`文件中import该类并在`__all__`中添加该方法对应的类名称。以后在调用该数据集时，只需输入该类的名称即可。
- 该方法对应的文件中务必添加`from .builder import DATASETS`
- 包装该数据集并返回以上五大信息的类前面务必添加`@DATASETS.register_module()`

一个例子如下：

```python
from torch.utils.data import Dataset
import numpy as np
from .builder import DATASETS
from .base_transform import base_transform_wrapper

class MixedGaussian_wrapper(base_transform_wrapper):
    """Transformations designed for MixedGaussian Dataset"""
    def __init__(self):
        super(MixedGaussian_wrapper, self).__init__()
        self.ord_train_transform = self.ord_eval_transform = None

class MixedGaussian(Dataset):  # 某个生成混合高斯分布的类
     def __init__(self, centers=[[-2.5, -4.], [2.5, -4.], [0., 0.]],
                 num_per_class=[2000, 2000, 2000],
                 n_dims=2, transform=None, target_transform=None):
        self.data, self.targets = generate_from_mixture(centers, num_per_class, n_dims)
        self.num_classes = len(num_per_class)
        self.transform = transform
        self.target_transform = target_transform

class MixedGaussianHandler(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        return x, y, index

    def __len__(self):
        return len(self.X)


@DATASETS.register_module()
class mixedgaussian(object):
    def __init__(self):
        data_train = MixedGaussian(num_per_class=[20000, 20000, 20000])
        self.X_train = data_train.data.astype(np.float32)
        self.Y_train = torch.from_numpy(data_train.targets)
        data_eval = MixedGaussian(num_per_class=[2000, 2000, 2000])
        self.X_eval = data_eval.data.astype(np.float32)
        self.Y_eval = torch.from_numpy(data_eval.targets)
        data_test = MixedGaussian(num_per_class=[2000, 2000, 2000])
        self.X_test = data_test.data.astype(np.float32)
        self.Y_test = torch.from_numpy(data_test.targets)
        self.handler = MixedGaussianHandler
        self.wrapper = MixedGaussian_wrapper

    def __call__(self):
        return self.X_train, self.Y_train,\
               self.X_eval, self.Y_eval,\
               self.X_test, self.Y_test,\
               3, {0: '0', 1: '1', 2: '2'}, self.handler, self.wrapper()
```
