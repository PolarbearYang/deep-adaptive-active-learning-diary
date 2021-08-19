# Tutorial 3: 加入新的主动学习策略

所有的方法都需要从query_strategies.strategy.Strategy基类中继承。与主动学习/自适应主动学习方法相关的核心函数为query和update两项：

- 其中query返回的是新查询的样本集
- update用来执行当前标注池的更新

也可以制定一些额外的函数来辅助这两个成员函数的执行。

上一代的结果指导（持续学习）、软加权等手段还有待添加。

## 新策略的添加

- 在`query_strategies`文件夹下添加该方法，并在`__init__.py`文件中import该类并在`__all__`中添加该方法对应的类名称。以后在调用该策略时，只需输入该类的名称即可。
- 该方法对应的文件中务必添加`from .builder import STRATEGIES`
- 类前面务必添加`@STRATEGIES.register_module()`

一个示例如下：

```python
import numpy as np
from .strategy import Strategy
from .builder import STRATEGIES


@STRATEGIES.register_module()
class LeastConfidence(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(LeastConfidence, self).__init__(X, Y, idxs_lb, net, handler, args)

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        U = probs.max(1)[0]
        return idxs_unlabeled[U.sort()[1][:n]]
```

目前的update函数只是简单粗暴地根据输入更新标注池，未作更改。这是需要调整的部分。
query函数只需以需要查询的样本量作为输入，不要更改输入。
