# 作图功能说明

我们目前知道图像都存储在`工作路径/active_round_{num}/figures`文件夹下，并且我们对训练/验证/测试集各画了一组图像，来可视化训练过程中分布产生偏移的情况。每个active round结束后都会作一次图。目前已有的图像包括：

- 条形图
  - `entropy_bar_in_range`: 显示不同entropy区间样本的数量
  - `entropy_bar_in_rank`: 显示不同entropy(by rank)区间样本的数量
  - `incorrect_bar`: 显示每一个类别中被模型标错的样本数量
  - `correct_incorrect_bar_by_class`: 显示每一个类别中被模型标对/标错的样本数量对比
  - `labeled_bar`: 显示每一个类别中标注的样本数量
  - `queried_bar_by_class`: 显示每一个类别中查询的样本数量
  - `incorrect_queried_bar_by_class`: 显示每一个类别在查询集中被模型标错的样本数量对比
  - `correct_incorrect_queried_bar_by_class`: 显示每一个类别在查询集中被模型标对/标错的样本数量对比
- 二维散点图，将数据降维后显示其分布
  - `{method}_by_class`: 画出所有样本根据模型输出的特征向量分布（降为2维）
  - `{method}_by_labeled_or_not`: 画出所有有标注/无标注样本的分布
  - `{method}_by_correct_or_not`: 画出所有被模型标对/标错的样本分布
  - `{method}_by_query` 画出每一代查询的点相对于全数据集的分布
  - `{method}_by_{value}`画出所有样本的热度图（关于某一个值，比如熵、加权、其他得分、度量等）
    - 这个功能比较花时间，需要优化

如果该文件夹下产生了另一个`with_predefined`文件夹，则该文件夹下所有的图都是基于某种预定义的分布进行的（预训练好的模型仅用于作图，可能会更好地显示数据集的实际分布）。

## 其他一些小细节

- 第一个active round中没有任何关于query的信息，这些图像将被忽略。
- 数据集中的eval和test部分不存在标注/query/加权的情况，因此没有与这三者相关的信息。
