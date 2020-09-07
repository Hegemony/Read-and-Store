# Read-and-Store


## Dependencies
* [Python 3.5+](https://www.continuum.io/downloads)
* [PyTorch 0.4.0+](http://pytorch.org/)


### 我们可以直接使用save函数和load函数分别存储和读取Tensor。save使用Python的pickle实用程序将对象进行序列化，然后将序列化的对象保存到disk，使用save可以保存各种对象,包括模型、张量和字典等。而load使用pickle unpickle工具将pickle的对象文件反序列化为内存。


### 在PyTorch中，Module的可学习参数(即权重和偏差)，模块模型包含在参数中(通过model.parameters()访问)。state_dict是一个从参数名称隐射到参数Tesnor的字典对象。

### 注意，只有具有可学习参数的层(卷积层、线性层等)才有state_dict中的条目。优化器(optim)也有一个state_dict，其中包含关于优化器状态以及所使用的超参数的信息。


### PyTorch中保存和加载训练模型有两种常见的方法:

1.仅保存和加载模型参数(state_dict)；
2.保存和加载整个模型。


### 通过save函数和load函数可以很方便地读写Tensor。
### 通过save函数和load_state_dict函数可以很方便地读写模型的参数。
