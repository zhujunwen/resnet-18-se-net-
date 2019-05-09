里面有两个模型文件，一个是resnet.py 一个是res2.py 由于懒得改名字了，所以res2.py里面是resnet-18的模型，而resnet.py是resnet-18+se-net的模型
在train.py中，语句 from xxx import ResNet18 是选择使用哪一个模型的，xxx是res2 或者 resnet 
最后 在train.py直接运行即可
