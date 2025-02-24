import torch
import torchvision
import random

#下载数据集
def downData(name="minist"):
    if name =="minist":
        train_dataset = torchvision.datasets.MNIST(root='C:/Users/xsj/Desktop/unlearning/data', train=True,
                                                   transform=torchvision.transforms.ToTensor(), download=True)
        test_dataset = torchvision.datasets.MNIST(root='C:/Users/xsj/Desktop/unlearning/data', train=False, transform=torchvision.transforms.ToTensor())
    return train_dataset,test_dataset

#数据污染
def dataPollution(train_dataset,ratio):
    # 获取数据
    ratio = 1 - ratio
    dataLen = len(train_dataset)
    right_data, right_targets = train_dataset.data, train_dataset.targets
    poll_data,poll_targets = train_dataset.data[int(dataLen * ratio):],train_dataset.targets[int(dataLen * ratio):]

    #数据污染
    for i in range(len(poll_targets)):
        poll_targets[i] = random.randint(0, 9)
    right_data = right_data.type(torch.float32).view(len(right_data), 1, 28, 28)
    poll_data = poll_data.type(torch.float32).view(len(poll_data), 1, 28, 28)
    right_targets = right_targets.type(torch.long).view(len(right_targets))
    poll_targets = poll_targets.type(torch.long).view(len(poll_targets))
    return (right_data, right_targets),(poll_data,poll_targets)


def Pollution_main(name,ratio):
    train_dataset,test_dataset = downData(name)
    print(type(test_dataset.data))
    train_right_dataset,train_poll_dataset = dataPollution(train_dataset,ratio)
    test_data = test_dataset.data.type(torch.float32).view(len(test_dataset.data),  1, 28, 28)
    test_targets = test_dataset.targets.type(torch.long).view(len(test_dataset.targets) )
    return train_right_dataset,train_poll_dataset,(test_data,test_targets)

if __name__ == '__main__':
    train_right_dataset,train_poll_dataset,test_dataset = Pollution_main("minist", 0.15)
    # print(train_right_dataset[0].size())