import os
import torch
import time
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from utils import Arguments
from utils.ModelUtils import *
from myModels import LeNet5, m_LeNet

args = Arguments.Arg()
Net = m_LeNet.Net()
r = 1


# 梯度上升
def gradient_up(model: Net, lr):
    model.fc1.weight.data = model.fc1.weight.data + lr * model.fc1.weight.grad.data
    model.fc1.bias.data = model.fc1.bias.data + lr * model.fc1.bias.grad.data
    model.fc2.weight.data = model.fc2.weight.data + lr * model.fc2.weight.grad.data
    model.fc2.bias.data = model.fc2.bias.data + lr * model.fc2.bias.grad.data
    model.fc3.weight.data = model.fc3.weight.data + lr * model.fc3.weight.grad.data
    model.fc3.bias.data = model.fc3.bias.data + lr * model.fc3.bias.grad.data


#  限制模型参数不远离参考模型
def gradient_up_1(model, Wref, lr, a):
    model.fc1.weight.data = (1 - a) * model.fc1.weight.data + lr * model.fc1.weight.grad.data + a * Wref.fc1.weight.data
    model.fc1.bias.data = model.fc1.bias.data + lr * model.fc1.bias.grad.data
    model.fc2.weight.data = (1 - a) * model.fc2.weight.data + lr * model.fc2.weight.grad.data + a * Wref.fc2.weight.data
    model.fc2.bias.data = model.fc2.bias.data + lr * model.fc2.bias.grad.data
    model.fc3.weight.data = (1 - a) * model.fc3.weight.data + lr * model.fc3.weight.grad.data + a * Wref.fc3.weight.data
    model.fc3.bias.data = model.fc3.bias.data + lr * model.fc3.bias.grad.data


# 聚合：FedAvg
def fedavg_updata_weight(model, c_model, cn, n):
    data = 0
    for i in range(len(c_model)):
        data += c_model[i].conv1.weight.data * cn[i]
    model.conv1.weight.set_(data / n)

    data = 0
    for i in range(len(c_model)):
        data += c_model[i].conv1.bias.data * cn[i]
    model.conv1.bias.set_(data / n)

    data = 0
    for i in range(len(c_model)):
        data += c_model[i].conv2.weight.data * cn[i]
    model.conv2.weight.set_(data / n)

    data = 0
    for i in range(len(c_model)):
        data += c_model[i].conv2.bias.data * cn[i]
    model.conv2.bias.set_(data / n)

    data = 0
    for i in range(len(c_model)):
        data += c_model[i].fc1.weight.data * cn[i]
    model.fc1.weight.set_(data / n)

    data = 0
    for i in range(len(c_model)):
        data += c_model[i].fc1.bias.data * cn[i]
    model.fc1.bias.set_(data / n)

    data = 0
    for i in range(len(c_model)):
        data += c_model[i].fc2.weight.data * cn[i]
    model.fc2.weight.set_(data / n)

    data = 0
    for i in range(len(c_model)):
        data += c_model[i].fc2.bias.data * cn[i]
    model.fc2.bias.set_(data / n)

    data = 0
    for i in range(len(c_model)):
        data += c_model[i].fc3.weight.data * cn[i]
    model.fc3.weight.set_(data / n)

    data = 0
    for i in range(len(c_model)):
        data += c_model[i].fc3.bias.data * cn[i]
    model.fc3.bias.set_(data / n)

    return model


# 参考模型
def ref(model, target_model, cn, n, test_data):
    test_dataset = torch.utils.data.TensorDataset(test_data[0], test_data[1])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)
    Wref = m_LeNet.Net()
    n1 = 0
    data = 0
    for i in range(len(target_model)):
        data += target_model[i].conv1.weight.data * cn[-i - 1]
        n1 += cn[-i - 1]
    Wref.conv1.weight.data = ((n * model.conv1.weight.data - data) / (n - n1))

    data = 0
    for i in range(len(target_model)):
        data += target_model[i].conv1.bias.data * cn[-i - 1]
    Wref.conv1.bias.data = ((n * model.conv1.bias.data - data) / (n - n1))

    data = 0
    for i in range(len(target_model)):
        data += target_model[i].conv2.weight.data * cn[-i - 1]
    Wref.conv2.weight.data = ((n * model.conv2.weight.data - data) / (n - n1))

    data = 0
    for i in range(len(target_model)):
        data += target_model[i].conv2.bias.data * cn[-i - 1]
    Wref.conv2.bias.data = ((n * model.conv2.bias.data - data) / (n - n1))

    data = 0
    for i in range(len(target_model)):
        data += target_model[i].fc1.weight.data * cn[-i - 1]
    Wref.fc1.weight.data = ((n * model.fc1.weight.data - data) / (n - n1))

    data = 0
    for i in range(len(target_model)):
        data += target_model[i].fc1.bias.data * cn[-i - 1]
    Wref.fc1.bias.data = ((n * model.fc1.bias.data - data) / (n - n1))

    data = 0
    for i in range(len(target_model)):
        data += target_model[i].fc2.weight.data * cn[-i - 1]
    Wref.fc2.weight.data = ((n * model.fc2.weight.data - data) / (n - n1))

    data = 0
    for i in range(len(target_model)):
        data += target_model[i].fc2.bias.data * cn[-i - 1]
    Wref.fc2.bias.data = ((n * model.fc2.bias.data - data) / (n - n1))

    data = 0
    for i in range(len(target_model)):
        data += target_model[i].fc3.weight.data * cn[-i - 1]
    Wref.fc3.weight.data = ((n * model.fc3.weight.data - data) / (n - n1))

    data = 0
    for i in range(len(target_model)):
        data += target_model[i].fc3.bias.data * cn[-i - 1]
    Wref.fc3.bias.data = ((n * model.fc3.bias.data - data) / (n - n1))

    print('Wref：')
    test(Wref, test_loader)  # 测试限制模型
    return Wref


#  欧氏距离
def l2_penalty(w):
    return ((w ** 2).sum()) ** 0.5


#  忘却训练
def unlearn(model: Net, unl_loader, Wref, lr):
    # optimizer =-098765= optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    model.train()
    k = 0
    test(model, unl_loader)
    for epoch_ind, (data, target) in enumerate(unl_loader):
        # data = data.send(client)
        # target = target.send(client)
        # optimizer.zero_grad()
        output = model(data)
        w1 = Wref.fc1.weight.data - model.fc1.weight.data
        w2 = Wref.fc2.weight.data - model.fc2.weight.data
        w3 = Wref.fc3.weight.data - model.fc3.weight.data
        Loss = F.nll_loss(output, target)
        # print("Loss=", Loss.data)
        Loss.backward()  # 反向传播的梯度计算
        if (l2_penalty(w1) >= r) + (l2_penalty(w2) >= r) + (l2_penalty(w3) >= r):
            print("Use gradient_up_1")
            gradient_up_1(model, Wref, lr, args.a)
        else:
            print("Use gradient_up")
            gradient_up(model, lr)
        if test(model, unl_loader) <= args.t:  # 测试劣质D测试集在David模型上的准确率
            print("acc<{:.2f}%. Early stop".format(100. * args.t))
            k = 1
            return k
        return k


#  恢复训练
def train(model: Net, train_loader: torch.utils.data.DataLoader):
    j = 0
    model.train()
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch_ind, (data, target) in enumerate(train_loader):
        if j > args.e - 1:
            break
        else:
            j += 1
            # data, target = data.to(device), target.to(device)
            opt.zero_grad()
            pred = model(data)
            loss = F.cross_entropy(pred, target)
            loss.backward()
            opt.step()  # 更新参数
            print("There is epoch:{} epoch_ind:{} in loss:{:.6f}".format(1, epoch_ind, loss.data.item()))


# #  测试
# def test(model, test_data):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         test_dataset = torch.utils.data.TensorDataset(test_data[0], test_data[1])
#         test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)
#         for epoch_ind, (data, target) in enumerate(test_loader):
#             # data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.cross_entropy(output, target, reduction='sum').item()
#             # test_loss += loss_function(output, target).item()
#             pred = output.argmax(1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()
#         test_loss /= len(test_data[0])
#         print('Test set : Average loss : {:.6f}, Accuracy: {}/{} ( {:.2f}%)'.format(
#             test_loss, correct, len(test_data[0]),
#             100. * correct / len(test_data[0])))
#         acc = correct / len(test_data[0])
#         return acc
def test(model, loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            # data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(loader.dataset)
        print('Test set : Average loss : {:.4f}, Accuracy: {}/{} ( {:.2f}%)'.format(
            test_loss, correct, len(loader.dataset),
            100. * correct / len(loader.dataset)))
        acc = correct / len(loader.dataset)
        return acc


# 分配好坏数据
def tar_datadistri(tar_num, low_data, good_data):
    low_data1, low_targets = low_data
    good_data1, good_targets = good_data
    low_len = len(low_data1)
    good_len = len(good_data1)
    print(low_len, good_len)
    low_dataset = []
    good_dataset = []
    for i in range(tar_num):
        indx1, indy1 = int(i * low_len / tar_num), int((i + 1) * low_len / tar_num)
        print(indx1, indy1)
        low_dataset.append((low_data1[indx1:indy1], low_targets[indx1:indy1]))
        indx2, indy2 = int(i * good_len / tar_num), int((i + 1) * good_len / tar_num)
        print(indx2, indy2)
        good_dataset.append((good_data1[indx2:indy2], good_targets[indx2:indy2]))
    return low_dataset, good_dataset


def federated_unlearning(model, client_model, target_model, low_data, good_data, test_data, cn, n):
    Wref = ref(model, target_model, cn, n, test_data)
    test_dataset = torch.utils.data.TensorDataset(test_data[0], test_data[1])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)
    tar_num = len(target_model)
    low_dataset, good_dataset = tar_datadistri(tar_num, low_data, good_data)  # 劣质数据集和优质数据集
    # 每个目标客户单独进行忘却训练和提升训练
    for i in range(tar_num):
        # target_model[i].train()
        # target_model[i].send(client_list[-i - 1])
        print("--------------------target_client{}--------------------".format(i + 1))
        print('忘却训练：')
        unlearn_dataset = torch.utils.data.TensorDataset(low_dataset[-i-1][0], low_dataset[-i-1][1])
        unlearn_loader = torch.utils.data.DataLoader(unlearn_dataset, batch_size=args.unl_batch_size, shuffle=True)
        for epoch in range(1, args.unl_epochs + 1):
            print("---------------epoch{}--------------".format(epoch))
            k = unlearn(target_model[i], unlearn_loader, Wref, args.lr)  # 忘却劣质D,训练集
            if k == 1:
                break
        print("提升训练：")
        boost_dataset = torch.utils.data.TensorDataset(good_dataset[-i-1][0], good_dataset[-i-1][1])
        boost_loader = torch.utils.data.DataLoader(boost_dataset, batch_size=args.batch_size, shuffle=True)
        train(target_model[i], boost_loader)
        # target_model[i].get()
        test(target_model[i], test_loader)
        # 用经过unlearning操作的模型代替原目标客户的模型，为聚合做准备
        client_model[-i - 1] = target_model[i]
        n -= cn[-i - 1]
        cn[-i - 1] = len(good_dataset[-i-1][0])
        n += cn[-i - 1]
    with torch.no_grad():
        # 更新权重
        fedavg_updata_weight(model, client_model, cn, n)
    test(model, test_loader)
    if args.save_model:
        torch.save(model.state_dict(), "unlearning_mnist5.pt")
    print("Success")
    return model
