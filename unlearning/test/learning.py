import syft as sy
import torch
from utils import Arguments
import torch.nn.functional as F

hook = sy.TorchHook(torch)
args = Arguments.Arg()


def test(model, test_data):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        test_dataset = torch.utils.data.TensorDataset(test_data[0], test_data[1])
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)
        for epoch_ind, (data, target) in enumerate(test_loader):
            # data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            # test_loss += loss_function(output, target).item()
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_data[0])
        print('Test set : Average loss : {:.6f}, Accuracy: {}/{} ( {:.2f}%)'.format(
            test_loss, correct, len(test_data[0]),
            100. * correct / len(test_data[0])))


def distribute_model(client_number, model):
    client_model = []
    client_optim = []

    for i in range(client_number):
        client_model.append(model.copy())
        client_optim.append(torch.optim.SGD(client_model[i].parameters(), lr=args.lr, momentum=args.momentum))
    return client_model, client_optim


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


def federated_learning(client_number, client_list, client_data, model, test_data):
    # 分配模型
    client_model, client_optim = distribute_model(client_number, model)
    print("----------开始训练----------")
    for i in range(client_number):
        client_model[i].train()
        client_model[i].send(client_list[i])
    for i in range(client_number):
        print("----------第{}个客户端开启训练----------".format(i + 1))
        train_dataset = torch.utils.data.TensorDataset(client_data[i][0], client_data[i][1])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        for epoch_ind, (data, target) in enumerate(train_loader):
            data = data.send(client_list[i])
            target = target.send(client_list[i])

            client_optim[i].zero_grad()
            pred = client_model[i](data)
            loss = F.nll_loss(pred, target)
            loss.backward()
            client_optim[i].step()
            if (epoch_ind+1) % 30 == 0:
                print("There is epoch:{}, loss:{:.6f}".format(epoch_ind+1, loss.get().data.item()))
    with torch.no_grad():
        # 更新权重
        cn = []
        n = 0
        for i in range(client_number):
            cn.append(int(len(client_data[i][0])))  # 每个客户端的数据量
            n += int(len(client_data[i][0]))  # 总数据量
            print("client {}: ".format(i+1))
            client_model[i].get()
            test(client_model[i], test_data)
        model = fedavg_updata_weight(model, client_model, cn, n)
        print("Global: ")
        test(model, test_data)
    return model, client_model
