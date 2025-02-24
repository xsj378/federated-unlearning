import syft as sy
import torch
from utils import Arguments
import torch.nn.functional as F
import random
import pandas as pd

hook = sy.TorchHook(torch)
args = Arguments.Arg()

def test(model, test_data):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        test_dataset = torch.utils.data.TensorDataset(test_data[0],test_data[1])
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)
        for epoch_ind, (data, target) in enumerate(test_loader):
            # data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            # test_loss += loss_function(output, target).item()
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_data[0])
        print('\nTest set : Average loss : {:.4f}, Accuracy: {}/{} ( {:.2f}%)\n'.format(
            test_loss, correct, len(test_data[0]),
            100. * correct / len(test_data[0])))
    return test_loss,100. * correct / len(test_data[0])

# def distribute_data_songwei(client_number,tran_right_data,tran_poll_data,model):
#     indx,indy = 0,0
#     right_data, right_targets = tran_right_data
#     total_len = len(right_data)
#     client_data = []
#     client_model = []
#     client_optim = []
#     # n-1个客户端数据均正常
#     for i in range(client_number-1):
#         indx,indy = int(i*total_len/client_number),int((i+1)*total_len/client_number)
#         client_data.append((right_data[indx:indy],right_targets[indx:indy]))
#     client_data.append(tran_poll_data)
#
#     for i in range(client_number):
#         client_model.append(model.copy())
#         client_optim.append(torch.optim.SGD(client_model[i].parameters(), lr=args.lr, momentum=args.momentum))
#     return client_data,client_model,client_optim,model.copy()
def distribute_data_wei(client_number,model):
    client_model = []
    client_optim = []
    for i in range(client_number):
        client_model.append(model.copy())
        client_optim.append(torch.optim.SGD(client_model[i].parameters(), lr=args.lr, momentum=args.momentum))
    return client_model,client_optim,model.copy()

def fedavg_updata_weight(model,client_list,global_change_model):
    n = 1/len(client_list)
    for i in range(len(client_list)):
        client_list[i].get()

    # 计算模型的聚合更新
    for name, param in global_change_model.named_parameters():
        data = 0
        for i in range(len(client_list)):
            client_list[i].state_dict()[name].copy_(client_list[i].state_dict()[name].data - model.state_dict()[name].data)
            data += client_list[i].state_dict()[name].data
            client_list[i].state_dict()[name].copy_(client_list[i].state_dict()[name].data + model.state_dict()[name].data)#加这句返回的就是模型而不是更新
        with torch.no_grad():
            global_change_model.state_dict()[name].copy_(data * n)
    #修改全局模型参数
    for name, param in model.named_parameters():
        data = model.state_dict()[name].data + global_change_model.state_dict()[name].data
        with torch.no_grad():
            model.state_dict()[name].copy_(data)

    return model,global_change_model,client_list

def federated_learning(client_number,model,client_data,test_data,Var):
    print("----------1.1 创建数量为{}的客户端----------".format(client_number))
    client_list = []
    for i in range(client_number):
        client_list.append(sy.VirtualWorker(hook, id=str(i)))
    print("----------创建成功----------")
    torch.manual_seed(args.seed)
    print("----------1.2 分配模型----------")
    client_model, client_optim, global_change_model = distribute_data_wei(client_number, model)
    print("----------分配成功----------")
    print("----------1.3 开始训练----------")
    for i in range(client_number):
        client_model[i].train()
        client_model[i].send(client_list[i])
    value = 0

    for i in range(client_number):
        print("----------第{}个客户端开启训练----------".format(i + 1))
        train_dataset = torch.utils.data.TensorDataset(client_data[i][0], client_data[i][1])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        mean_loss = 0
        for epoch in range(args.local_epochs):#局部训练次数
            summ = 0
            k = 0
            for epoch_ind, (data, target) in enumerate(train_loader):
                k = k + 1
                data = data.send(client_list[i])
                target = target.send(client_list[i])
                client_optim[i].zero_grad()
                pred = client_model[i](data)
                loss = F.cross_entropy(pred, target)
                loss.backward()
                client_optim[i].step()
                value = loss.get().data.item()
                summ = summ + value
                if k % 100 == 0:
                    print("There is epoch:{} epoch_ind:{} in loss:{:.6f}".format(k, epoch_ind,value))
            mean_loss = summ/len(test_data[0])
            if epoch == 0:
                Var.insert_var("user_loss_start_epoch",mean_loss, format="list", epoch=True)
            if epoch == args.local_epochs - 1:
                Var.insert_var("user_loss_end_epoch",mean_loss, format="list", epoch=True)
        Var.insert_var("train_loss_epoch",mean_loss,format = "list",epoch= True)
    with torch.no_grad():
        # 更新权重
        model,global_change_model,client_model = fedavg_updata_weight(model, client_model,global_change_model)
        loss,acc = test(model, test_data)
        Var.insert_var("client_model", client_model, format="list")
        Var.insert_var("acc", acc, "list")
        Var.insert_var("global_change_model", global_change_model,"list")
    return model




