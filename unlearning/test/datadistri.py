import syft as sy
import torch
from utils import Arguments
import random

hook = sy.TorchHook(torch)
args = Arguments.Arg()


def distribute_data(normal_number, target_number, tran_dataset1, target_dataset):
    # 目标客户端数量
    train_data, train_targets = tran_dataset1
    target_data, target_targets = target_dataset
    normal_len = len(train_data)
    target_len = len(target_data)
    client_data = []
    # 四个客户端数据均正常
    for i in range(normal_number):
        indx, indy = int(i * normal_len / normal_number), int((i + 1) * normal_len / normal_number)
        client_data.append((train_data[indx:indy], train_targets[indx:indy]))
    for i in range(target_number):
        indx, indy = int(i * target_len / target_number), int((i + 1) * target_len / target_number)
        client_data.append((target_data[indx:indy], target_targets[indx:indy]))
    # print(client_data[27])
    return client_data


def client_create(client_number, tran_dataset1, target_dataset):
    print("----------① 创建数量为{}的客户端----------".format(client_number))
    client_list = []
    for i in range(client_number):
        client_list.append(sy.VirtualWorker(hook, id=str(i)))
    print("----------创建成功----------")
    torch.manual_seed(args.seed)
    print("----------② 分配数据----------")
    target_number = int(client_number * args.target_client_ratio)
    normal_number = client_number - target_number
    client_data = distribute_data(normal_number, target_number, tran_dataset1, target_dataset)
    # 数据分布不平衡
    for i in range(normal_number):
        data = client_data[i][1]
        for j in range(len(data)):
            if data[j] in [(i % 10), (i ** 2 % 10)]:
                # print("before:{}".format(data[j]))
                data[j] = random.randint(0, 9)
                # print("after:{}".format(data[j]))
    print("----------分配成功----------")
    return client_list, client_data
