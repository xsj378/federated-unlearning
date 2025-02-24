from data import datapro
from learning import federated_learning
from myModels import m_LeNet
from utils import Arguments
from datadistri import client_create
from unlearning.fed_unlearning import ref, federated_unlearning

args = Arguments.Arg()

client_number = args.client_number
model = m_LeNet.Net()
epoch = args.epochs
target_number = int(client_number * args.target_client_ratio)
ratio = args.target_client_ratio
print("客户端数量：{}，目标客户数量：{}".format(client_number, target_number))

if __name__ == '__main__':
    print("----------1.处理数据阶段----------")
    # 分配给正常客户的数据，分配给目标客户的数据(2个：优质数据、混合数据），劣质数据，测试数据
    tran_dataset1, tran_right_dataset, tran_poll_dataset, low_data, test_dataset = datapro.Pollution_main("mnist",

                                                                                                          ratio)
    print("----------2.联邦学习阶段----------")
    # 创建客户端，分配数据
    client_list, client_data = client_create(client_number, tran_dataset1, tran_right_dataset)
    # 正常训练
    for i in range(epoch - 1):
        print("---------------------------epoch={}---------------------------".format(i + 1))
        model, client_model = federated_learning(client_number, client_list, client_data, model, test_dataset)

    print("----------3. {:.2f}%的客户端加入劣质数据----------".format(100. * args.target_client_ratio))
    # 加入劣质数据
    client_list, client_data = client_create(client_number, tran_dataset1, tran_poll_dataset)
    # 继续训练1轮
    print("----------epoch={}----------".format(epoch))
    model, client_model = federated_learning(client_number, client_list, client_data, model, test_dataset)

    print("----------4.识别目标客户端----------")

    print("----------5.忘却学习阶段----------")
    target_model = []
    for i in range(target_number):
        target_model.append(client_model[- i - 1])  # 目标客户的局部模型
    print("target_client:{}".format(len(target_model)))
    cn = []
    n = 0
    for i in range(client_number):
        cn.append(int(len(client_data[i][0])))  # 每个客户端的数据量
        n += int(len(client_data[i][0]))  # 总数据量
    Wref = ref(model, target_model, cn, n, test_dataset)  # 限制模型
    model = federated_unlearning(model, client_model, target_model, low_data, tran_right_dataset, test_dataset, Wref, cn, n)
