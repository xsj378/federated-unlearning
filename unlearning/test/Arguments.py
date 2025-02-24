class Arguments:
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 1000
        self.unl_batch_size = 1024
        self.epochs = 2
        self.unl_epochs = 5
        self.lr = 0.01
        self.momentum = 0.9
        # self.no_cuda = False
        self.seed = 1
        self.log_interval = 1
        self.save_model = True
        self.client_number = 20
        self.target_client_ratio = 0.3  # 目标客户端占比
        self.low_ratio = 0.5  # 劣质数据占比
        self.t = 0.10  # 提早-停止阈值
        self.a = 1  # 衰减率
        self.e = 5  # 恢复训练轮次

def Arg():
    return Arguments()

if __name__ == '__main__':
    print(Arg())
