class Arguments:
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 64
        self.epochs = 4
        self.local_epochs = 3
        self.lr = 0.001
        self.momentum = 0.9
        # self.no_cuda = False
        self.seed = 1
        self.log_interval = 1
        self.save_model = True
        self.client_number = 6
        self.save_path = r"C:/Users/xsj/Desktop/unlearning/save"
        self.unl_batch_size = 1024
        self.unl_epochs = 5
        self.a = 1  # 衰减率
        self.e = 5  # 恢复训练轮次
        self.t = 0.10  # 提早-停止阈值

def Arg():
    return Arguments()

if __name__ == '__main__':
    print(Arg())
