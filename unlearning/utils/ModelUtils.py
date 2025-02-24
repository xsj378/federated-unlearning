import torch
import torch.nn as nn
import os

from utils import Arguments

Arg = Arguments.Arg()


class ModelUtils():
    def __init__(self):
        self.path = Arg.save_path + "/models/"

    def save_model(self,Model_name,model):
        Model_path = self.path + Model_name
        torch.save(model.state_dict(), Model_path)
        print("模型保存成功~路径为" + Model_path)

    def load_model(self,Model_name,model):
        Model_path = self.path + Model_name
        model_params = torch.load(Model_path)
        model.load_state_dict(model_params)
        return model