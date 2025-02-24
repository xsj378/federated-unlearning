from utils import Arguments
import numpy as np

args = Arguments.Arg()

def Mean_remove(loss,j):
    loss1 = []
    for i in range(len(loss)):
        if(i != j):
            loss1.append(loss[i])
    return np.mean(loss1)

def ca_impact(Var):
    all_start_loss = Var.get_var("user_loss_start_epoch")
    all_end_loss = Var.get_var("user_loss_end_epoch")
    time = args.epochs
    client_number = args.client_number
    impact = [0 for i in range(client_number)]

    for i in range(time-1):
        end_loss = all_end_loss[i]
        start_loss = all_start_loss[i+1]
        for j in range(client_number):
            impact[j] += Mean_remove(end_loss,j) - Mean_remove(start_loss,-1)

    return np.array(impact)-np.mean(impact)