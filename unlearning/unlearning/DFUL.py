import torch
from myModels import LeNet5, m_LeNet

def unlearning_DFUL(global_model,Var,epochs,idx):
    local_model_set = Var.get_var("client_model")
    change_models = Var.get_var("global_change_model")
    for i in range(epochs-1,epochs):
        local_t = local_model_set[i]
        # 删除操作
        delUserModel = local_t[idx]
        model1 = model_neg(delUserModel)
        global_model = agg_model_plus(model1, global_model)

        #修正操作
        # tri = 1 / (epochs - i)
        # for j in range(i + 1, epochs):
        #     local_j = local_model_set[j][idx]
        #     global_change_model = change_models[j]
        #     model2 = agg_model_neg(global_change_model, local_model_set[j], tri)
        #     #model2 = agg_model_neg(global_change_model, local_j, epochs, tri)
        #     global_model = model_matrix_cheng(global_model, model2, delUserModel, abs(j-epochs), tri)
        break
    return global_model

def agg_model_plus(model1,model2):
    for name, param in model1.named_parameters():
        model1.state_dict()[name].copy_(model1.state_dict()[name].data + model2.state_dict()[name].data/6)
    return model1

def model_neg(model1):
    model = m_LeNet.Net()
    for name, param in model1.named_parameters():
        model.state_dict()[name].copy_(-model1.state_dict()[name].data)
    return model

# def agg_model_neg(model1,model2,M,tri):
#     model = m_LeNet.Net()
#     for name, param in model1.named_parameters():
#         model.state_dict()[name].copy_((M*model1.state_dict()[name].data-model2.state_dict()[name].data)/(M-1)*tri)
#     return model

def agg_model_neg(model1,local,tri):
    model = m_LeNet.Net()
    a = local[3:]
    for name, param in model1.named_parameters():
        model.state_dict()[name].copy_((len(local)*model1.state_dict()[name].data-a[0].state_dict()[name].data-a[1].state_dict()[name].data-a[2].state_dict()[name].data)/(len(local)-3))
    return model

def model_matrix_cheng(global_model,model1,model2,n,tri):
    model = m_LeNet.Net()
    for name, param in model1.named_parameters():
        #a = model2.state_dict()[name].data/torch.max(torch.abs(model2.state_dict()[name].data)).item()
        norm = torch.abs(model2.state_dict()[name].data)/torch.norm(model2.state_dict()[name].data, p=2)
        #print(torch.norm(model2.state_dict()[name].data, p=2))
        # n = model2.state_dict()[name].data.shape
        # a_flat = model1.state_dict()[name].data.view(n[0], -1)
        # b_flat = model2.state_dict()[name].data.view(n[0], -1)
        # proj = torch.mm(a_flat, a_flat.t())/torch.mm(a_flat.t(), a_flat)
        # print(proj)
        model.state_dict()[name].copy_(global_model.state_dict()[name].data + model1.state_dict()[name].data*norm*n/tri)
    return model

