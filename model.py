import os
import torch
import torch.nn as nn


from models.small_resnet import resnet20, resnet32, resnet44, resnet56, resnet110
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
model_dict = {
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    }
def StudentModel(opt):
    model = model_dict[opt.student_model](num_classes=opt.n_class)
    return model
def SmartModel(opt):
    model = model_dict[opt.smart_model](num_classes=opt.n_class)
    
    model_file = os.path.join(opt.smart_model_path, opt.dataset, opt.smart_model,'student_lastest.pth')
    checkpoint = torch.load(model_file)#, map_location=lambda storage, loc: storage)

    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)


    return model


class LabelPerturbation(nn.Module):
    def __init__(self, in_features, out):
        super(LabelPerturbation, self).__init__()
       
        self.perturbation = nn.Sequential(
                        nn.Linear(in_features, in_features // 2),                   
                        nn.ReLU(),
                        nn.BatchNorm1d(in_features // 2),
                        nn.Linear(in_features // 2, out)                          # batch x 32 x 28 x 28    
        )


    def forward(self,x):
 
        x = self.perturbation(x)
            
        return x