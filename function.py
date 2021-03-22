import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import AverageMeter, get_ap_score, accuracy
from sklearn.preprocessing import OneHotEncoder

def CrossEntropy(predicted, target):
    return torch.mean(torch.sum(-nn.Softmax()(target) * torch.nn.LogSoftmax()(predicted), 1))

class LabelSmoothingLoss(nn.Module): 
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__() 
        self.confidence = 1.0 - smoothing 
        self.smoothing = smoothing 
        self.cls = classes 
        self.dim = dim 
    def forward(self, pred, target): 
        pred = pred.log_softmax(dim=self.dim) 
        with torch.no_grad(): # true_dist = pred.data.clone() 
            true_dist = torch.zeros_like(pred) 
            true_dist.fill_(self.smoothing / (self.cls)) 
            for i in range(target.size(0)):
                true_dist[i][int(target[i].data)] += self.confidence
         
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def model_fit(pred, target,train_method, T=1):
    labelsmoothingcrossentropy = LabelSmoothingLoss(classes = int(pred.size(1)))
    if train_method == 'scratch': 
        loss = F.cross_entropy(pred, target)
      
    elif train_method == 'kd':
        loss = T*T*nn.KLDivLoss()(F.log_softmax(pred / T, dim=1), F.softmax(target / T, dim=1))
        
    elif train_method == 'bce':
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        loss = criterion(pred, target)
    elif train_method == 'SL':
        loss =  T*T*CrossEntropy(pred/ T, target/ T)
    elif train_method == 'label_smoothing':
        loss = labelsmoothingcrossentropy(pred, target) 
    return loss


def loss_fn(output_st, Teacher, labels, opt):
    output_sm, perturbation = Teacher
    
    softmax = nn.Softmax()
    totalLoss = 0
    if not perturbation is None:
        perturbation = softmax(perturbation)
        output_sm += perturbation
        totalLoss += model_fit(output_sm, labels, 'ce', T=opt.T) 
        
    if opt.method == 'SL':
        

        if opt.use_gt:
            labels = np.array(labels.cpu().numpy()).reshape(-1,1)
            enc  = OneHotEncoder(n_values=opt.n_class)
            enc.fit(labels)
            one_hot_vector = torch.Tensor(enc.transform(labels).toarray()).cuda()
            labels = (one_hot_vector) + (output_sm)
        else:
            labels = output_sm
        totalLoss += opt.alpha * model_fit(output_st, labels, 'SL', T=opt.T)
        
        return totalLoss
    elif opt.method == 'label_smoothing':
        totalLoss = model_fit(output_st, labels, 'label_smoothing', T=opt.T)
        return totalLoss
    else:
        Student_loss = model_fit(output_st, labels, opt.method, T=opt.T)
        
        if output_sm is not None:
            kd_loss = model_fit(output_st, output_sm, 'kd', T=opt.T)
        

        totalLoss += Student_loss + opt.alpha * (kd_loss)
        return totalLoss




def train(model, device, train_loader, optimizer, epoch, opt):
    # set model as training mode
    studentModel, smartModel, perturbation_model = model
    studentModel.train()
    if opt.isSource:
        smartModel = smartModel.train()
        if opt.isPerturbation:
            perturbation_model = perturbation_model.train()
   

    losses = AverageMeter()
    Targetscores = AverageMeter()


    targetloss = AverageMeter()
    auxiliaryimagenet = AverageMeter()
    auxiliaryplaces = AverageMeter()

    softmax = nn.Softmax()
    sigmoid = torch.nn.Sigmoid()    
    N_count = 0  
    for batch_idx, (images, y) in enumerate(train_loader):
        images, y = images.to(device), y.to(device)
        N_count+= images.size(0)
      
        optimizer.zero_grad()
    
        output_st, feature_st = studentModel(images)

        if opt.isSource:
            with torch.no_grad():
                output_sm, feature_sm = smartModel(images)
                perturbation = None
                if opt.isPerturbation:
                    inputFeature = torch.randn(images.size(0), 100).to(device)
                   

                    perturbation = perturbation_model(inputFeature)

            loss = loss_fn(output_st, [output_sm, perturbation], y, opt)
                
        else:
            loss = model_fit(output_st, y, opt.method, T=opt.T)
            
        losses.update(loss.item(), images.size()[0])

        y_pred = torch.max(output_st, 1)[1]  
        step_score = accuracy(output_st.data, y.data, topk=(1,))[0]
        Targetscores.update(step_score,images.size()[0])        
      
        loss.backward()
        optimizer.step()

        if (batch_idx) % 10 == 0:
            
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                epoch, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), losses.avg, Targetscores.avg))
            
   
    return losses, Targetscores


def validation(model, device, optimizer, test_loader, opt):
    studentModel, smartModel, perturbation_model = model
    studentModel.eval()
    if opt.isSource:
        smartModel = smartModel.eval()
        # if opt.isPerturbation:
        #     perturbation_model = perturbation_model.eval()

    accs = AverageMeter()
    losses = AverageMeter()

    softmax = nn.Softmax()
    sigmoid = torch.nn.Sigmoid()

    with torch.no_grad():
        for images, y in test_loader:
            # distribute data to device
            images, y = images.to(device), y.to(device)
           
            output_st, feature_st = studentModel(images)
         
            if opt.isSource:
                with torch.no_grad():
                    output_sm, feature_sm = smartModel(images)
                    perturbation = None
                    if opt.isPerturbation:
                        inputFeature = torch.randn(images.size(0), 100).to(device)
                        perturbation = perturbation_model(inputFeature)

                loss = loss_fn(output_st, [output_sm, perturbation], y, opt)

            else:
                loss = model_fit(output_st, y, opt.method, T=opt.T)
            losses.update(loss.item(), images.size()[0])                
            prec = accuracy(output_st.data, y.data, topk=(1,))[0]
            accs.update(prec.item(), images.size()[0])
   
        

    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(test_loader.dataset), losses.avg, accs.avg))
  
    
    return losses, accs

