import os
from datetime import datetime

import json
import csv
from sklearn.metrics import average_precision_score


def save_path(opt):

    date = datetime.today().strftime("%Y%m%d%H%M") 

    if opt.method in ['scratch', 'LP', 'label_smoothing']:
        date_method = os.path.join(opt.dataset, opt.method, opt.student_model, date)
    else:
        date_method = os.path.join(opt.dataset, opt.method, '_'.join(opt.student_model, opt.smart_model), date)
        
            
    if not os.path.exists(os.path.join(opt.result, date_method)):
        os.makedirs(os.path.join(opt.result, date_method))
    with open(os.path.join(opt.result, date_method, 'opt.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)
    print(os.path.join(opt.result, date_method))
    save_model_path = os.path.join(opt.result, date_method, opt.save_model_path)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    
    if not os.path.exists(os.path.join(opt.result, date_method,'logfile')):
        os.makedirs(os.path.join(opt.result, date_method,'logfile'))
    
    return opt.result, date_method, save_model_path

class AverageMeter(object):
    """
    Computes and stores the average and
    current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()

def get_ap_score(y_true, y_scores):
    """
    Get average precision score between 2 1-d numpy arrays
    
    Args:
        y_true: batch of true labels
        y_scores: batch of confidence scores
=
    Returns:
        sum of batch average precision
    """
    scores = 0.0
    
    for i in range(y_true.shape[0]):
        scores += average_precision_score(y_true = y_true[i], y_score = y_scores[i])
    
    return scores / y_true.shape[0]


def cifar2ann():
    object_categories = {'airplane':0,  'automobile':1,  'bird':2,  'cat':3,  'deer':4,  'dog':5,  'frog':6,  'horse':7,  'ship':8,  'truck':9}
    data_path = '/raid/video_data/cifa/cifar10/test'


    f = open('/raid/video_data/cifa/cifar10/test.txt', 'w')
    for object_name in os.listdir(data_path):
        class_label = os.path.join(data_path, object_name)
        for img in os.listdir(class_label):
            img_path = os.path.join(data_path, object_name, img)
            classes = object_categories[object_name]
            f.write(img_path + ',' + str(classes) + '\n')

    f.close()
if __name__ == '__main__':
    cifar2ann()