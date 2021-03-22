import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        default='./dataset/', type=str, help='Root of directory path of data'
    )
    parser.add_argument(
        '--dataset',
        default='cifar10', type=str, help='used dataset of cifar10 | cifar100 | imagenet | object detection | Instance segmentation'
    )
    parser.add_argument(
        '--n_class',
        default=10, type=int, help='Number of class'
    )
   
    parser.add_argument(
        '--batch_size',
        default=128, type=int,
    )
    parser.add_argument(
        '--epochs',
        default=200, type=int,
    )
    parser.add_argument(
        '--student_model',
        default='resnet18', type=str, help='student model (resnet | vgg | inception | dense)'
    )

    parser.add_argument(
        '--smart_model',
        default='', type=str, help='places365 pretrained base model'
    )
    parser.add_argument(
        '--smart_model_path',
        default='imagenet', type=str, help='places | cifar pretrained model'
    )
    parser.add_argument(
        '--lr',
        default=1e-3, type=float, help='learning rate'
    )
 
    parser.add_argument(
        '--result',
        default='/raid/video_data/output/result', type=str, help='output path'
    )
    parser.add_argument(
        '--save_model_path',
        default='model_ckp', type=str, help='save_model_path path'
    )

    parser.add_argument(
        '--T',
        default=3, type=float, help='pretrained model'
    )
    parser.add_argument(
        '--alpha',
        default=0.1, type=float, help='pretrained model'
    )

 
    parser.add_argument(
        '--is_smart_model',
        action='store_true', help='Source Network is used'
    )
    parser.set_defaults(isSource=False)
 

    parser.add_argument(
        '--method',
        default='SL',
        help='cel(cross entropy loss) | fl (focal loss)'
    )
   
    parser.add_argument(
        '--use_gt', action='store_true')
    parser.add_argument(
        '--isPerturbation', action='store_true')
        


    parser.add_argument('--optim', default="sgd", type=str,
                    help='optimizer : sgd | adam')

    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')


    args = parser.parse_args()
    return args
    