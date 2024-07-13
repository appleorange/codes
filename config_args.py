import os.path as path
import os
import numpy as np
from pdb import set_trace as stop


def get_args(parser, eval=False):
    parser.add_argument('--dataroot', type=str, default='./data/')
    # if run_testing_only is True, then the model will be loaded from the saved model and run testing only.
    parser.add_argument('--run_testing_only', type=bool, default=False)
    parser.add_argument('--dump_testing_details', type=bool, default=True)
    parser.add_argument('--save_debugging_to_gdrive', type=bool, default=True)

    # a number of pretrained models are available in torchvision.models
    # resnet18, resnet34, resnet50, efficientnet-b0, efficientnet-b1, efficientnet-b2, efficientnet-b3, efficientnet_v2s, efficientnet_v2m
    parser.add_argument('--model', type=str, default='resnet18')
    

    parser.add_argument('--load_saved_model', type=bool, default=False)
    parser.add_argument('--load_from_saved_model_name', type=str, default='best_model.pth')
    parser.add_argument('--save_best_model_to_gdrive', type=bool, default=True)
    parser.add_argument('--save_model_dir', type=str, default="./models") #default="/content/drive/MyDrive/UIUC_research/models")

    #resnet18: 80, *, 60 # 32 best for T4, 128 best for L4
    #resnet34: 64, *, 32
    parser.add_argument('--batch_size', type=int, default=16) # 32 best for T4, 128 best for L4
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--workers', type=int, default=10) # 24 best for T4 high memory, 20 best for L4

    parser.add_argument('--dataset', type=str,
                        choices=['coco', 'voc', 'coco1000', 'nus', 'vg', 'news', 'cub', 'youhome_multi', 'youhome_activity',
                                 'youhome_multi_cross'], 
                                 default='youhome_activity') 
                                 ##default='coco')
    #parser.add_argument('--workers', type=int, default=20)
    parser.add_argument('--results_dir', type=str, default='results/')

    # Optimization
    parser.add_argument('--optim', type=str, choices=['adam', 'sgd'], default='adam')
    parser.add_argument('--lr', type=float, default=0.0002)
    #parser.add_argument('--batch_size', type=int, default=8) #32
    parser.add_argument('--test_batch_size', type=int, default=-1)
    parser.add_argument('--grad_ac_steps', type=int, default=1)
    parser.add_argument('--scheduler_step', type=int, default=1000)
    parser.add_argument('--scheduler_gamma', type=float, default=0.1)
    #parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--int_loss', type=float, default=0.0)
    parser.add_argument('--aux_loss', type=float, default=0.0)
    parser.add_argument('--loss_type', type=str, choices=['bce', 'mixed', 'class_ce', 'soft_margin'], default='bce')
    parser.add_argument('--scheduler_type', type=str, choices=['plateau', 'step'], default='plateau')
    parser.add_argument('--loss_labels', type=str, choices=['all', 'unk'], default='all')
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_samples', type=int, default=-1)
    parser.add_argument('--max_batches', type=int, default=-1)
    parser.add_argument('--warmup_scheduler', action='store_true', help='')

    # Image Sizes
    parser.add_argument('--scale_size', type=int, default=640)
    parser.add_argument('--crop_size', type=int, default=384) #384 576 640

    # Testing Models
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--saved_model_name', type=str, default='')

    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--name', type=str, default='')
    args = parser.parse_args()
    print(args.saved_model_name)
    model_name = args.dataset
    if args.dataset == 'youhome_activity':
        args.num_labels = 45
    elif args.dataset == 'youhome_multi':
        args.num_labels = 73
    else:
        print('dataset not included')
        exit()

    model_name += '.bsz_{}'.format(int(args.batch_size * args.grad_ac_steps))
    model_name += '.' + args.optim + str(args.lr)  # .split('.')[1]
    args.train_known_labels = 0

    if args.name != '':
        model_name += '.' + args.name

    # if not os.path.exists(args.results_dir):
    #     os.makedirs(args.results_dir)

    model_name = os.path.join(args.results_dir, model_name)

    args.model_name = model_name

    if args.inference:
        args.epochs = 1

    # if os.path.exists(args.model_name) and (not args.overwrite) and (not 'test' in args.name) and (not eval) and (
    # not args.inference) and (not args.resume):
    #     print(args.model_name)
    #     overwrite_status = input('Already Exists. Overwrite?: ')
    #     if overwrite_status == 'rm':
    #         os.system('rm -rf ' + args.model_name)
    #     elif not 'y' in overwrite_status:
    #         exit(0)
    # elif not os.path.exists(args.model_name):
    #     os.makedirs(args.model_name)

    return args
