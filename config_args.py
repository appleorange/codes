import os.path as path
import os
import numpy as np
from pdb import set_trace as stop


def get_args(parser, eval=False):
    parser.add_argument('--dataroot', type=str, default='./data/')
    parser.add_argument('--dataset', type=str,
                        choices=['coco', 'voc', 'coco1000', 'nus', 'vg', 'news', 'cub', 'youhome_multi',
                                 'youhome_multi_cross'], 
                                 default='youhome_multi') 
                                 ##default='coco')
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--results_dir', type=str, default='results/')

    # Optimization
    parser.add_argument('--optim', type=str, choices=['adam', 'sgd'], default='adam')
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=-1)
    parser.add_argument('--grad_ac_steps', type=int, default=1)
    parser.add_argument('--scheduler_step', type=int, default=1000)
    parser.add_argument('--scheduler_gamma', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=3)
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
    parser.add_argument('--crop_size', type=int, default=576)

    # Testing Models
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--saved_model_name', type=str, default='')

    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--name', type=str, default='')
    args = parser.parse_args()
    print(args.saved_model_name)
    model_name = args.dataset
    if args.dataset == 'youhome_multi':
        args.num_labels = 45 #73
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
