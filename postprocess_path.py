from utils import ensure_path
import os
import datetime
def sub_set_save_path(args):
    if args.project == 'teen':
        args.save_path = args.save_path +\
            f"-tw_{args.softmax_t}-{args.shift_weight}-{args.soft_mode}"
    else:
        raise NotImplementedError
    return args

def set_save_path(args):
    # base info
    time_str = datetime.datetime.now().strftime('%m%d-%H-%M-%S-%f')[:-3]
    args.time_str = time_str
    mode = args.base_mode + '-' + args.new_mode
    if not args.not_data_init:
        mode = mode + '-' + 'data_init'
    args.save_path = '%s/' % args.dataset
    args.save_path = args.save_path + '%s/' % args.project
    args.save_path = args.save_path + '%s-start_%d/' % (mode, args.start_session)
    
    # optimizer & scheduler
    if args.schedule == 'Milestone':
        mile_stone = str(args.milestones).replace(" ", "").replace(',', '_')[1:-1]
        args.save_path = args.save_path +\
            f'{args.time_str}-Epo_{args.epochs_base}-Bs_{args.batch_size_base}'\
            f'-{args.optim}-Lr_{args.lr_base}-decay{args.decay}-Mom_{args.momentum}'\
            f'-MS_{mile_stone}-Gam_{args.gamma}'
            
    elif args.schedule == 'Step':
        args.save_path = args.save_path +\
            f'{args.time_str}-Epo_{args.epochs_base}-Bs_{args.batch_size_base}'\
            f'-{args.optim}-Lr_{args.lr_base}-decay{args.decay}-Mom_{args.momentum}'\
            f'-Step_{args.step}-Gam_{args.gamma}'
            
    elif args.schedule == 'Cosine':
        args.save_path = args.save_path +\
            f'{args.time_str}-Epo_{args.epochs_base}-Bs_{args.batch_size_base}'\
            f'-{args.optim}-Lr_{args.lr_base}-decay{args.decay}-Mom_{args.momentum}'\
            f'-Max_{args.tmax}'
    else:
        raise NotImplementedError
    
    # feature normalize
    if args.feat_norm:
        args.save_path = args.save_path + '-NormT'
    else:
        args.save_path = args.save_path + '-NormF'
    
    # train mode
    if 'cos' in mode:
        args.save_path = args.save_path + '-T_%.2f' % (args.temperature)
    if 'ft' in args.new_mode:
        args.save_path = args.save_path + '-ftLR_%.3f-ftEpoch_%d' % (
            args.lr_new, args.epochs_new)
    
    # specific parameters
    args = sub_set_save_path(args)

    if args.debug:
        args.save_path = os.path.join('debug', args.save_path)
        
    args.save_path = os.path.join('./checkpoint', args.save_path)
    ensure_path(args.save_path)
    return args
