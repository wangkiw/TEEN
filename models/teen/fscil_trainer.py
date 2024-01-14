import logging
from copy import deepcopy

import torch.nn as nn
from base import Trainer
from dataloader.data_utils import get_dataloader
from utils import *

from .helper import *
from .Network import MYNET


class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_up_model()
        
    def set_up_model(self):
        self.model = MYNET(self.args, mode=self.args.base_mode)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()
        
        if self.args.model_dir is not None:
            logging.info('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir, 
                                              map_location={'cuda:3':'cuda:0'})['params']
        else:
            logging.info('random init params')
            if self.args.start_session > 0:
                logging.info('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())

    def train(self,):
        args = self.args
        t_start_time = time.time()
        # init train statistics
        result_list = [args]
        for session in range(args.start_session, args.sessions):
            train_set, trainloader, testloader = get_dataloader(args, session)
            self.model.load_state_dict(self.best_model_dict)
            if session == 0:  # load base class train img label
                if not args.only_do_incre:
                    logging.info(f'new classes for this session:{np.unique(train_set.targets)}')
                    optimizer, scheduler = get_optimizer(args, self.model)
                    for epoch in range(args.epochs_base):
                        start_time = time.time()
                        
                        tl, ta = base_train(self.model, trainloader, optimizer, scheduler, epoch, args)
                        tsl, tsa = test(self.model, testloader, epoch, args, session, result_list=result_list)

                        # save better model
                        if (tsa * 100) >= self.trlog['max_acc'][session]:
                            self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                            self.trlog['max_acc_epoch'] = epoch
                            save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                            torch.save(dict(params=self.model.state_dict()), save_model_dir)
                            torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                            self.best_model_dict = deepcopy(self.model.state_dict())
                            logging.info('********A better model is found!!**********')
                            logging.info('Saving model to :%s' % save_model_dir)
                        logging.info('best epoch {}, best test acc={:.3f}'.format(
                            self.trlog['max_acc_epoch'], self.trlog['max_acc'][session]))

                        self.trlog['train_loss'].append(tl)
                        self.trlog['train_acc'].append(ta)
                        self.trlog['test_loss'].append(tsl)
                        self.trlog['test_acc'].append(tsa)
                        lrc = scheduler.get_last_lr()[0]

                        logging.info(
                            'epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                                epoch, lrc, tl, ta, tsl, tsa))
                        print('This epoch takes %d seconds' % (time.time() - start_time),
                            '\n still need around %.2f mins to finish this session' % (
                                    (time.time() - start_time) * (args.epochs_base - epoch) / 60))
                        scheduler.step()
                    
                    # Finish base train
                    logging.info('>>> Finish Base Train <<<')
                    result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                        session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session]))
                else:
                    logging.info('>>> Load Model &&& Finish base train...')
                    assert args.model_dir is not None

                if not args.not_data_init:
                    self.model.load_state_dict(self.best_model_dict)
                    self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                    best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                    logging.info('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    torch.save(dict(params=self.model.state_dict()), best_model_dir)

                    self.model.module.mode = 'avg_cos'
                    tsl, tsa = test(self.model, testloader, 0, args, session, result_list=result_list)
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        logging.info('The new best test acc of base session={:.3f}'.format(
                            self.trlog['max_acc'][session]))
            
            # incremental learning sessions
            else:  
                logging.info("training session: [%d]" % session)
                self.model.module.mode = self.args.new_mode
                self.model.eval()
                trainloader.dataset.transform = testloader.dataset.transform
                
                if args.soft_mode == 'soft_proto':
                    self.model.module.update_fc(trainloader, np.unique(train_set.targets), session)
                    self.model.module.soft_calibration(args, session)
                else:
                    raise NotImplementedError
                
                tsl, (seenac, unseenac, avgac) = test(self.model, testloader, 0, args, session, result_list=result_list)

                # update results and save model
                self.trlog['seen_acc'].append(float('%.3f' % (seenac * 100)))
                self.trlog['unseen_acc'].append(float('%.3f' % (unseenac * 100)))
                self.trlog['max_acc'][session] = float('%.3f' % (avgac * 100))
                self.best_model_dict = deepcopy(self.model.state_dict())
                
                logging.info(f"Session {session} ==> Seen Acc:{self.trlog['seen_acc'][-1]} "
                             f"Unseen Acc:{self.trlog['unseen_acc'][-1]} Avg Acc:{self.trlog['max_acc'][session]}")
                result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))
        
        # Finish all incremental sessions, save results.
        result_list, hmeans = postprocess_results(result_list, self.trlog)
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)
        if not self.args.debug:
            save_result(args, self.trlog, hmeans)
        
        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        logging.info(f"Base Session Best epoch:{self.trlog['max_acc_epoch']}")
        logging.info('Total time used %.2f mins' % total_time)
        logging.info(self.args.time_str)
