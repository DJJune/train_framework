

from opt.tst_opts import TestOptions
from utils.logger import get_logger
from models import create_model
import os
from data import create_trn_val_tst_dataset
import time
import numpy as np
from sklearn.metrics import accuracy_score



def eval(model, val_iter, tst_iter, is_save=False, phase='test'):
    model.eval()
    total_pred = []
    total_label = []
    
    for i, data in enumerate(val_iter):  # inner loop within one epoch
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.test()
        pred = model.pred.argmax(dim=1).detach().cpu().numpy()
        label = data['label']
        total_pred.append(pred)
        total_label.append(label)

    for i, data in enumerate(tst_iter):  # inner loop within one epoch
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.test()
        pred = model.pred.argmax(dim=1).detach().cpu().numpy()
        label = data['label']
        total_pred.append(pred)
        total_label.append(label)
    
    # calculate metrics
    total_pred = np.concatenate(total_pred)
    total_label = np.concatenate(total_label)
    acc = accuracy_score(total_label, total_pred)

    model.train()
    
    # save test results
    if is_save:
        save_dir = model.save_dir
        np.save(os.path.join(save_dir, '{}_pred.npy'.format(phase)), total_pred)
        np.save(os.path.join(save_dir, '{}_label.npy'.format(phase)), total_label)

    return acc


def test(opt):

    trn_dataset, val_dataset, tst_dataset = create_trn_val_tst_dataset(opt)

    logger.info('The number of validation samples = %d' % len(val_dataset))
    logger.info('The number of testing samples = %d' % len(tst_dataset))

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)
    model.cuda()

    # test
    logger.info('Loading model : epoch-%d' % opt.eval_epoch)
    model.load_networks(opt.eval_epoch)
    logger.info('Finish loading model')

    acc= eval(model, val_dataset, tst_dataset, is_save=False, phase='val')
    logger.info('Val result acc %.4f' % (acc))

    acc= eval(model, val_dataset, tst_dataset, is_save=False, phase='test')
    logger.info('Tst result acc %.4f' % (acc))



        

if __name__ == '__main__':

    opt = TestOptions().parse()                        # get training options

    logger_path = os.path.join(opt.log_dir, opt.name)   # get logger path
    if not os.path.exists(logger_path):                 # make sure logger path exists
        os.mkdir(logger_path)

    if opt.suffix:
        suffix = '_'.join([opt.dataset_mode])    # get logger suffix
    else:
        suffix = '_'.join([opt.dataset_mode, opt.suffix])    # get logger suffix
    
    logger = get_logger(logger_path, suffix)                # get logger

    test(opt)