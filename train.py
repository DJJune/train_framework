

from opt.train_opts import TrainOptions
from utils.logger import get_logger
from tensorboardX import SummaryWriter
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

def clean_chekpoints(expr_name, store_epoch):
    root = os.path.join('checkpoints', expr_name)
    for checkpoint in os.listdir(root):
        if not checkpoint.startswith(str(store_epoch)+'_') and checkpoint.endswith('pth'):
            os.remove(os.path.join(root, checkpoint))

def train(opt):

    trn_dataset, val_dataset, tst_dataset = create_trn_val_tst_dataset(opt)
    dataset_size = len(trn_dataset)
    logger.info('The number of training samples = %d' % dataset_size)
    writer = SummaryWriter()

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)
    model.cuda()

    best_eval_acc = 0              # record the best eval UAR
    total_iters = 0                # the total number of training iterations
    best_eval_epoch = -1           # record the best eval epoch

    for epoch in range(1, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(trn_dataset):  # inner loop within one epoch
            iter_start_time = time.time()   # timer for computation per iteration
            total_iters += 1                # opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)           # unpack data from dataset and apply preprocessing
            model.optimize_parameters(epoch)   # calculate loss functions, get gradients, update network weights
                
            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                logger.info('Cur epoch {}'.format(epoch) + ' loss ' + 
                        ' '.join(map(lambda x:'{}:{{{}:.4f}}'.format(x, x), model.loss_names)).format(**losses))

                writer.add_scalars('training_loss', dict(losses), total_iters)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                logger.info('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            logger.info('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        logger.info('End of training epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        
        model.update_learning_rate(logger)                     # update learning rates at the end of every epoch.
        
        acc = eval(model, val_dataset, tst_dataset)

        logger.info('Val result of epoch %d / %d acc %.4f ' % (epoch, opt.niter + opt.niter_decay, acc))

        if acc > best_eval_acc:
            best_eval_epoch = epoch
            best_eval_acc = acc
    
    writer.close()

    # print best eval result
    logger.info('Best eval epoch %d found with acc %f' % (best_eval_epoch, best_eval_acc))

    # test
    logger.info('Loading best model found on val set: epoch-%d' % best_eval_epoch)
    model.load_networks(best_eval_epoch)

    acc= eval(model, val_dataset, tst_dataset, is_save=True, phase='test')
    logger.info('Tst result acc %.4f' % (acc))

    clean_chekpoints(opt.name, best_eval_epoch)



        

if __name__ == '__main__':

    opt = TrainOptions().parse()                        # get training options

    logger_path = os.path.join(opt.log_dir, opt.name)   # get logger path
    if not os.path.exists(logger_path):                 # make sure logger path exists
        os.mkdir(logger_path)

    if opt.suffix:
        suffix = '_'.join([opt.dataset_mode])    # get logger suffix
    else:
        suffix = '_'.join([opt.dataset_mode, opt.suffix])    # get logger suffix
    
    logger = get_logger(logger_path, suffix)                # get logger

    train(opt)