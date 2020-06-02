import torch
import os
import torch.nn.functional as F
from .base_model import BaseModel
from modules.bilstm_encoder import BiLSTMEncoder
from modules.classifier import MaxPoolFc
from modules.tools import init_net
from modules.classifier import SimpleClassifier
import torch.nn as nn

class ExampleModel(BaseModel):
    
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # parser.set_defaults(no_dropout=True)
        parser.add_argument('--output_dim', type=int, default=3, help='output dim')
        parser.add_argument('--hidden_size', type=int, default=100, help='hidden state of lstm model')
        parser.add_argument('--dropout_rate', type=float, default=0.2, help='rate of dropout')
        parser.add_argument('--embedding_size', type=int, default=34, help='seq input size')
        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        self.loss_names = ['CE']
        # build model

        self.model_names = ['A', 'C'] 
        self.netA = BiLSTMEncoder(opt.embedding_size, opt.hidden_size, opt.dropout_rate)
                            
        self.netC = MaxPoolFc(opt.hidden_size*2, opt.output_dim)
       

        if self.isTrain:
            self.criterion = torch.nn.CrossEntropyLoss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            parameters = [{'params': getattr(self, 'net'+ net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
           
        # modify save_dir
        self.save_dir = os.path.join(self.save_dir)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
    
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        """
        self.seq = input['seq'].float().cuda()
        self.length = input['length'].long().cuda()
        self.label = input['label'].long().cuda()
        self.mask = input['mask'].float().cuda()
        # self.label = self.label.argmax(dim=1)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.sequence, self.final_hidden = self.netA(self.seq, self.length)
        self.logits = self.netC(self.sequence)
        self.pred = F.softmax(self.logits, dim=1)
   
    def backward(self):
        """Calculate the loss for back propagation"""
        torch.autograd.set_detect_anomaly(True)
        self.loss_CE = self.criterion(self.logits, self.label)
        self.loss_CE.backward()
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net'+model).parameters(), 1.)

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()   
        # backward
        self.optimizer.zero_grad()  
        self.backward()            
        self.optimizer.step() 
