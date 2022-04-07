from ignite.engine import Engine, Events
from model.Prediction.utils import lstToCuda,maskedNLL,maskedMSE,maskedNLLTest, maskedMSETest
import math
import torch
from ignite.contrib.handlers import ProgressBar
import os
import numpy as np
from tensorboardX import SummaryWriter

class TrajPredEngine:

    def __init__(self, net, optim, train_loader, val_loader, args):
        self.net = net
        self.args = args
        self.pretrainEpochs = args["pretrainEpochs"]
        self.trainEpochs = args["trainEpochs"]
        self.optim = optim
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cuda = args['cuda']
        self.device = args['device']
        self.dsId = self.args['dsId']
        self.n_iterations = max(len(train_loader), len(train_loader) / args["batch_size"])

        ## training metrics to keep track of, consider making a metrics class
        # remember to 0 these out
        self.avg_trn_loss = 0

        self.metrics = {"Avg train loss": 0, "Avg val loss": 0 }
        ## validation metrics
        self.avg_val_loss = 0
        self.val_batch_count = 1

        # only if using maneuvers
        self.avg_lat_acc = 0
        self.avg_lon_acc = 0

        self.trainer = None
        self.evaluator = None

        self.makeTrainer()

        self.save_name = args['name']

        # testing stuff wow need 2 clean this so bad

        self.lossVals = torch.zeros(self.args['out_length']).cuda(self.device) if self.cuda else torch.zeros(self.args['out_length'])
        self.counts = torch.zeros(self.args['out_length']).cuda(self.device) if self.cuda else torch.zeros(self.args['out_length'])
        self.lastTestLoss = 0
        
        self.writer = None
        self.log_dir = args['log_dir']
        self.tensorboard = args['tensorboard']

    def netPred(self, batch):
        raise NotImplementedError

    def saveModel(self, engine):

        os.makedirs(self.args['modelLoc'], exist_ok=True)
        name = os.path.join(self.args['modelLoc'], self.args['name'])
        torch.save(self.net.state_dict(), name)
        print("Model saved {}.".format(name))

    def train_a_batch(self, engine, batch):

        self.net.train_flag = True
        epoch = engine.state.epoch

        _, _, _, _, _, _, _, fut, op_mask = batch

        fut_pred = self.netPred(batch)
        
        if self.cuda:
            fut = fut.cuda(self.device)
            op_mask = op_mask.cuda(self.device)

        if epoch < self.pretrainEpochs:
            if self.args["pretrain_loss"] == 'MSE':
                l = maskedMSE(fut_pred, fut, op_mask, device=self.device)
            elif self.args['pretrain_loss'] == 'NLL':
                l = maskedNLL(fut_pred, fut, op_mask, device=self.device)
            else:
                l = maskedMSE(fut_pred, fut, op_mask, device=self.device)
        else:
            if self.args["train_loss"] == 'MSE':
                l = maskedMSE(fut_pred, fut, op_mask, device=self.device)
            elif self.args['train_loss'] == 'NLL':
                l = maskedNLL(fut_pred, fut, op_mask, device=self.device)
            else:
                l = maskedNLL(fut_pred, fut, op_mask, device=self.device)

        # if self.args['nll_only']:
        #     l = maskedNLL(fut_pred, fut, op_mask)
        # else:
        #     if epoch < self.pretrainEpochs:
        #         l = maskedMSE(fut_pred, fut, op_mask)
        #     else:
        #         l = maskedNLL(fut_pred, fut, op_mask)

        # Backprop and update weights
#        if l.item() != l.item():
#            print(l.item())
#            exit(1)
#            return 1
        self.optim.zero_grad()
        l.backward()
        self.optim.step()

        # Track average train loss:
        self.avg_trn_loss += l.item()
        self.metrics["Avg train loss"] += l.item() / 100.0
           
        if self.writer:
            self.writer.add_scalar("{}epoch/trainingloss".format(engine.state.epoch), l.item() , engine.state.iteration)
        
        return l.item()

    def eval_a_batch(self, engine, batch):
        self.net.train_flag = False

        epoch = engine.state.epoch

        _, _, _, _, _, _, _, fut, op_mask = batch
        fut_pred = self.netPred(batch)
        if self.cuda:
            fut = fut.cuda(self.device)
            op_mask = op_mask.cuda(self.device)

        # Forward pass

        if epoch < self.pretrainEpochs:
            if self.args["pretrain_loss"] == 'MSE':
                l = maskedMSE(fut_pred, fut, op_mask, device=self.device)
            elif self.args['pretrain_loss'] == 'NLL':
                l = maskedNLL(fut_pred, fut, op_mask, device=self.device)
            else:
                l = maskedMSE(fut_pred, fut, op_mask, device=self.device)
        else:
            if self.args["train_loss"] == 'MSE':
                l = maskedMSE(fut_pred, fut, op_mask, device=self.device)
            elif self.args['train_loss'] == 'NLL':
                l = maskedNLL(fut_pred, fut, op_mask, device=self.device)
            else:
                l = maskedNLL(fut_pred, fut, op_mask, device=self.device)


        # if self.args['nll_only']:
        #     l = maskedNLL(fut_pred, fut, op_mask)
        # else:
        #     if epoch_num < pretrainEpochs:
        #         l = maskedMSE(fut_pred, fut, op_mask)
        #     else:
        #         l = maskedNLL(fut_pred, fut, op_mask)

        self.avg_val_loss += l.item()
        self.metrics["Avg val loss"] += l.item()/ (self.val_batch_count * 100.0)
        self.val_batch_count += 1

        return fut_pred, fut

    def validate(self, engine):
        self.evaluator.run(self.val_loader)
        max_epochs =self.args["pretrainEpochs"] + self.args["trainEpochs"]

        # if not self.eval_only:
        print("{}/{} Epochs in dataset{}".format(engine.state.epoch, max_epochs, self.dsId))
        # print(max((engine.state.epoch / max_epochs) * 100,1))
        print("EPOCH {}: Train loss: {}  Val loss: {}\n".format(engine.state.epoch, self.metrics["Avg train loss"], self.metrics["Avg val loss"]))
        # else:
        #     print("EPOCH {}: Test loss: {}\n".format(engine.state.epoch, self.metrics["Avg val loss"]))
        
        if self.writer:
            self.writer.add_scalar("training_avg_loss", self.metrics['Avg train loss'], engine.state.epoch)
            self.writer.add_scalar("validating_avg_loss", self.metrics['Avg val loss'], engine.state.epoch)

        self.metrics["Avg train loss"] = 0
        self.metrics["Avg val loss"] = 0

    def zeroMetrics(self, engine):
        self.val_batch_count = 1
        self.metrics["Avg val loss"] = 0 

    def zeroTrainLoss(self, engine):
        self.metrics["Avg train loss"] = 0

    def zeroValLoss(self, engine):
        self.metrics["Avg val loss"] = 0

    def makeTrainer(self):
        self.trainer = Engine(self.train_a_batch)
        self.evaluator = Engine(self.eval_a_batch)

        pbar = ProgressBar(persist=True, postfix=self.metrics)
        pbar.attach(self.trainer)
        pbar.attach(self.evaluator)

        ## attach hooks 
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.validate)
        self.trainer.add_event_handler(Events.ITERATION_COMPLETED, self.zeroMetrics)
        self.trainer.add_event_handler(Events.COMPLETED, self.saveModel)
        # zero out metrics for next epoch


    def create_summary_writer(self, model, data_loader, log_dir):
        writer = SummaryWriter(logdir=log_dir)
        data_loader_iter = iter(data_loader)
        b = next(data_loader_iter)
        b = tuple(x.cuda(self.device) for x in b)
        try:
            writer.add_graph(model, b[:7])
        except Exception as e:
            print("Failed to save model graph: {}".format(e))
        return writer

    def start(self):
        max_epochs =self.args["pretrainEpochs"] + self.args["trainEpochs"]

        if self.tensorboard:
            self.writer = self.create_summary_writer(self.net, self.train_loader, self.log_dir)


#        @self.trainer.on(Events.ITERATION_COMPLETED)
#        def log_training_loss(engine):
#            iter = (engine.state.iteration - 1) % len(self.train_loader) + 1
#            if iter % 10 == 0:
#                self.writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)

        # if not self.eval_only:
        self.trainer.run(self.train_loader, max_epochs=max_epochs)
        # else:
            # self.trainer.run(self.train_loader, max_epochs=1)

        if self.tensorboard:
            self.writer.close()


    def test_a_batch(self, engine, batch):
        _, _, _, _, _, _, _, fut, op_mask, _, _, _, _ = batch

        # Initialize Variables
        if self.cuda:
            fut = fut.cuda(self.device)
            op_mask = op_mask.cuda(self.device)

        if self.args["train_loss"] == 'NLL':
            # Forward pass
            if self.args['use_maneuvers']:
                fut_pred, lat_pred, lon_pred = self.netPred(batch)
                l,c = maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask, device=self.device, cuda=self.args.cuda)
            else:
                fut_pred = self.netPred(batch)
                l, c = maskedNLLTest(fut_pred, 0, 0, fut, op_mask, device=self.device, use_maneuvers=False, cuda=self.cuda)
        else:
            # Forward pass
            if self.args['use_maneuvers']:
                fut_pred, lat_pred, lon_pred = self.netPred(batch)
                fut_pred_max = torch.zeros_like(fut_pred[0])
                for k in range(lat_pred.shape[0]):
                    lat_man = torch.argmax(lat_pred[k, :]).detach()
                    lon_man = torch.argmax(lon_pred[k, :]).detach()
                    indx = lon_man*3 + lat_man
                    fut_pred_max[:,k,:] = fut_pred[indx][:,k,:]
                l, c = maskedMSETest(fut_pred_max, fut, op_mask, device=self.device)
            else:
                fut_pred = self.netPred(batch)            
                l, c = maskedMSETest(fut_pred, fut, op_mask, device=self.device)


        self.lossVals +=l.detach()
        self.lastTestLoss = l.detach()
        self.counts += c.detach()



    def eval(self, test_loader):


        self.test_batch_size = len(test_loader)
        tester = Engine(self.test_a_batch)

        pbar = ProgressBar(persist=True, postfix=self.metrics)
        pbar.attach(tester)
        print('evaluating on dataset{}...'.format(self.dsId))
        tester.run(test_loader)

        if(self.args["train_loss"]) == "NLL" :
            nll_loss = self.lossVals / self.counts
            nll_loss[nll_loss != nll_loss] = 0
            print("Last Test loss: " + str(self.lastTestLoss.mean().item()))
            print("Avg Test loss: " + str(nll_loss.mean().item()))
        else:
            rmse = torch.pow(self.lossVals / self.counts, 0.5) * .3048 # converting from feet to meters
            rmse[torch.isnan(rmse)] = 0
            # self.lastTestLoss = torch.pow(self.lastTestLoss, 0.5) * .3048
            # print(self.lastTestLoss)
            seq_loss = rmse.tolist()
            seq_loss = [x for x in seq_loss if x != 0]
            print(rmse)
            print("Last Test loss: " + str(seq_loss[-1]))
            print("Avg Test loss: " + str(rmse.mean().item()))
