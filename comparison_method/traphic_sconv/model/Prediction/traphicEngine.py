from model.Prediction.trajPredEngine import TrajPredEngine
import torch
import datetime
import numpy as np
class TraphicEngine(TrajPredEngine):
    """
    Implementation of abstractEngine for traphic
    TODO:maneuver metrics, too much duplicate code with socialEngine
    """

    def __init__(self, net, optim, train_loader, val_loader, args):
        super().__init__(net, optim, train_loader, val_loader, args)
        self.save_name = "traphic"

    def netPred(self, batch):
        hist, upp_nbrs, nbrs, upp_mask, mask, lat_enc, lon_enc, fut, op_mask, b, d, v, f = batch

        if self.args['cuda']:
            hist = hist.cuda(self.device)
            nbrs = nbrs.cuda(self.device)
            upp_nbrs = upp_nbrs.cuda(self.device)
            mask = mask.cuda(self.device)
            upp_mask = upp_mask.cuda(self.device)
            lat_enc = lat_enc.cuda(self.device)
            lon_enc = lon_enc.cuda(self.device)
            fut = fut.cuda(self.device)
            op_mask = op_mask.cuda(self.device)

        fut_pred  = self.net(hist, upp_nbrs, nbrs, upp_mask, mask, lat_enc, lon_enc)

        return fut_pred

