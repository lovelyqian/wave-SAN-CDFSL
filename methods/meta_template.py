import torch.nn as nn
import torch
import numpy as np
from abc import abstractmethod
from tensorboardX import SummaryWriter
import torch.nn.functional as F


def consistency_loss(scoresM1, scoresM2, type='euclidean'):
    if(type=='euclidean'):
       avg_pro = (scoresM1 + scoresM2)/2.0
       matrix1 = torch.sqrt(torch.sum((scoresM1 - avg_pro)**2,dim=1))
       matrix2 = torch.sqrt(torch.sum((scoresM2 - avg_pro)**2,dim=1))
       dis1 = torch.mean(matrix1)
       dis2 = torch.mean(matrix2)
       dis = (dis1+dis2)/2.0
    elif(type=='KL'):
       avg_pro = (scoresM1 + scoresM2)/2.0
       matrix1 = torch.sum( F.softmax(scoresM1,dim=-1) * (F.log_softmax(scoresM1, dim=-1) - F.log_softmax(avg_pro, dim=-1)), 1)
       matrix2 = torch.sum( F.softmax(scoresM2,dim=-1) * (F.log_softmax(scoresM2, dim=-1) - F.log_softmax(avg_pro,dim=-1)), 1)
       dis1 = torch.mean(matrix1)
       dis2 = torch.mean(matrix2)
       dis = (dis1+dis2)/2.0
    else:
       return
    return dis

 
class MetaTemplate(nn.Module):
  def __init__(self, model_func, n_way, n_support, flatten=True, leakyrelu=False, tf_path=None, change_way=True):
    super(MetaTemplate, self).__init__()
    self.n_way      = n_way
    self.n_support  = n_support
    self.n_query    = -1 #(change depends on input)
    self.feature    = model_func(flatten=flatten, leakyrelu=leakyrelu)
    self.feat_dim   = self.feature.final_feat_dim
    self.change_way = change_way  #some methods allow different_way classification during training and test
    self.tf_writer = SummaryWriter(log_dir=tf_path) if tf_path is not None else None

  @abstractmethod
  def set_forward(self,x,is_feature):
    pass

  @abstractmethod
  def set_forward_loss(self, x):
    pass

  def forward(self,x):
    out  = self.feature.forward(x)
    return out

  def parse_feature(self,x,is_feature):
    x = x.cuda()
    if is_feature:
      z_all = x
    else:
      x           = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:])
      z_all       = self.feature.forward(x)
      z_all       = z_all.view( self.n_way, self.n_support + self.n_query, -1)
    z_support   = z_all[:, :self.n_support]
    z_query     = z_all[:, self.n_support:]

    return z_support, z_query

  def correct(self, x):
    scores, loss = self.set_forward_loss(x)
    y_query = np.repeat(range( self.n_way ), self.n_query )

    topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
    topk_ind = topk_labels.cpu().numpy()
    top1_correct = np.sum(topk_ind[:,0] == y_query)
    return float(top1_correct), len(y_query), loss.item()*len(y_query)


  def train_loop(self, epoch, train_loader_A, train_loader_B, optimizer, total_it):
    print_freq = len(train_loader_A) // 10
    avg_loss=0
    for ((i, (x_A1, _ )),(j,(x_B1, _))) in zip(enumerate(train_loader_A), enumerate(train_loader_B)):
      self.n_query = x_A1.size(1) - self.n_support
      if self.change_way:
        self.n_way  = x_A1.size(0)
      optimizer.zero_grad()

      # initial forward
      scores_A1, loss_A1 = self.set_forward_loss(x_A1)
      scores_B1, loss_B1 = self.set_forward_loss(x_B1)
      loss_FSL_init = 0.5*loss_A1 + 0.5*loss_B1

      # stylewarp FSL loss
      scores_A2, loss_A2, scores_B2, loss_B2 = self.set_forward_loss_wavelet_styleswap(x_A1, x_B1)
      loss_FSL_styleswap = 0.5*loss_A2 + 0.5*loss_B2

      # consistency loss between initial and styleswap
      consistency_type = 'KL'
      loss_KL_A_styleswap = consistency_loss(scores_A1, scores_A2, consistency_type)
      loss_KL_B_styleswap = consistency_loss(scores_B1, scores_B2, consistency_type)
      loss_KL_styleswap = 0.5*loss_KL_A_styleswap + 0.5*loss_KL_B_styleswap

      # FINAL LOSS
      loss = 0.2*loss_FSL_init + 0.8*loss_FSL_styleswap + loss_KL_styleswap 
      loss.backward()
      optimizer.step()
      avg_loss = avg_loss+loss.item()

      if (i + 1) % print_freq==0:
        print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i + 1, len(train_loader_A), avg_loss/float(i+1)))
      if (total_it + 1) % 10 == 0 and self.tf_writer is not None:
        self.tf_writer.add_scalar('loss_FSL_init:', loss_FSL_init.item(), total_it +1)
        self.tf_writer.add_scalar('loss_FSL_styleswap:', loss_FSL_styleswap.item(), total_it +1)
        self.tf_writer.add_scalar('loss_KL_styleswap:', loss_KL_styleswap.item(), total_it +1)
        self.tf_writer.add_scalar('total_loss:', loss.item(), total_it +1)
      total_it += 1
    return total_it

  def test_loop(self, test_loader, record = None):
    loss = 0.
    count = 0
    acc_all = []

    iter_num = len(test_loader)
    for i, (x,_) in enumerate(test_loader):
      self.n_query = x.size(1) - self.n_support
      if self.change_way:
        self.n_way  = x.size(0)
      correct_this, count_this, loss_this = self.correct(x)
      acc_all.append(correct_this/ count_this*100  )
      loss += loss_this
      count += count_this

    acc_all  = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std  = np.std(acc_all)
    print('--- %d Loss = %.6f ---' %(iter_num,  loss/count))
    print('--- %d Test Acc = %4.2f%% +- %4.2f%% ---' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

    return acc_mean
