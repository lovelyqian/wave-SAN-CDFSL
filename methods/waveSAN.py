import torch
import torch.nn as nn
import numpy as np
import random
from methods.meta_template import MetaTemplate
from pytorch_wavelets import DWTForward, DWTInverse
from methods.gnn import GNN_nl
from methods import backbone_multiBlocks
from methods.adain import adaptive_instance_normalization_us  as adain


class waveSAN(MetaTemplate):
  maml=False
  def __init__(self, model_func,  n_way, n_support, tf_path=None):
    super(waveSAN, self).__init__(model_func, n_way, n_support, tf_path=tf_path)

    # loss function
    self.loss_fn = nn.CrossEntropyLoss()

    # metric function
    self.fc = nn.Sequential(nn.Linear(self.feat_dim, 128), nn.BatchNorm1d(128, track_running_stats=False)) if not self.maml else nn.Sequential(backbone.Linear_fw(self.feat_dim, 128), backbone.BatchNorm1d_fw(128, track_running_stats=False))
    self.gnn = GNN_nl(128 + self.n_way, 96, self.n_way)
    self.method = 'GnnNet'

    # fix label for training the metric function   1*nw(1 + ns)*nw
    support_label = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).unsqueeze(1)
    support_label = torch.zeros(self.n_way*self.n_support, self.n_way).scatter(1, support_label, 1).view(self.n_way, self.n_support, self.n_way)
    support_label = torch.cat([support_label, torch.zeros(self.n_way, 1, n_way)], dim=1)
    self.support_label = support_label.view(1, -1, self.n_way)

  def cuda(self):
    self.feature.cuda()
    self.fc.cuda()
    self.gnn.cuda()
    self.support_label = self.support_label.cuda()
    return self

  def set_forward(self,x,is_feature=False):
    x = x.cuda()

    if is_feature:
      # reshape the feature tensor: n_way * n_s + 15 * f
      assert(x.size(1) == self.n_support + 15)
      z = self.fc(x.view(-1, *x.size()[2:]))
      z = z.view(self.n_way, -1, z.size(1))
    else:
      # get feature using encoder
      x = x.view(-1, *x.size()[2:])
      z = self.fc(self.feature(x))
      z = z.view(self.n_way, -1, z.size(1))
  
    torch.randn(1, 1, z.size()[2])  #

    # stack the feature for metric function: n_way * n_s + n_q * f -> n_q * [1 * n_way(n_s + 1) * f]
    z_stack = [torch.cat([z[:, :self.n_support], z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1, z.size(2)) for i in range(self.n_query)]
    assert(z_stack[0].size(1) == self.n_way*(self.n_support + 1))
    scores = self.forward_gnn(z_stack)
    return scores


  def wavelet_DWT_and_styleswap_and_IWT(self, A_fea, B_fea, J):
      # wavelet: DWT
      xfm = DWTForward(J=J, wave='haar', mode='zero').cuda()
      A_fea_Yl, A_fea_Yh = xfm(A_fea)
      B_fea_Yl, B_fea_Yh = xfm(B_fea) 

      # styleswap the A_yl and B_yl
      A_fea_Yl_styleswap = adain(A_fea_Yl, B_fea_Yl)
      B_fea_Yl_styleswap = adain(B_fea_Yl, A_fea_Yl)

      # wavelet: IWT1
      ifm = DWTInverse(wave='haar', mode='zero').cuda()
      A_fea_styleswap = ifm((A_fea_Yl_styleswap, A_fea_Yh))
      B_fea_styleswap = ifm((B_fea_Yl_styleswap, B_fea_Yh))
      return A_fea_styleswap, B_fea_styleswap



  def set_forward_wavelet_styleswap(self,A,B,is_feature=False):
    A = A.cuda()
    B = B.cuda()

    if is_feature:
      assert(x.size(1) == self.n_support + 15)
      z = self.fc(x.view(-1, *x.size()[2:]))
      z = z.view(self.n_way, -1, z.size(1))

    else:
      swap_type = 'block123'
      wavelet_J_list = [1, 1, 1]  

      dropout_k = 4
      dropout_type = 'None'  #without Cdrop
      Cdrop_type='batch_level'

      # get feature using encoder
      A = A.view(-1, *A.size()[2:])
      B = B.view(-1, *B.size()[2:])

      # forward block1
      A_block1 = self.feature.forward_block1(A)
      B_block1 = self.feature.forward_block1(B)

      # wavelet
      A_block1_styleswap, B_block1_styleswap = self.wavelet_DWT_and_styleswap_and_IWT(A_block1, B_block1, wavelet_J_list[0]) 
      A_block2 = self.feature.forward_block2(A_block1_styleswap)
      B_block2 = self.feature.forward_block2(B_block1_styleswap)
      A_block2_styleswap, B_block2_styleswap = self.wavelet_DWT_and_styleswap_and_IWT(A_block2, B_block2, wavelet_J_list[1])

      A_block3 = self.feature.forward_block3(A_block2_styleswap)
      B_block3 = self.feature.forward_block3(B_block2_styleswap)
      A_block3_styleswap, B_block3_styleswap = self.wavelet_DWT_and_styleswap_and_IWT(A_block3, B_block3, wavelet_J_list[2])
 
      A_block4 = self.feature.forward_block4(A_block3_styleswap)
      B_block4 = self.feature.forward_block4(B_block3_styleswap)
 
      A_fea = self.feature.forward_rest(A_block4)
      B_fea = self.feature.forward_rest(B_block4)

      A_z = self.fc(A_fea)
      B_z = self.fc(B_fea)
   
      A_z = A_z.view(self.n_way, -1, A_z.size(1))
      B_z = B_z.view(self.n_way, -1, B_z.size(1))

    # stack the feature for metric function: n_way * n_s + n_q * f -> n_q * [1 * n_way(n_s + 1) * f]
    A_z_stack = [torch.cat([A_z[:, :self.n_support], A_z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1, A_z.size(2)) for i in range(self.n_query)]
    B_z_stack = [torch.cat([B_z[:, :self.n_support], B_z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1, B_z.size(2)) for i in range(self.n_query)]
    assert(A_z_stack[0].size(1) == self.n_way*(self.n_support + 1))
    assert(B_z_stack[0].size(1) == self.n_way*(self.n_support + 1))
    scores_A = self.forward_gnn(A_z_stack)
    scores_B = self.forward_gnn(B_z_stack)
    return scores_A, scores_B


  def forward_gnn(self, zs):
    # gnn inp: n_q * n_way(n_s + 1) * f
    nodes = torch.cat([torch.cat([z, self.support_label], dim=2) for z in zs], dim=0)
    scores = self.gnn(nodes)

    # n_q * n_way(n_s + 1) * n_way -> (n_way * n_q) * n_way
    scores = scores.view(self.n_query, self.n_way, self.n_support + 1, self.n_way)[:, :, -1].permute(1, 0, 2).contiguous().view(-1, self.n_way)
    return scores

  def set_forward_loss(self, x):
    y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query))
    y_query = y_query.cuda()
    scores = self.set_forward(x)
    loss = self.loss_fn(scores, y_query)
    return scores, loss


  def set_forward_loss_wavelet_styleswap(self, A, B):
    y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query))
    y_query = y_query.cuda()
    scores_A, scores_B = self.set_forward_wavelet_styleswap(A,B)
    loss_A = self.loss_fn(scores_A, y_query)
    loss_B = self.loss_fn(scores_B, y_query) 
    return scores_A, loss_A, scores_B, loss_B

