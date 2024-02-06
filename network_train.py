import numpy as np
import torch
import torch.optim
import os
import random 

from methods.backbone_multiBlocks import model_dict
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.waveSAN import waveSAN
from options import parse_args, get_resume_file, load_warmup_state

from test_function import test_bestmodel



def train(base_loader, val_loader,  model, start_epoch, stop_epoch, params, base_loader_B = None):

  # get optimizer and checkpoint path
  optimizer = torch.optim.Adam(model.parameters())
  if not os.path.isdir(params.checkpoint_dir):
    os.makedirs(params.checkpoint_dir)

  # for validation
  max_acc = 0
  total_it = 0

  # start
  for epoch in range(start_epoch,stop_epoch):
    model.train()
    if(params.stage=='pretrain'):
      total_it = model.train_loop(epoch, base_loader,  optimizer, total_it) #model are called by reference, no need to return
    elif(params.stage=='metatrain'):
      total_it = model.train_loop(epoch, base_loader, base_loader_B, optimizer, total_it) #model are called by reference, no need to return
    model.eval()

    acc = model.test_loop( val_loader)
    if acc > max_acc :
      print("best model! save...")
      max_acc = acc
      outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
      torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
    else:
      print("GG! best accuracy {:f}".format(max_acc))

    if ((epoch + 1) % params.save_freq==0) or (epoch==stop_epoch-1):
      outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
      torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

  return model


def record_test_result(params):
  print('hhhhhhh')
  acc_file_path = os.path.join(params.checkpoint_dir, 'acc_fwt.txt')
  acc_file = open(acc_file_path,'w')
  epoch_id = -1
  print('epoch', epoch_id, 'miniImagenet:', 'cub:', 'cars:', 'places:', 'plantae:', file = acc_file)
  name = params.name
  n_shot = params.n_shot
  test_bestmodel(acc_file, name, 'miniImagenet', n_shot, -1)
  test_bestmodel(acc_file, name, 'cub', n_shot, -1)
  test_bestmodel(acc_file, name, 'cars', n_shot, -1)
  test_bestmodel(acc_file, name, 'places', n_shot, -1)
  test_bestmodel(acc_file, name, 'plantae', n_shot, -1)
  acc_file.close()
  return


def record_test_result_bscdfsl(params):
  print('hhhhhhh testing for bscdfsl')
  acc_file_path = os.path.join(params.checkpoint_dir, 'acc_bscdfsl.txt')
  acc_file = open(acc_file_path,'w')
  epoch_id = -1
  print('epoch', epoch_id, 'ChestX:', 'ISIC:', 'EuroSAT:', 'CropDisease', file = acc_file)
  name = params.name
  n_shot = params.n_shot
  method = params.method
  test_bestmodel(acc_file, name, method, 'ChestX', n_shot, -1)
  test_bestmodel(acc_file, name, method, 'ISIC', n_shot, -1)
  test_bestmodel(acc_file, name, method, 'EuroSAT', n_shot, -1)
  test_bestmodel(acc_file, name, method, 'CropDisease', n_shot, -1)

  acc_file.close()
  return


# --- main function ---
if __name__=='__main__':
  # set seed
  seed = 0
  print("set seed = %d" % seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False 

  # parser argument
  params = parse_args('train')
  print('params:', params)

  # output and tensorboard dir
  params.tf_dir = '%s/log/%s'%(params.save_dir, params.name)
  params.checkpoint_dir = '%s/checkpoints/%s'%(params.save_dir, params.name)
  if not os.path.isdir(params.checkpoint_dir):
    os.makedirs(params.checkpoint_dir)

  # dataloader
  print('\n--- prepare dataloader ---')
  print('  train with base source domain {}'.format(params.dataset))
  base_file  = os.path.join(params.data_dir, params.dataset, 'base.json')
  val_file   = os.path.join(params.data_dir, params.dataset, 'val.json')

  # model
  print('\n--- build model ---')
  image_size = 224

  if params.stage == 'pretrain':
    print('  pre-training the feature encoder {}'.format(params.model))
    base_datamgr    = SimpleDataManager(image_size, batch_size=16)
    base_loader     = base_datamgr.get_data_loader( base_file , aug=params.train_aug )
    val_datamgr     = SimpleDataManager(image_size, batch_size=64)
    val_loader      = val_datamgr.get_data_loader(val_file, aug=False)
    model           = BaselineTrain(model_dict[params.model], params.num_classes, tf_path=params.tf_dir)

  elif params.stage == 'metatrain':
    print('meta-train the model with feature encoder {}'.format(params.model))
    #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
    n_query = max(1, int(16* params.test_n_way/params.train_n_way))
    train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot)
    base_datamgr  = SetDataManager(image_size, n_query = n_query,  **train_few_shot_params)
    base_loader_A = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
    base_loader_B = base_datamgr.get_data_loader( base_file , aug = params.train_aug )   
    test_few_shot_params  = dict(n_way = params.test_n_way, n_support = params.n_shot)
    val_datamgr   = SetDataManager(image_size, n_query = n_query, **test_few_shot_params)
    val_loader    = val_datamgr.get_data_loader( val_file, aug = False)
    model         = waveSAN( model_dict[params.model], tf_path=params.tf_dir, **train_few_shot_params)

  model = model.cuda()

  # load model
  start_epoch = params.start_epoch
  stop_epoch = params.stop_epoch
  if params.resume != '':
    resume_file = get_resume_file('%s/checkpoints/%s'%(params.save_dir, params.resume), params.resume_epoch)
    if resume_file is not None:
      tmp = torch.load(resume_file)
      start_epoch = tmp['epoch']+1
      model.load_state_dict(tmp['state'])
      print('  resume the training with at {} epoch (model file {})'.format(start_epoch, params.resume))
  elif params.stage == 'metatrain':
    if params.warmup == 'gg3b0':
      raise Exception('Must provide the pre-trained feature encoder file using --warmup option!')
    state = load_warmup_state('%s/checkpoints/%s'%(params.save_dir, params.warmup))
    model.feature.load_state_dict(state, strict=False)
  
  # training
  print('\n--- start the training ---')
  import time
  start = time.clock()
  if(params.stage=='pretrain'):
    model = train(base_loader, val_loader,  model, start_epoch, stop_epoch, params)
  elif(params.stage=='metatrain'):
    model = train(base_loader_A, val_loader, model, start_epoch, stop_epoch, params, base_loader_B)
    #pass
  end = time.clock()
  print('Running time: %s Seconds: %s Min: %s Min per epoch'%(end-start, (end-start)/60, (end-start)/60/params.stop_epoch))
  # testing fwt's benchmark
  record_test_result(params)
  #testing bscdfsl
  record_test_result_bscdfsl(params)

