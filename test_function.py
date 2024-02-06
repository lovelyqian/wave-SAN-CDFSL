import torch
import os
import h5py

from methods.backbone_multiBlocks import model_dict
from data.datamgr import SimpleDataManager
from options import parse_args, get_best_file, get_assigned_file
from data import ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot, Chest_few_shot
from methods.waveSAN import waveSAN
import data.feature_loader as feat_loader
import random
import numpy as np

# extract and save image features
def save_features(model, data_loader, featurefile):
  f = h5py.File(featurefile, 'w')
  max_count = len(data_loader)*data_loader.batch_size
  all_labels = f.create_dataset('all_labels',(max_count,), dtype='i')
  all_feats=None
  count=0
  for i, (x,y) in enumerate(data_loader):
    if (i % 10) == 0:
      print('    {:d}/{:d}'.format(i, len(data_loader)))
    x = x.cuda()
    feats = model(x)
    if all_feats is None:
      all_feats = f.create_dataset('all_feats', [max_count] + list( feats.size()[1:]) , dtype='f')
    all_feats[count:count+feats.size(0)] = feats.data.cpu().numpy()
    all_labels[count:count+feats.size(0)] = y.cpu().numpy()
    count = count + feats.size(0)

  count_var = f.create_dataset('count', (1,), dtype='i')
  count_var[0] = count
  f.close()

# evaluate using features
def feature_evaluation(cl_data_file, model, n_way = 5, n_support = 5, n_query = 15):
  class_list = cl_data_file.keys()
  select_class = random.sample(class_list,n_way)
  z_all  = []
  for cl in select_class:
    img_feat = cl_data_file[cl]
    perm_ids = np.random.permutation(len(img_feat)).tolist()
    z_all.append( [ np.squeeze( img_feat[perm_ids[i]]) for i in range(n_support+n_query) ] )
  z_all = torch.from_numpy(np.array(z_all) )

  model.n_query = n_query
  scores  = model.set_forward(z_all, is_feature = True)
  pred = scores.data.cpu().numpy().argmax(axis = 1)
  y = np.repeat(range( n_way ), n_query )
  acc = np.mean(pred == y)*100
  return acc



def test_bestmodel(acc_file, name,dataset,n_shot, save_epoch=-1):
  # parse argument
  params = parse_args('test')
  params.n_shot = n_shot
  params.dataset = dataset
  params.name = name
  params.save_epoch = save_epoch  #-1 = best
 
  print('Testing! {} shots on {} dataset with {} epochs of {}'.format(params.n_shot, params.dataset, params.save_epoch, params.name))
  remove_featurefile = True

  print('\nStage 1: saving features')
  # dataset
  print('  build dataset')
  image_size = 224

  split = params.split 
  if params.dataset in ["miniImagenet", "cub", "cars", "places", "plantae"]:
    loadfile = os.path.join(params.data_dir, params.dataset, split + '.json')
    datamgr         = SimpleDataManager(image_size, batch_size = 64)
    data_loader      = datamgr.get_data_loader(loadfile, aug = False)

  else:
    if params.dataset in ["ISIC"]:
        datamgr         = ISIC_few_shot.SimpleDataManager(image_size, batch_size = 64)
        data_loader     = datamgr.get_data_loader(aug = False )

    elif params.dataset in ["EuroSAT"]:

        datamgr         = EuroSAT_few_shot.SimpleDataManager(image_size, batch_size = 64)
        data_loader     = datamgr.get_data_loader(aug = False )

    elif params.dataset in ["CropDisease"]:
        datamgr         = CropDisease_few_shot.SimpleDataManager(image_size, batch_size = 64)
        data_loader     = datamgr.get_data_loader(aug = False )

    elif params.dataset in ["ChestX"]:
        datamgr         = Chest_few_shot.SimpleDataManager(image_size, batch_size = 64)
        data_loader     = datamgr.get_data_loader(aug = False )

  print('  build feature encoder')
  # feature encoder
  checkpoint_dir = '%s/checkpoints/%s'%(params.save_dir, params.name)
  if params.save_epoch != -1:
    modelfile   = get_assigned_file(checkpoint_dir,params.save_epoch)
  else:
    modelfile   = get_best_file(checkpoint_dir)
  model = model_dict[params.model]()
  model = model.cuda()
  tmp = torch.load(modelfile)
  try:
    state = tmp['state']
  except KeyError:
    state = tmp['model_state']
  except:
    raise
  state_keys = list(state.keys())
  print('state_keys:', state_keys, len(state_keys))
  for i, key in enumerate(state_keys):
    if "feature." in key and not 'gamma' in key and not 'beta' in key:
      newkey = key.replace("feature.","")
      state[newkey] = state.pop(key)
    else:
      state.pop(key)
  print('state keys:', list(state.keys()), len(list(state.keys())))

  model.load_state_dict(state)
  model.eval()

  # save feature file
  print('  extract and save features...')
  if params.save_epoch != -1:
    featurefile = os.path.join( checkpoint_dir.replace("checkpoints","features"), split + "_" + str(params.save_epoch)+ ".hdf5")
  else:
    featurefile = os.path.join( checkpoint_dir.replace("checkpoints","features"), split + ".hdf5")
  dirname = os.path.dirname(featurefile)
  if not os.path.isdir(dirname):
    os.makedirs(dirname)
  save_features(model, data_loader, featurefile)

  print('\nStage 2: evaluate')
  acc_all = []
  iter_num = 1000
  few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot)
  # model
  print('  build metric-based model')
  model = waveSAN( model_dict[params.model], **few_shot_params)
  model = model.cuda()
  model.eval()

  # load model
  checkpoint_dir = '%s/checkpoints/%s'%(params.save_dir, params.name)
  if params.save_epoch != -1:
    modelfile = get_assigned_file(checkpoint_dir, params.save_epoch)
  else:
    modelfile = get_best_file(checkpoint_dir)
  if modelfile is not None:
    tmp = torch.load(modelfile)
    try:
      model.load_state_dict(tmp['state'])
    except RuntimeError:
      print('warning! RuntimeError when load_state_dict()!')
      model.load_state_dict(tmp['state'], strict=False)
    except KeyError:
      for k in tmp['model_state']:   ##### revise latter
        if 'running' in k:
          tmp['model_state'][k] = tmp['model_state'][k].squeeze()
      model.load_state_dict(tmp['model_state'], strict=False)
    except:
      raise

  # load feature file
  print('  load saved feature file')
  cl_data_file = feat_loader.init_loader(featurefile)

  # start evaluate
  print('  evaluate')
  for i in range(iter_num):
    acc = feature_evaluation(cl_data_file, model, n_query=15, **few_shot_params)
    acc_all.append(acc)

  # statics
  print('  get statics')
  acc_all = np.asarray(acc_all)
  acc_mean = np.mean(acc_all)
  acc_std = np.std(acc_all)
  print('  %d test iterations: Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
  print('  %d test iterations: Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num)), file = acc_file)

  # remove feature files [optional]
  if remove_featurefile:
    os.remove(featurefile)


if __name__ == '__main__':
  params = parse_args('test')
  print('params:', params)
  params.checkpoint_dir = '%s/checkpoints/%s'%(params.save_dir, params.name)
  acc_file_path = os.path.join(params.checkpoint_dir, 'acc_tmp.txt')
  acc_file = open(acc_file_path,'w')
  epoch_id = -1
  name = params.name
  dataset = params.dataset
  n_shot = params.n_shot
  test_bestmodel(acc_file, name, dataset, n_shot, -1)
  acc_file.close()
