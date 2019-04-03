import os
import tables
from unet.model import unet_model_3d
from unet import generator

config={}
config['is_on_cheaha']=False
config['h5_dir']='$USER_SCRATCH' if config['is_on_cheaha'] else '.'
config['h5_file']='brats_data.h5'
config['train_percentage']=0.7



img_data=tables.open_file(os.path.join(config['h5_dir'],config['h5_file']),'r')
data_num=img_data.root.data.shape[0]
partition={}
partition['train']=list(range(int(data_num*config['train_percentage'])))
partition['test']=list(range(int(data_num*config['train_percentage']),data_num))

print("Data shape: ",img_data.root.data.shape)
print("Truth shape: ",img_data.root.truth.shape)
gen=generator.batch_generator(img_data=img_data,idx_list=partition['train'])