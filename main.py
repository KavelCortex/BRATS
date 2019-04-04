import os
import tables
from unet.model import unet_model_3d,load_old_model
from unet import generator
import matplotlib.pyplot as plt
import numpy as np
import keras

config={}
config['is_on_cheaha']=False
config['data_dir']='$USER_SCRATCH' if config['is_on_cheaha'] else 'data'
config['brats_file']='brats_data_64.h5'
config['model_file']='model.hdf5'
config['train_percentage']=0.7
config['batch_size']=2
config['epoch']=100



img_data=tables.open_file(os.path.join(config['data_dir'],config['brats_file']),'r')
data_num=img_data.root.data.shape[0]
partition={}
partition['train']=list(range(int(data_num*config['train_percentage'])))
partition['test']=list(range(int(data_num*config['train_percentage']),data_num))

print("Data shape: ",img_data.root.data.shape)
print("Truth shape: ",img_data.root.truth.shape)
training_generator,training_steps=generator.get_generator(img_data=img_data,idx_list=partition['train'],batch_size=config['batch_size'])
validation_generator,validation_steps=generator.get_generator(img_data=img_data,idx_list=partition['test'],batch_size=config['batch_size'])

model=None
if os.path.exists(config['model_file']):
    model=load_old_model(config['model_file'])
else:
    model = unet_model_3d(input_shape=img_data.root.data.shape[-4:])

# model.summary()

# Train
cb_1=keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
cb_2=keras.callbacks.ModelCheckpoint(filepath=os.path.join(config['data_dir'],'weights_{epoch:02d}_{val_loss:.5f}.hdf5'), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
results = model.fit_generator(generator=training_generator,steps_per_epoch=training_steps,validation_data=validation_generator,validation_steps=validation_steps,epochs=config['epoch'],callbacks=[cb_1,cb_2])