import os
import tables
from unet.model import unet_model_3d, load_old_model
from unet import generator
import matplotlib.pyplot as plt
import numpy as np
import keras

config = {}
config['env'] = 'cheaha'  # win/osx/cheaha
if config['env'] == 'win':
    config['data_dir'] = "C:\\Users\\kavel\\workspace\\3DUnetCNN"
elif config['env'] == 'osx':
    config['data_dir'] = 'data'
elif config['env'] == 'cheaha':
    config['data_dir'] = '/data/scratch/jwang96'
config['brats_file'] = 'brats_data.h5'
config['model_file'] = 'model.h5'
config['train_key_file'] = 'train_key.pkl'
config['validation_key_file'] = 'validation_key.pkl'
config['image_shape'] = (144, 144, 144)
config['patch_shape'] = (64, 64, 64)
config['nb_channels'] = 4
if "patch_shape" in config and config["patch_shape"] is not None:
    config["input_shape"] = tuple(
        [config["nb_channels"]] + list(config["patch_shape"]))
else:
    config["input_shape"] = tuple(
        [config["nb_channels"]] + list(config["image_shape"]))
config['train_percentage'] = 0.7
config['batch_size'] = 2
config['epoch'] = 200
config['multi-worker'] = 0

def main():
    img_data = tables.open_file(os.path.join(
        config['data_dir'], config['brats_file']), 'r')
    data_num = img_data.root.data.shape[0]
    partition = {}
    partition['train'] = list(range(int(data_num*config['train_percentage'])))
    partition['test'] = list(
        range(int(data_num*config['train_percentage']), data_num))

    print("Data shape: ", img_data.root.data.shape)
    print("Truth shape: ", img_data.root.truth.shape)
    training_generator, training_steps = generator.get_generator(
        img_data=img_data, idx_list=partition['train'], batch_size=config['batch_size'], patch_shape=config['patch_shape'], key_file=config['train_key_file'])
    validation_generator, validation_steps = generator.get_generator(
        img_data=img_data, idx_list=partition['test'], batch_size=config['batch_size'], patch_shape=config['patch_shape'], key_file=config['validation_key_file'])

    model = None
    # Load Checkpoint
    checkpoint_path = os.path.join(config['data_dir'], config['model_file'])
    if os.path.exists(checkpoint_path):
        print('Loading last checkpoint: ', str(checkpoint_path))
        model = load_old_model(checkpoint_path)
    else:
        model = unet_model_3d(input_shape=config["input_shape"])

    # model.summary()

    # Train
    cb_1 = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto')
    cb_2 = keras.callbacks.ModelCheckpoint(filepath=os.path.join(
        config['data_dir'], 'weights_{val_loss:.5f}.hdf5'), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    results = model.fit_generator(generator=training_generator, steps_per_epoch=training_steps, validation_data=validation_generator,
                                  validation_steps=validation_steps, epochs=config['epoch'], callbacks=[cb_1, cb_2], workers=config['multi-worker'])

if __name__ == "__main__":
    main()
