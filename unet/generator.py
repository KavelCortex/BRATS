import numpy as np
import keras
import random
import unet.utils.patches as patches
import cv2


def batch_generator(img_data, idx_list,batch_size=2):
    '''
    Standard batch generator with yield
    '''
    random.shuffle(idx_list)
    X_batch = []
    y_batch = []
    while len(idx_list)>0:
        idx=idx_list.pop()
        X,y=get_data_by_ID(img_data=img_data,idx=idx)
        if np.any(y!=0):
            X_batch.append(X)
            y_batch.append(y)
        if len(X_batch)==batch_size or (len(idx_list)==0 and len(X_batch)>0):
            yield X_batch,y_batch
            X_batch=[]
            y_batch=[]


class SequenceGenerator(keras.utils.Sequence):
    '''
    Inherit from Sequence to support Keras multiprocessing
    '''

    def __init__(self,
                 img_data,
                 list_IDs,
                 patch_size=None,
                 batch_size=2,
                 n_channels=4,
                 n_classes=3,
                 shuffle=True,
                 shrink_size=None):
        self.dim = img_data.root.data.shape[-3:]
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.img_data = img_data
        self.shrink_size = shrink_size

        # small patches
        self.patch_size = patch_size
        self.patch_idxes = None
        if self.patch_size is not None:
            self.patch_idxes = patches.compute_patch_indices(
                image_shape=self.dim, patch_size=patch_size, overlap=0)

        # TODO: small origins
        if self.shrink_size is not None:
            self.dim = [shrink_size, shrink_size, shrink_size]

        self.on_epoch_end()  # assign indexes

    def __len__(self):
        # calculate the number of batches per epoch
        if self.patch_size is None:
            return int(np.floor(len(self.list_IDs) / self.batch_size))
        else:
            return int(
                np.floor(
                    len(self.list_IDs) * len(self.patch_idxes) /
                    self.batch_size))

    def __getitem__(self, index):
        # get the indexes for ith batch
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self.batch_size]
        # get ID list by indexes
        list_IDs_temp = [self.list_IDs[i] for i in indexes]

        # return data
        X, y = self.__data_generation(list_IDs_temp)  # X is too much
        return X, y

    def on_epoch_end(self):
        # generate indexes for current epoch, shuffle if possible
        if self.patch_size is None:
            self.indexes = np.arange(len(self.list_IDs))
        else:
            self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # generate data with batch_size samples
        X = []
        y = []
        # generate data
        if self.patch_size is None:
            X = np.empty((self.batch_size, self.n_channels, *self.dim))
            y = np.empty((self.batch_size, self.n_classes, *self.dim))
        else:
            patch_dim = [self.patch_size, self.patch_size, self.patch_size]
            X = np.empty((self.batch_size, self.n_channels, *patch_dim))
            y = np.empty((self.batch_size, self.n_classes, *patch_dim))
        for i, idx in enumerate(list_IDs_temp):
            data, seg_mask_3ch = get_data_by_ID(
                idx, patch_size=self.patch_size, patch_idxes=self.patch_idxes)
            X[i, ] = data
            y[i, ] = seg_mask_3ch

        return X, y


def get_data_by_ID(img_data,idx, patch_size=None, patch_idxes=None):
    ID = idx if patch_size is None else int(
        np.floor(idx / len(patch_idxes)))
    data = img_data.root.data[ID]  # (n_channels, dim)
    seg_mask = img_data.root.truth[ID]  # (1, dim)

    if patch_size is not None and patch_idxes is not None:
        idx_in_patch = int(np.floor(idx % len(patch_idxes)))
        patch_idx = patch_idxes[idx_in_patch]
        data = patches.get_patch_from_3d_data(
            data, [patch_size, patch_size, patch_size],
            patch_idx)  # (n_channels, patch_dim)
        seg_mask = patches.get_patch_from_3d_data(
            seg_mask, [patch_size, patch_size, patch_size],
            patch_idx)  # (n_channels, patch_dim)

    seg_mask_1 = np.zeros_like(seg_mask)
    seg_mask_1[seg_mask.astype(int) == 1] = 1
    seg_mask_2 = np.zeros_like(seg_mask)
    seg_mask_2[seg_mask.astype(int) == 2] = 1
    seg_mask_3 = np.zeros_like(seg_mask)
    seg_mask_3[seg_mask.astype(int) == 4] = 1
    seg_mask_3ch = np.concatenate([seg_mask_1, seg_mask_2, seg_mask_3],
                                  axis=0).astype(int)

    return data, seg_mask
