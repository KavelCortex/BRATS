import numpy as np
import keras
import random
import unet.utils.patches as patches
import unet.utils.augment as augments
import cv2
import itertools


def get_generator(img_data, idx_list, batch_size=2, patch_shape=None):
    generator = batch_generator(
        img_data=img_data, idx_list=idx_list, batch_size=batch_size, patch_shape=None)
    n_samples = len(idx_list)
    if patch_shape:
        patch_idxes = create_patch_idx_list(
            idx_list=idx_list, img_shape=img_data.root.data.shape[-3:], patch_shape=patch_shape)
        n_samples = get_num_of_patches(
            img_data=img_data, idx_list=patch_idxes, patch_shape=patch_shape)
    step_per_epoch = get_num_of_steps(
        n_samples=n_samples, batch_size=batch_size)
    return generator, step_per_epoch


def create_patch_idx_list(idx_list, img_shape, patch_shape):
    '''
    creates a list mixed with id and patch_idx.
    [[id1,pch1],[id1,pch2],...,[id1,pchn],...,[idn,pchn]]
    '''
    patch_idxes = patches.compute_patch_indices(
        image_shape=img_shape, patch_size=patch_shape, overlap=0)
    return itertools.product(idx_list, patch_idxes)


def get_num_of_patches(img_data, idx_list, patch_shape):
    '''
    return the actual number of fed patches by testing if its not empty.
    '''
    total_count=0
    notnull_count = 0
    for idx in idx_list:
        total_count+=1
        X, y = get_data_by_ID(img_data=img_data, idx=idx,
                              patch_shape=patch_shape)
        if np.any(y != 0):
            notnull_count += 1
            print('Preparing Patches: ID=%d Patch=%d/%d'%(idx[0],notnull_count,total_count),end='\r')
    print('\n Patches extracted: %d/%d'%(notnull_count,total_count))
    return notnull_count


def batch_generator(img_data, idx_list, batch_size=2, patch_shape=None, augment=True, permute=True):
    '''
    Standard batch generator with yield
    '''

    while True:
        idx_epoch = idx_list.copy()
        random.shuffle(idx_epoch)

        if patch_shape:
            idx_epoch = create_patch_idx_list(
                idx_list=idx_epoch, img_shape=img_data.root.data.shape[-3:], patch_shape=patch_shape)

        X_batch = []
        y_batch = []
        while len(idx_epoch) > 0:
            idx = idx_epoch.pop()

            X, y = get_data_by_ID(img_data=img_data, idx=idx)

            if augment:
                if patch_shape is None:
                    affine = img_data.root.affine[idx]
                else:
                    affine = img_data.root.affine[idx[0]]

                X, y = augments.augment_data(X, y, affine)

            if permute:
                X, y = augments.random_permutation_x_y(
                    X, y[np.newaxis])  # (n_channels, dim)
            else:
                y = y[np.newaxis]

            if np.any(y != 0):
                X_batch.append(X)
                y_batch.append(y)

            if len(X_batch) == batch_size or (len(idx_epoch) == 0 and len(X_batch) > 0):
                X_batch = np.asarray(X_batch)
                y_batch = np.asarray(y_batch)
                y_batch = get_multi_class_labels(y_batch)
                yield X_batch, y_batch
                X_batch = []
                y_batch = []


def get_data_by_ID(img_data, idx, patch_shape=None):
    '''
    get single pieces by index
    when it is whole, return the image by ID.
    when it is patched, only return the specified patch by patch_index.
    '''
    if patch_shape:
        ID, patch_idx = idx
        data, seg = get_data_by_ID(img_data, ID)
        X = patches.get_patch_from_3d_data(
            data=data, patch_shape=patch_shape, patch_index=patch_idx)
        y = patches.get_patch_from_3d_data(
            data=seg, patch_shape=patch_shape, patch_index=patch_idx)
    else:
        ID = idx
        X = img_data.root.data[ID]  # (n_channels, dim)
        y = img_data.root.truth[ID, 0]  # (dim)
    return X, y

    # data=[]
    # seg_mask=[]

    # if patch_shape is None:
    #     ID = idx
    # else:
    #     ID,patch_idx=idx

    # data = img_data.root.data[ID]  # (n_channels, dim)
    # seg_mask = img_data.root.truth[ID, 0]  # (dim)

    # if augment:
    #     affine = img_data.root.affine[ID]
    #     data, seg_mask = augments.augment_data(data, seg_mask, affine)
    # if permute:
    #     data, seg_mask = augments.random_permutation_x_y(
    #         data, seg_mask[np.newaxis])  # (n_channels, dim)

    # if patch_shape is not None:
    #     ID,patch_idx=idx
    #     patch_idxes=patches.compute_patch_indices(img_data.root.data.shape[-3:])
    #     data=patches.get_patch_from_3d_data(data=data,patch_shape=patch_shape,patch_idx=patch_idx)
    #     seg_mask=patches.get_patch_from_3d_data(data=seg_mask,patch_shape=patch_shape,patch_idx=patch_idx)

    # if augment:
    #   affine=img_data.root.affine[index[0]]
    # if patch_size is not None and patch_idxes is not None:
    #     idx_in_patch = int(np.floor(idx % len(patch_idxes)))
    #     patch_idx = patch_idxes[idx_in_patch]
    #     data = patches.get_patch_from_3d_data(
    #         data, [patch_size, patch_size, patch_size],
    #         patch_idx)  # (n_channels, patch_dim)
    #     seg_mask = patches.get_patch_from_3d_data(
    #         seg_mask, [patch_size, patch_size, patch_size],
    #         patch_idx)  # (1, patch_dim)


def get_multi_class_labels(data, n_labels=3, labels=[1, 2, 4]):
    """
    Translates a label map into a set of binary labels.
    :param data: numpy array containing the label map with shape: (n_samples, 1, ...).
    :param n_labels: number of labels.
    :param labels: integer values of the labels.
    :return: binary numpy array of shape: (n_samples, n_labels, ...)
    """
    new_shape = [data.shape[0], n_labels] + list(data.shape[2:])
    y = np.zeros(new_shape, np.int8)
    for label_index in range(n_labels):
        if labels is not None:
            y[:, label_index][data[:, 0] == labels[label_index]] = 1
        else:
            y[:, label_index][data[:, 0] == (label_index + 1)] = 1
    return y


def get_num_of_steps(n_samples, batch_size):
    if n_samples <= batch_size:
        return n_samples
    elif np.remainder(n_samples, batch_size) == 0:
        return n_samples/batch_size
    else:
        return n_samples//batch_size+1

# class SequenceGenerator(keras.utils.Sequence):
#     '''
#     Inherit from Sequence to support Keras multiprocessing
#     '''

#     def __init__(self,
#                  img_data,
#                  list_IDs,
#                  patch_size=None,
#                  batch_size=2,
#                  n_channels=4,
#                  n_classes=3,
#                  shuffle=True,
#                  shrink_size=None):
#         self.dim = img_data.root.data.shape[-3:]
#         self.list_IDs = list_IDs
#         self.batch_size = batch_size
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.shuffle = shuffle
#         self.img_data = img_data
#         self.shrink_size = shrink_size

#         # small patches
#         self.patch_size = patch_size
#         self.patch_idxes = None
#         if self.patch_size is not None:
#             self.patch_idxes = patches.compute_patch_indices(
#                 image_shape=self.dim, patch_size=patch_size, overlap=0)

#         # TODO: small origins
#         if self.shrink_size is not None:
#             self.dim = [shrink_size, shrink_size, shrink_size]

#         self.on_epoch_end()  # assign indexes

#     def __len__(self):
#         # calculate the number of batches per epoch
#         if self.patch_size is None:
#             return int(np.floor(len(self.list_IDs) / self.batch_size))
#         else:
#             return int(
#                 np.floor(
#                     len(self.list_IDs) * len(self.patch_idxes) /
#                     self.batch_size))

#     def __getitem__(self, index):
#         # get the indexes for ith batch
#         indexes = self.indexes[index * self.batch_size:(index + 1) *
#                                self.batch_size]
#         # get ID list by indexes
#         list_IDs_temp = [self.list_IDs[i] for i in indexes]

#         # return data
#         X, y = self.__data_generation(list_IDs_temp)  # X is too much
#         return X, y

#     def on_epoch_end(self):
#         # generate indexes for current epoch, shuffle if possible
#         if self.patch_size is None:
#             self.indexes = np.arange(len(self.list_IDs))
#         else:
#             self.indexes = np.arange(len(self.list_IDs))
#         if self.shuffle == True:
#             np.random.shuffle(self.indexes)

#     def __data_generation(self, list_IDs_temp):
#         # generate data with batch_size samples
#         X = []
#         y = []
#         # generate data
#         if self.patch_size is None:
#             X = np.empty((self.batch_size, self.n_channels, *self.dim))
#             y = np.empty((self.batch_size, self.n_classes, *self.dim))
#         else:
#             patch_dim = [self.patch_size, self.patch_size, self.patch_size]
#             X = np.empty((self.batch_size, self.n_channels, *patch_dim))
#             y = np.empty((self.batch_size, self.n_classes, *patch_dim))
#         for i, idx in enumerate(list_IDs_temp):
#             data, seg_mask = get_data_by_ID(self.img_data,
#                                             idx, patch_size=self.patch_size, patch_idxes=self.patch_idxes)
#             X[i, ] = data
#             y[i, ] = seg_mask

#         X = np.asarray(X)
#         y = np.asarray(y)
#         y = get_multi_class_labels(y)
#         return X, y
