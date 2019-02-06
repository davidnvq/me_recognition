import os
import sys
import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import pandas as pd
import numpy as np

from PIL import Image


class Dataset(data.Dataset):

	def __init__(self, root, img_paths, img_labels, transform=None, get_aux=False, aux=None):
		"""Load image paths and labels from gt_file"""
		self.root = root
		self.transform = transform
		self.get_aux = get_aux
		self.img_paths = img_paths
		self.img_labels = img_labels
		self.aux = aux

	def __getitem__(self, idx):
		"""Load image.

		Args:
			idx (int): image idx.

		Returns:
			img (tensor): image tensor

		"""
		img_path = self.img_paths[idx]
		img = Image.open(os.path.join(self.root, img_path)).convert('RGB')
		label = self.img_labels[idx]

		if self.transform:
			img = self.transform(img)

		if self.get_aux:
			return img, label,

		return img, label, self.aux[idx]

	def __len__(self):
		return len(self.img_paths)


def get_triple_meta_data(file_path):
	df = pd.read_csv(file_path)
	on_paths = list(df.on_frame_path)
	apex_paths = list(df.apex_frame_path)
	off_paths = list(df.off_frame_path)

	paths = [(on, apex, off) for (on, apex, off) in zip(on_paths, apex_paths, off_paths)]
	labels = list(df.label)
	return paths, labels


def get_meta_data(df):
	paths = list(df.apex_frame_path)
	labels = list(df.label)

	return paths, labels


def data_split(file_path, subject_out_idx=0):
	"""Split dataset into train set and validation set
	"""
	# data, subject, clipID, label, apex_frame, apex_frame_path
	data_sub_column = 'data_sub'

	df = pd.read_csv(file_path)
	subject_list = list(df[data_sub_column].unique())
	subject_out = subject_list[subject_out_idx]
	print('subject_out', subject_out)
	df_train = df[df[data_sub_column] != subject_out]
	df_val = df[df[data_sub_column] == subject_out]

	return df_train, df_val


def upsample_subdata(df, df_four, number=4):
    result = df.copy()
    for i in range(df.shape[0]):
        quotient = number // 1
        remainder = number % 1
        remainder = 1 if np.random.rand() < remainder else 0
        value = quotient + remainder

        tmp = df_four[df_four['data_sub'] == df.iloc[i]['data_sub']]
        tmp = tmp[tmp['clip'] == df.iloc[i]['clip']]
        value = min(value, tmp.shape[0])
        tmp = tmp.sample(int(value))
        result = pd.concat([result, tmp])
    return result


def sample_data(df, df_four):
	df_neg = df[df.label == 0]
	df_pos = df[df.label == 1]
	df_sur = df[df.label == 2]

	num_sur = 4
	num_pos = 5 * df_sur.shape[0] / df_pos.shape[0] - 1
	num_neg = 5 * df_sur.shape[0] / df_neg.shape[0] - 1

	df_neg = upsample_subdata(df_neg, df_four, num_neg)
	df_pos = upsample_subdata(df_pos, df_four, num_pos)
	df_sur = upsample_subdata(df_sur, df_four, num_sur)
	print('df_neg', df_neg.shape)
	print('df_pos', df_pos.shape)
	print('df_sur', df_sur.shape)

	df = pd.concat([df_neg, df_pos, df_sur])
	return df

