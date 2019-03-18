import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import recall_score, f1_score

class Meter(object):

	def __init__(self):
		"""
		To record the measure the performance.
		"""
		self.Y_true = np.array([], dtype=np.int)
		self.Y_pred = np.array([], dtype=np.int)


	def add(self, y_true, y_pred, verbose=False):
		if len(self.Y_true.shape) != len(y_true.shape):
			print('shape self.Y_true', self.Y_true.shape)
			print('y_true', y_true.shape)

		self.Y_true = np.concatenate((self.Y_true, y_true))
		self.Y_pred = np.concatenate((self.Y_pred, y_pred))

	def reset(self):
		self.Y_true = np.array([], dtype=np.int)
		self.Y_pred = np.array([], dtype=np.int)

	def value(self):
		eye = np.eye(3, dtype=np.int)
		Y_true = eye[self.Y_true]
		Y_pred = eye[self.Y_pred]
		uar = recall_score(Y_true, Y_pred, average=None)
		uf1 = f1_score(Y_true, Y_pred, average=None)
		return uar, uf1


class UF1(nn.Module):

	def __init__(self):
		super(UF1, self).__init__()


	def forward(self, outputs, targets):
		"""Compute Unweighted F1 score on k folds of LOSO
		Args:
			outputs (list): [k folds]
			targets (list): [k folds]

		Returns:
			UF1 = F1c/C
		"""

		k_folds = len(outputs)
		num_classes = outputs[0].size(1)
		UF1 = 0.0
		for c in range(num_classes):
			TPc, FPc, FNc = 0.0, 0.0, 0.0

			for fold in range(k_folds):
				res = self.compute_on_fold(outputs[fold][:, c], targets[fold][:, c])
				TPc += res[0]
				FPc += res[1]
				FNc += res[2]

			F1c = (2*TPc) / (2 * TPc + FPc + FNc)

		UF1 += F1c / num_classes
		return UF1

	@staticmethod
	def compute_on_fold(output, target):
		"""
		Args
			output (torch.tensor): [1, 0, 1, 1, 1]
			target (torch.tensor): [1, 1, 1, 1, 0]
		Returns:
			(TP, FP, FN): True Positive, False Positive, False Negative
			TP = 3, FP = 1, FN = 1
		"""
		output = output >= 0.5
		target = target >= 0.5

		TP = target.__and__(output).sum()
		FP = (1 - target).__and__(output).sum()
		FN = target.__and__(1 - output).sum()
		return TP.float(), FP.float(), FN.float()


class UARecall(nn.Module):

	def __init__(self):
		super(UARecall, self).__init__()

	def forward(self, outputs, targets):
		"""Compute Unweighted Average Recall
		Args:
			outputs:
			targets:
		Returns:
			UAR = 1/C * sum (Acc_c) where Acc_c is per-class accuracy: Acc_c = TPc/n_c
		"""
		num_classes = outputs.size(1)
		UAR = 0.0

		for c in range(num_classes):
			Acc_c = self.compute_acc(outputs[:, c], targets[:, c])
			UAR += Acc_c / float(num_classes)
		return UAR

	@staticmethod
	def compute_acc(output, target):
		output = output >= 0.5
		target = target >= 0.5

		TP = target.__and__(output).sum()
		Nc = len(target)
		if Nc == 0:
			print('Nc = 0! and TP = ', TP)
		else:
			print("NC is not Zero", Nc, 'while TP=', TP)
		acc = TP.float() / Nc
		return acc

