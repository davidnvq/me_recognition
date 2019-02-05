import os
import torch
import numpy as np
import moviepy.editor as mpy


def one_hot_encode(labels, num_classes=None):
	if num_classes is None:
		num_classes = len(labels.unique())

	return torch.eye(num_classes, dtype=torch.long).index_select(dim=0, index=labels)


def make_gif(images, fname, duration=1, true_image=False):
	def make_frame(t):
		try:
			x = images[int(len(images) / duration * t)]
		except:
			x = images[-1]

		if true_image:
			return x.astype(np.uint8)
		else:
			return ((x + 1) / 2 * 255).astype(np.uint8)

	clip = mpy.VideoClip(make_frame, duration=duration)
	clip.write_gif(fname, fps=len(images) / duration)


class Saver(object):

	def __init__(self, save_dir="./checkpoints", args="",
	             train_parallel=False, save_modes=['epoch', 'best']):
		self.save_dir = save_dir
		self.args = args
		self.train_parallel = train_parallel
		self.save_modes = save_modes
		self.history = []
		self.best_acc = 0.0

		if not os.path.exists(self.save_dir):
			os.makedirs(self.save_dir)


	def save_module(self, module, file_name=None):
		"""Save module only"""
		file_name = "module" if file_name is None else file_name

		file_path = os.path.join(self.save_dir, file_name)
		torch.save(module.state_dict(), file_path)


	def save_model(self, model, epoch, accuracy):
		"""Save the checkpoint

		Args:
			model (nn.Module): The model
			epoch (int): The current epoch
			accuracy (int): the current accuracy
		"""
		self.history.append([epoch, accuracy])

		for save_mode in self.save_modes:
			if save_mode == 'epoch':
				file_name = 'model_epoch_%i_acc_%.3f.pth' % (epoch, accuracy)
			elif save_mode == 'best':
				file_name = 'resnet_model_best_acc_epoch%d_%.3f.pth' % (epoch, accuracy)
				if self.best_acc < accuracy:
					self.best_acc = accuracy
				else:
					continue
			else:
				raise TypeError("Invalid save_mode! Please set values: 'epoch' or 'best'.")

			self._save_checkpoint(model, file_name, epoch, accuracy)


	def _save_checkpoint(self, model, file_name, epoch, accuracy):
		if self.train_parallel:
			state_dict = model.module.state_dict()
		else:
			state_dict = model.state_dict()

		checkpoint = {
			"model"   : state_dict,
			"epoch"   : epoch,
			"accuracy": accuracy,
			"args"    : self.args,
			"history" : self.history
			}

		model_path = os.path.join(self.save_dir, file_name)
		torch.save(checkpoint, model_path)
