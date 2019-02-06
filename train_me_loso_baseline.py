import os
import sys
import torch

from torch.optim import Adam, SGD, lr_scheduler

from capsule.data import load_me, data_split, sample_data
from capsule.loss import me_loss

from torchvision import transforms
from capsule.data import get_meta_data, Dataset
from capsule.evaluations import Meter

from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, f1_score
from sklearn.utils import shuffle
import pickle
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss



criterion = CrossEntropyLoss()

# VGG Baseline
class VGG(nn.Module):
	def __init__(self):
		super(VGG, self).__init__()
		self.model = models.vgg11(pretrained=True)
		self.model.classifier[6] = nn.Linear(in_features=4096, out_features=3)

	def forward(self, x):
		output = F.softmax(self.model(x), dim=-1)
		return output

# ResNet Baseline
class ResNet(nn.Module):
	def __init__(self):
		super(ResNet, self).__init__()
		self.model = models.resnet18(pretrained=True)
		self.model.fc = nn.Linear(in_features=512, out_features=3)

		for module in ['conv1', 'bn1', 'layer1']:
			for param in getattr(self.model, module).parameters():
				param.requires_grad = False

	def forward(self, x):
		output = F.softmax(self.model(x), dim=-1)
		return output


data_apex_frame_path = 'datasets/data_apex.csv'
data_four_frames_path = 'datasets/data_four_frames.csv'
data_root = '/home/ubuntu/Datasets/MEGC/process/'
batch_size = 32
lr = 0.0001
lr_decay_value = 0.9
num_classes = 3
epochs = 30

x_meter = Meter()
batches_scores = []

def load_me_data(data_root, file_path, subject_out_idx, batch_size=32, num_workers=4):
	df_train, df_val = data_split(file_path, subject_out_idx)
	df_four = pd.read_csv(data_four_frames_path)
	df_train_sampled = sample_data(df_train, df_four)
	df_train_sampled = shuffle(df_train_sampled)

	train_paths, train_labels = get_meta_data(df_train_sampled)

	train_transforms = transforms.Compose([transforms.Resize((234, 240)),
	                                       transforms.RandomRotation(degrees=(-8, 8)),
	                                       transforms.RandomHorizontalFlip(),
	                                       transforms.ColorJitter(brightness=0.2, contrast=0.2,
	                                                              saturation=0.2, hue=0.2),
	                                       transforms.RandomCrop((224, 224)),
	                                       transforms.ToTensor()])

	train_dataset = Dataset(root=data_root,
	                        img_paths=train_paths,
	                        img_labels=train_labels,
	                        transform=train_transforms)

	val_transforms = transforms.Compose([transforms.Resize((234, 240)),
	                                     transforms.RandomRotation(degrees=(-8, 8)),
	                                     transforms.CenterCrop((224, 224)),
	                                     transforms.ToTensor()])

	val_paths, val_labels = get_meta_data(df_val)
	val_dataset = Dataset(root=data_root,
	                      img_paths=val_paths,
	                      img_labels=val_labels,
	                      transform=val_transforms)

	train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
	                                           batch_size=batch_size,
	                                           num_workers=num_workers,
	                                           shuffle=True)

	val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
	                                         batch_size=batch_size,
	                                         num_workers=num_workers,
	                                         shuffle=False)
	return train_loader, val_loader





def on_epoch(model, optimizer, lr_decay, train_loader, test_loader, epoch):
	model.train()
	lr_decay.step()  # decrease the learning rate by multiplying a factor `gamma`
	train_loss = 0.0
	correct = 0.
	meter = Meter()

	steps = len(train_loader.dataset) // batch_size + 1
	with tqdm(total=steps) as progress_bar:
		for i, (x, y) in enumerate(train_loader):  # batch training
			y = torch.zeros(y.size(0), num_classes).scatter_(1, y.view(-1, 1),
			                                                 1.)  # change to one-hot coding
			x, y = x.cuda(), y.cuda()  # convert input data to GPU Variable

			optimizer.zero_grad()  # set gradients of optimizer to zero
			y_pred = model(x)  # forward
			y_true = y.data.max(1)[1]
			loss = criterion(y_pred, y_true)
			loss.backward()  # backward, compute all gradients of loss w.r.t all Variables
			train_loss += loss.item() * x.size(0)  # record the batch loss
			optimizer.step()  # update the trainable parameters with computed gradients

			y_pred = y_pred.data.max(1)[1]

			meter.add(y_true.cpu().numpy(), y_pred.cpu().numpy())
			correct += y_pred.eq(y_true).cpu().sum()

			progress_bar.set_postfix(loss=loss.item(), correct=correct)
			progress_bar.update(1)

		train_loss /= float(len(train_loader.dataset))
		train_acc = float(correct.item()) / float(len(train_loader.dataset))
		scores = meter.value()
		meter.reset()
		print('Training UAR: %.4f' % (scores[0].mean()), scores[0])
		print('Training UF1: %.4f' % (scores[1].mean()), scores[1])


	correct = 0.
	test_loss = 0.

	model.eval()
	for i, (x, y) in enumerate(test_loader):  # batch training
		y = torch.zeros(y.size(0), num_classes).scatter_(1, y.view(-1, 1),
		                                                 1.)  # change to one-hot coding
		x, y = x.cuda(), y.cuda()  # convert input data to GPU Variable

		y_pred = model(x)  # forward
		y_true = y.data.max(1)[1]

		loss = criterion(y_pred, y_true)  # compute loss
		test_loss += loss.item() * x.size(0)  # record the batch loss

		y_pred = y_pred.data.max(1)[1]

		meter.add(y_true.cpu().numpy(), y_pred.cpu().numpy())
		correct += y_pred.eq(y_true).cpu().sum()

		if (epoch + 1) % 2 == 0 and i % steps == 0:
			print('y_true\n', y_true[:30])
			print('y_pred\n', y_pred[:30])
			print('y_true', y.sum(dim=0))

	scores = meter.value()
	print('y_true', y.sum(dim=0))
	print('Testing UAR: %.4f' % (scores[0].mean()), scores[0])
	print('Testing UF1: %.4f' % (scores[1].mean()), scores[1])

	test_loss /= float(len(test_loader.dataset))
	test_acc = float(correct.item()) / float(len(test_loader.dataset))
	return train_loss, train_acc, test_loss, test_acc, meter


def train_eval(subject_out_idx):
	best_val_uf1 = 0.0
	best_val_uar = 0.0

	# Model & others
	model = VGG()
	model.cuda()
	optimizer = Adam(model.parameters(), lr=lr)
	lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay_value)

	for epoch in range(epochs):
		train_loader, test_loader = load_me_data(data_root, data_apex_frame_path,
		                                         subject_out_idx=subject_out_idx,
		                                         batch_size=batch_size)

		train_loss, train_acc, test_loss, test_acc, meter = on_epoch(model, optimizer, lr_decay,
		                                                              train_loader, test_loader,
		                                                              epoch)

		print("==> Subject out: %02d - Epoch %02d: loss=%.5f, train_acc=%.5f, val_loss=%.5f, "
		      "val_acc=%.4f"
		      % (subject_out_idx, epoch, train_loss, train_acc,
		         test_loss, test_acc))

		scores = meter.value()
		if scores[1].mean() >= best_val_uf1:
			best_val_uar = scores[0].mean()
			best_val_uf1 = scores[1].mean()
			x_meter.add(meter.Y_true, meter.Y_pred)

	return best_val_uar, best_val_uf1



for i in range(68):
	scores = train_eval(subject_out_idx=i)
	batches_scores.append(scores)
	x_scores = x_meter.value()
	print('final uar', x_scores[0], x_scores[0].mean())
	print('final uf1', x_scores[1], x_scores[1].mean())

with open('scores_vgg11_no_macro.pkl', 'wb') as file:
	data = dict(meter=x_meter, batches_scores=batches_scores)
	pickle.dump(data, file)