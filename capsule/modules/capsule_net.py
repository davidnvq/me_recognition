import torch
import torch.nn as nn
import torch.nn.functional as F
from .capsule_layers import PrimaryCapsule, MECapsule
from .activations import squash
from torchvision import models


class ResNet(nn.Module):
	def __init__(self, is_freeze=False):
		super(ResNet, self).__init__()
		self.model = models.resnet18(pretrained=True)
		delattr(self.model, 'layer4')
		delattr(self.model, 'avgpool')
		delattr(self.model, 'fc')

		if is_freeze:
			for index, p in enumerate(self.model.parameters()):
				if index == 15:
					break
				p.requires_grad = False

	def forward(self, x):
		output = self.model.conv1(x)
		output = self.model.bn1(output)
		output = self.model.relu(output)
		output = self.model.layer1(output)
		output = self.model.layer2(output)
		output = self.model.layer3(output)
		return output


class VGG(nn.Module):
	def __init__(self, is_freeze=True):
		super(VGG, self).__init__()
		self.model = models.vgg11(pretrained=True).features[:11]

		if is_freeze:
			for i in range(4):
				for p in self.model[i].parameters():
					p.requires_grad = False

	def forward(self, x):
		# x = [B, 3, 224, 224]
		return self.model(x) # [B, 256, 20, 20]

backbone = {'vgg' : VGG, 'resnet' : ResNet}


class MECapsuleNet(nn.Module):
	"""
	A Capsule Network on Micro-expression.
	:param input_size: data size = [channels, width, height]
	:param classes: number of classes
	:param routings: number of routing iterations
	Shape:
		- Input: (batch, channels, width, height), optional (batch, classes) .
		- Output:((batch, classes), (batch, channels, width, height))
	"""

	def __init__(self, input_size, classes, routings, conv_name='resnet', is_freeze=True):
		super(MECapsuleNet, self).__init__()
		self.input_size = input_size
		self.classes = classes
		self.routings = routings

		self.conv = backbone[conv_name](is_freeze)

		self.conv1 = nn.Conv2d(256, 256, kernel_size=9, stride=1, padding=0)

		self.primarycaps = PrimaryCapsule(256, 32 * 8, 8, kernel_size=9, stride=2, padding=0)

		self.digitcaps = MECapsule(in_num_caps=32 * 6 * 6,
		                           in_dim_caps=8,
		                           out_num_caps=self.classes,
		                           out_dim_caps=16,
		                           routings=routings)

		self.relu = nn.ReLU()

	def forward(self, x, y=None):
		x = self.conv(x)
		x = self.relu(self.conv1(x))
		x = self.primarycaps(x)
		x = self.digitcaps(x)
		length = x.norm(dim=-1)
		return length
