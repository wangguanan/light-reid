import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.optim as optim
from bisect import bisect_right
import os
from collections import OrderedDict

from .nets import Res50BNNeck, Res50IBNaBNNeck, osnet_ain_x1_0
from tools import CrossEntropyLabelSmooth, TripletLoss, os_walk


class Base:
	'''
	a base module includes model, optimizer, loss, and save/resume operations.
	'''

	def __init__(self, config):

		self.config = config
		# Model Config
		self.cnnbackbone = config.cnnbackbone
		self.pid_num = config.pid_num
		self.margin = config.margin
		# Logger Configuration
		self.max_save_model_num = config.max_save_model_num
		self.output_path = config.output_path
		# Train Configuration
		self.base_learning_rate = config.base_learning_rate
		self.weight_decay = config.weight_decay
		self.milestones = config.milestones

		# init
		self._init_device()
		self._init_model()
		self._init_creiteron()
		self._init_optimizer()


	def _init_device(self):
		self.device = torch.device('cuda')


	def _init_model(self):
		pretrained = False if self.config.mode != 'train' else True
		if self.cnnbackbone == 'res50':
			self.model = Res50BNNeck(class_num=self.pid_num, pretrained=pretrained).to(self.device)
		elif self.cnnbackbone == 'res50ibna':
			self.model = Res50IBNaBNNeck(class_num=self.pid_num, pretrained=pretrained).to(self.device)
		elif self.cnnbackbone == 'osnetain':
			self.model = osnet_ain_x1_0(num_classes=self.pid_num, pretrained=pretrained, loss='softmax')
		else:
			assert 0, 'cnnbackbone error, expect res50, res50ibna, osnetain'
		self.model = self.model.to(self.device)


	def _init_creiteron(self):
		self.ide_creiteron = CrossEntropyLabelSmooth(self.pid_num)
		self.triplet_creiteron = TripletLoss(self.margin, 'euclidean')


	def _init_optimizer(self):
		if 'res' in self.cnnbackbone:
			self.optimizer = optim.Adam(self.model.parameters(), lr=self.base_learning_rate, weight_decay=self.weight_decay)
			self.lr_scheduler = WarmupMultiStepLR(self.optimizer, self.milestones, gamma=0.1, warmup_factor=0.01, warmup_iters=10)
		elif self.cnnbackbone == 'osnetain':
			self.optimizer = optim.Adam(self.model.parameters(), lr=self.base_learning_rate, amsgrad=True, weight_decay=self.weight_decay)
			self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.config.total_train_epochs)


	def save_model(self, save_epoch):
		'''save model as save_epoch'''
		# save model
		file_path = os.path.join(self.output_path, 'model_{}.pkl'.format(save_epoch))
		torch.save(self.model.state_dict(), file_path)
		# if saved model is more than max num, delete the model with smallest iter
		if self.max_save_model_num > 0:
			# find all files in format of *.pkl
			root, _, files = os_walk(self.output_path)
			for file in files:
				if '.pkl' not in file:
					files.remove(file)
			# remove extra model
			if len(files) > self.max_save_model_num:
				file_iters = sorted([int(file.replace('.pkl', '').split('_')[1]) for file in files], reverse=False)
				file_path = os.path.join(root, 'model_{}.pkl'.format(file_iters[0]))
				os.remove(file_path)

	def resume_last_model(self):
		'''resume model from the last one in path self.output_path'''
		# find all files in format of *.pkl
		root, _, files = os_walk(self.output_path)
		for file in files:
			if '.pkl' not in file:
				files.remove(file)
		# find the last one
		if len(files) > 0:
			# get indexes of saved models
			indexes = []
			for file in files:
				indexes.append(int(file.replace('.pkl', '').split('_')[-1]))
			indexes = sorted(list(set(indexes)), reverse=False)
			# resume model from the latest model
			self.resume_model(indexes[-1])
			#
			start_train_epoch = indexes[-1]
			return start_train_epoch
		else:
			return 0

	def resume_model(self, resume_epoch):
		'''resume model from resume_epoch'''
		model_path = os.path.join(self.output_path, 'model_{}.pkl'.format(resume_epoch))
		self.model.load_state_dict(torch.load(model_path), strict=False)
		print(('successfully resume model from {}'.format(model_path)))

	def resume_from_model(self, model_path):
		'''resume from model. model_path shoule be like /path/to/model.pkl'''
		# self.model.load_state_dict(torch.load(model_path), strict=False)
		# print(('successfully resume model from {}'.format(model_path)))
		state_dict = torch.load(model_path)
		model_dict = self.model.state_dict()
		new_state_dict = OrderedDict()
		matched_layers, discarded_layers = [], []
		for k, v in state_dict.items():
			if k.startswith('module.'):
				k = k[7:]  # discard module.
			if k in model_dict and model_dict[k].size() == v.size():
				new_state_dict[k] = v
				matched_layers.append(k)
			else:
				discarded_layers.append(k)
		model_dict.update(new_state_dict)
		self.model.load_state_dict(model_dict)
		if len(discarded_layers) > 0:
			print('discarded layers: {}'.format(discarded_layers))

	def set_train(self):
		'''set model as train mode'''
		self.model = self.model.train()
		self.training = True

	def set_eval(self):
		'''set model as eval mode'''
		self.model = self.model.eval()
		self.training = False


class DemoBase(Base):
	'''
	base for demo
	remove unnecessary operations such as init optimizer
	'''

	def __init__(self, config):
		self.pid_num = config.pid_num
		# init model
		self._init_device()
		self._init_model()


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):

	def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3, warmup_iters=500, warmup_method="linear", last_epoch=-1):
		if not list(milestones) == sorted(milestones):
			raise ValueError(
				"Milestones should be a list of" " increasing integers. Got {}",
				milestones,
			)

		if warmup_method not in ("constant", "linear"):
			raise ValueError(
				"Only 'constant' or 'linear' warmup_method accepted"
				"got {}".format(warmup_method)
			)
		self.milestones = milestones
		self.gamma = gamma
		self.warmup_factor = warmup_factor
		self.warmup_iters = warmup_iters
		self.warmup_method = warmup_method
		super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

	def get_lr(self):
		warmup_factor = 1
		if self.last_epoch < self.warmup_iters:
			if self.warmup_method == "constant":
				warmup_factor = self.warmup_factor
			elif self.warmup_method == "linear":
				alpha = float(self.last_epoch) / float(self.warmup_iters)
				warmup_factor = self.warmup_factor * (1 - alpha) + alpha
		return [
			base_lr
			* warmup_factor
			* self.gamma ** bisect_right(self.milestones, self.last_epoch)
			for base_lr in self.base_lrs
		]
