import torch
from tools import *


def train_an_epoch(config, base, loaders):

	base.set_train()
	meter = MultiItemAverageMeter()

	### we assume 200 iterations as an epoch
	for _ in range(200):

		### load a batch data
		imgs, pids, _ = loaders.train_iter.next_one()
		imgs, pids = imgs.to(base.device), pids.to(base.device)

		### forward
		features, cls_score = base.model(imgs)

		### loss
		ide_loss = base.ide_creiteron(cls_score, pids)
		triplet_loss = base.triplet_creiteron(features, features, features, pids, pids, pids)
		loss = ide_loss + triplet_loss
		acc = accuracy(cls_score, pids, [1])[0]

		### optimize
		base.optimizer.zero_grad()
		loss.backward()
		base.optimizer.step()

		### recored
		meter.update({'ide_loss': ide_loss, 'triplet_loss': triplet_loss, 'acc': acc})

	return meter.get_val(), meter.get_str()