import torch
import numpy as np
from tools import *


def test(config, base, loaders, test_dataset):

	base.set_eval()

	# meters
	query_features_meter, query_pids_meter, query_cids_meter = CatMeter(), CatMeter(), CatMeter()
	gallery_features_meter, gallery_pids_meter, gallery_cids_meter = CatMeter(), CatMeter(), CatMeter()

	# init dataset
	if test_dataset == 'market_test':
		loaders = [loaders.market_query_loader, loaders.market_gallery_loader]
	elif test_dataset == 'duke_test':
		loaders = [loaders.duke_query_loader, loaders.duke_gallery_loader]

	# compute query and gallery features
	with torch.no_grad():
		for loader_id, loader in enumerate(loaders):
			for data in loader:
				# compute feautres
				images, pids, cids = data
				features = base.model(images)
				# save as query features
				if loader_id == 0:
					query_features_meter.update(features.data)
					query_pids_meter.update(pids)
					query_cids_meter.update(cids)
				# save as gallery features
				elif loader_id == 1:
					gallery_features_meter.update(features.data)
					gallery_pids_meter.update(pids)
					gallery_cids_meter.update(cids)

	print('Time: {}, Successfully Compute Feature'.format(time_now()))

	# norm features
	query_features = query_features_meter.get_val_numpy()
	gallery_features = gallery_features_meter.get_val_numpy()
	print('Time: {}, Successfully Norm Feature'.format(time_now()))
	print query_features.shape, gallery_features.shape

	# compute mAP and rank@k
	result = PersonReIDMAP(
		query_features, query_cids_meter.get_val_numpy(), query_pids_meter.get_val_numpy(),
		gallery_features, gallery_cids_meter.get_val_numpy(), gallery_pids_meter.get_val_numpy(), dist='cosine')

	return result.mAP, list(result.CMC[0: 150])


