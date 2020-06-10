import torch
from tools import CatMeter, cosine_dist, visualize_ranked_results


def visualize(config, base, loaders):

	base.set_eval()

	# meters
	query_features_meter, query_pids_meter, query_cids_meter = CatMeter(), CatMeter(), CatMeter()
	gallery_features_meter, gallery_pids_meter, gallery_cids_meter = CatMeter(), CatMeter(), CatMeter()

	# init dataset
	if config.visualize_dataset == 'market':
		_datasets = [loaders.market_query_samples, loaders.market_gallery_samples]
		_loaders = [loaders.market_query_loader, loaders.market_gallery_loader]
	elif config.visualize_dataset == 'duke':
		_datasets = [loaders.duke_query_samples, loaders.duke_gallery_samples]
		_loaders = [loaders.duke_query_loader, loaders.duke_gallery_loader]
	elif config.visualize_dataset == 'customed':
		_datasets = [loaders.query_samples, loaders.gallery_samples]
		_loaders = [loaders.query_loader, loaders.gallery_loader]

	# compute query and gallery features
	with torch.no_grad():
		for loader_id, loader in enumerate(_loaders):
			for data in loader:
				# compute feautres
				images, pids, cids = data
				images = images.cuda()
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

	# compute distance
	query_features = query_features_meter.get_val()
	gallery_features = gallery_features_meter.get_val()
	distance = cosine_dist(query_features, gallery_features).data.cpu().numpy()

	# visualize
	visualize_ranked_results(distance, _datasets, config.visualize_output_path, mode=config.visualize_mode, only_show=config.visualize_mode_onlyshow)
