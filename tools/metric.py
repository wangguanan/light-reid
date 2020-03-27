import torch
import torch.nn.functional as F

def cosine_dist(x, y):
	'''
	compute cosine distance between two matrix x and y
	with size (n1, d) and (n2, d) and type torch.tensor
	return a matrix (n1, n2)
	'''

	x = F.normalize(x, dim=1)
	y = F.normalize(y, dim=1)
	return torch.matmul(x, y.transpose(0,1))


def euclidean_dist(x, y):
	"""
	compute euclidean distance between two matrix x and y
	with size (n1, d) and (n2, d) and type torch.tensor
	return a matrix (n1, n2)
	"""
	m, n = x.size(0), y.size(0)
	xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
	yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
	dist = xx + yy
	dist.addmm_(1, -2, x, y.t())
	dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
	return dist
