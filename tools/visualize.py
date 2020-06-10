import numpy as np
import os
import os.path as osp
import shutil
import sys
from PIL import Image, ImageOps, ImageDraw

from .utils import make_dirs


def visualize_ranked_results2(distmat, dataset, save_dir='', topk=20):
    """Visualizes ranked results.

    Supports both image-reid and video-reid.

    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        dataset (tuple): a 2-tuple containing (query, gallery), each of which contains
            tuples of (img_path(s), pid, camid).
        save_dir (str): directory to save output images.
        topk (int, optional): denoting top-k images in the rank list to be visualized.
    """
    num_q, num_g = distmat.shape

    print('Visualizing top-{} ranks'.format(topk))
    print('# query: {}\n# gallery {}'.format(num_q, num_g))
    print('Saving images to "{}"'.format(save_dir))

    query, gallery = dataset
    assert num_q == len(query)
    assert num_g == len(gallery)

    indices = np.argsort(distmat, axis=1)
    make_dirs(save_dir)

    def _cp_img_to(src, dst, rank, prefix):
        """
        Args:
            src: image path or tuple (for vidreid)
            dst: target directory
            rank: int, denoting ranked position, starting from 1
            prefix: string
        """
        if isinstance(src, tuple) or isinstance(src, list):
            dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3))
            make_dirs(dst)
            for img_path in src:
                shutil.copy(img_path, dst)
        else:
            dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3) + '_name_' + osp.basename(src))
            shutil.copy(src, dst)

    for q_idx in range(num_q):
        qimg_path, qpid, qcamid = query[q_idx]
        if isinstance(qimg_path, tuple) or isinstance(qimg_path, list):
            qdir = osp.join(save_dir, osp.basename(qimg_path[0]))
        else:
            qdir = osp.join(save_dir, osp.basename(qimg_path))
        make_dirs(qdir)
        _cp_img_to(qimg_path, qdir, rank=0, prefix='query')

        rank_idx = 1
        for g_idx in indices[q_idx, :]:
            gimg_path, gpid, gcamid = gallery[g_idx]
            invalid = (qpid == gpid) & (qcamid == gcamid)
            if not invalid:
                _cp_img_to(gimg_path, qdir, rank=rank_idx, prefix='gallery')
                rank_idx += 1
                if rank_idx > topk:
                    break

    print("Done")


def visualize_ranked_results(distmat, dataset, save_dir='', topk=20, sort='descend', mode='inter-camera', only_show=None):
    """Visualizes ranked results.
    Args:
        dismat (numpy.ndarray): distance matrix of shape (nq, ng)
        dataset (tupple): a 2-tuple including (query,gallery), each of which contains
            tuples of (img_paths, pids, camids)
        save_dir (str): directory to save output images.
        topk (int, optional): denoting top-k images in the rank list to be visualized.
        sort (string): ascend means small value is similar, otherwise descend
        mode (string): intra-camera/inter-camera/all
            intra-camera only visualize results in the same camera with the query
            inter-camera only visualize results in the different camera with the query
            all visualize all results
    """
    num_q, num_g = distmat.shape

    print('Visualizing top-{} ranks'.format(topk))
    print('# query: {}\n# gallery {}'.format(num_q, num_g))
    print('Saving images to "{}"'.format(save_dir))

    query, gallery = dataset
    assert num_q == len(query)
    assert num_g == len(gallery)
    assert sort in ['descend', 'ascend']
    assert mode in ['intra-camera', 'inter-camera', 'all']

    if sort is 'ascend':
        indices = np.argsort(distmat, axis=1)
    elif sort is 'descend':
        indices = np.argsort(distmat, axis=1)[:, ::-1]

    make_dirs(save_dir)

    def cat_imgs_to(image_list, hit_list, text_list, target_dir):

        images = []
        for img, hit, text in zip(image_list, hit_list, text_list):
            img = Image.open(img).resize((64, 128))
            d = ImageDraw.Draw(img)
            d.text((3, 1), "{:.3}".format(text), fill=(255, 255, 0))
            if hit:
                img = ImageOps.expand(img, border=4, fill='green')
            else:
                img = ImageOps.expand(img, border=4, fill='red')
            images.append(img)

        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        new_im.save(target_dir)

    counts = 0
    for q_idx in range(num_q):

        image_list = []
        hit_list = []
        text_list = []

        # query image
        qimg_path, qpid, qcamid = query[q_idx]
        image_list.append(qimg_path)
        hit_list.append(True)
        text_list.append(0.0)

        # target dir
        if isinstance(qimg_path, tuple) or isinstance(qimg_path, list):
            qdir = osp.join(save_dir, osp.basename(qimg_path[0]))
        else:
            qdir = osp.join(save_dir, osp.basename(qimg_path))

        # matched images
        rank_idx = 1
        for ii, g_idx in enumerate(indices[q_idx, :]):
            gimg_path, gpid, gcamid = gallery[g_idx]
            if mode == 'intra-camera':
                valid = qcamid == gcamid
            elif mode == 'inter-camera':
                valid = (qpid != gpid and qcamid == gcamid) or (qcamid != gcamid)
            elif mode == 'all':
                valid = True
            if valid:
                if only_show == 'pos' and qpid != gpid: continue
                if only_show == 'neg' and qpid == gpid: continue
                image_list.append(gimg_path)
                hit_list.append(qpid == gpid)
                text_list.append(distmat[q_idx, g_idx])
                rank_idx += 1
                if rank_idx > topk:
                    break

        counts += 1
        cat_imgs_to(image_list, hit_list, text_list, qdir)
        print(counts, qdir)