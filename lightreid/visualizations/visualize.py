"""
@author:    Guan'an Wang
@contact:   guan.wang0706@gmail.com
"""

import numpy as np
import os.path as osp
import os
from PIL import Image, ImageOps, ImageDraw


def make_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print('Successfully make dirs: {}'.format(dir))
    else:
        print('Existed dirs: {}'.format(dir))


def visualize_ranked_results(distmat, dataset, save_dir='./vis-results/', sort='ascend', topk=20, mode='inter-camera', show='all', display_score=False):
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
        show(string): pos/neg/all
            pos onlu show those true matched images
            neg only show those false matched images
            all show both
    """
    num_q, num_g = distmat.shape

    print('Visualizing top-{} ranks'.format(topk))
    print('# query: {}\n# gallery {}'.format(num_q, num_g))
    print('Saving images to "{}"'.format(save_dir))

    query, gallery = dataset
    assert num_q == len(query)
    assert num_g == len(gallery)
    assert sort in ['ascend', 'descend']
    assert mode in ['intra-camera', 'inter-camera', 'all']
    assert show in ['pos', 'neg', 'all']

    if sort == 'ascend':
        indices = np.argsort(distmat, axis=1)
    else:
        indices = np.argsort(distmat, axis=1)[:, ::-1]
    os.makedirs(save_dir, exist_ok=True)

    def cat_imgs_to(image_list, hit_list, text_list, target_dir):

        images = []
        for img, hit, text in zip(image_list, hit_list, text_list):
            img = Image.open(img).resize((64, 128))
            d = ImageDraw.Draw(img)
            if display_score:
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
                if show == 'pos' and qpid != gpid: continue
                if show == 'neg' and qpid == gpid: continue
                image_list.append(gimg_path)
                hit_list.append(qpid == gpid)
                text_list.append(distmat[q_idx, g_idx])
                rank_idx += 1
                if rank_idx > topk:
                    break

        counts += 1
        cat_imgs_to(image_list, hit_list, text_list, qdir)
        print(counts, qdir)