import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from .pose_config import config as pose_config
from .pose_resnet import get_pose_net

import os
import cv2
import math
import numpy as np


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(batch_heatmaps, pose_processing=True):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if pose_processing:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array([hm[py][px+1] - hm[py][px-1],
                                     hm[py+1][px]-hm[py-1][px]])
                    coords[n][p] += np.sign(diff) * .25

    return coords, maxvals


def im_preprocessing_for_skeleton(im):
    '''
    preprocessing the input image
    @param 
        im: opencv image (bgr)
    @return
        im
        (h_scale,w_scale) the scale of height and width
    '''
    o_h,o_w,_ = im.shape
    width, height = pose_config.MODEL.IMAGE_SIZE
    im = cv2.resize(im,(width,height))
    
    h_scale = o_h / height
    w_scale = o_w / width

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im/255.0
    im -= np.array([0.485, 0.456, 0.406]) # mean: rgb
    im /= np.array([0.229, 0.224, 0.225]) # std: rgb

    im = np.transpose(im,(2,0,1)) # hwc --> chw

    return im,(h_scale,w_scale)


def get_skeleton(model, inputs):
    '''
    get skeleton of inputs
    @param
        model: the torch model of skeleton
        input: ndarray of input images, the dim should be nchw
    @return
        predict joints, joints shape (batchsize, num_joints, 2(u,v))
        maxvalue of joints on response heatmap, shape(batchsize, num_joints)
    '''
    # assert inputs.ndim==4,'input dim should be nchw'
    n,c,h,w = inputs.shape
    # inputs = torch.from_numpy(inputs)
    # inputs = inputs.type(torch.FloatTensor)
    # print("----",type(inputs))
    # if len(pose_config.GPUS.split(','))>0:
    #     # print("use gpu")
    #     model = model.cuda()
    #     inputs = inputs.cuda()
    model.eval()
    with torch.no_grad():
        heatmaps = model(inputs)
        
        # get scale of width and height 
        _,_,h1,w1 = heatmaps.size()
        h_scale = h/h1
        w_scale = w/w1

        #  do something of the heatmap
        heatmaps = heatmaps.cpu().numpy()

        # get the u,v of each joints and its max response value
        joints,maxvalue = get_final_preds(heatmaps) # joints shape (batchsize, num_joints, 2(u,v))

        # re-calculate joints location on input images
        joints[:,:,0] *= w_scale
        joints[:,:,1] *= h_scale

        return joints, maxvalue


def filter_joints_by_threshold(maxvalue, threshold=0.3):
    '''
    filter joints by maxvalue and threshold 
    @param
        maxvalue of joints on response heatmap, shape(num_joints,)
        threshold, minimum value of confidence of joints
    @return 
        joints index, which joints is beyound uppon threshold
        joints confidence
    '''
    joint_indexs = np.where(maxvalue>threshold)
    confidence = maxvalue[joint_indexs]

    return joint_indexs, confidence


def crop_by_joints(image, joints, expand_pixels=(2,4,2,2)):
    '''
    get bounding box by selected joints, expand the box by expand_pixels
    @param
        image, opencv image
        joints, selected joints, shape(num_joints,2(u,v))
        expand_pixel: (bottom,top,left,right) pixels to expand the joints based box
    @return 
        croped image 
    '''
    if len(joints)<=0:
        # when no joint given, we just return the input image
        return image
    bottom,top,left,right = expand_pixels
    height,width,_ = image.shape
    min_u = int(np.min(joints[:,0]) - left)
    min_v = int(np.min(joints[:,1]) - top)
    max_u = int(np.max(joints[:,0]) + right)
    max_v = int(np.max(joints[:,1]) + bottom)
    
    if min_u<0:
        min_u = 0 
    if min_v<0:
        min_v = 0 
    if max_u > width:
        max_u = width 
    if max_v > height:
        max_v = height 

    croped = image[min_v:max_v,min_u:max_u]

    return croped


def crop_query(model, query, threshold=0.3, expand_pixels=(2,4,2,2)):
    '''
    crop images by the specific predicted skeleton
    @param
        model: the torch model of skeleton
        query: query image
        threshold: minimum value of confidence of joints
        expand_pixel: (bottom,top,left,right) pixels to expand the joints based box
    @return
        croped_query: croped query image
        joint_index: joints index of query image, the occluded joints has been filter out
    '''
    query_h, query_w,_ = query.shape
    query_input, query_scale = im_preprocessing_for_skeleton(query)
    query_scale_h, query_scale_w = query_scale

    query_input = np.expand_dims(query_input, axis=0)
    # get 2d joint on resized input image
    joint, maxvalue = get_skeleton(model,query_input)
    joint_query = joint[0]
    maxvalue_query = maxvalue[0,:,0]
    # get final joint on query image
    joint_query[:,0] *= query_scale_w
    joint_query[:,1] *= query_scale_h
    
    # only for test purpose
    # add the points on the original images
    # query = add_joints_to_image(query,joint_query,maxvalue_query)

    # filter out the occuluded joints
    joint_index, confidence = filter_joints_by_threshold(maxvalue_query,threshold)

    # crop query image
    selected_query_joints = joint_query[joint_index]
    croped_query = crop_by_joints(query,selected_query_joints,expand_pixels)

    return croped_query, joint_index


def crop_gallery(model, gallery,joint_index, expand_pixels=(2,4,2,2)):
    '''
    crop the gallery image by joint_index of query image
    @param
        model: the torch model of skeleton
        gallery: gallery image
        joint_index: joints index of query image
        expand_pixel: (bottom,top,left,right) pixels to expand the joints based box
    @return 
        croped gallery image
    '''
    gallery_h, gallery_w,_ = gallery.shape
    gallery_input, gallery_scale = im_preprocessing_for_skeleton(gallery)
    gallery_scale_h, gallery_scale_w = gallery_scale

    gallery_input = np.expand_dims(gallery_input, axis=0)
    # get 2d joint on resized input image
    joint, maxvalue = get_skeleton(model,gallery_input)
    joint_gallery = joint[0]
    maxvalue_gallery = maxvalue[0]

    # get final joint on gallery image
    joint_gallery[:,0] *= gallery_scale_w
    joint_gallery[:,1] *= gallery_scale_h

    # only for test purpose
    # add the points on the original images
    # gallery = add_joints_to_image(gallery,joint_gallery,maxvalue_gallery)

    selected_gallery_joints = joint_gallery[joint_index]
    croped_gallery = crop_by_joints(gallery,selected_gallery_joints,expand_pixels)

    return croped_gallery


def crop_query_galley(model, query_image, gallery_images,threshold=0.3, expand_pixels=(2,4,2,2)):
    '''
    crop the query and gallery
    @param
        model: the torch model of skeleton
        query_image: as is, ndarray of image, (hwc,bgr) 
        gallery_images: list of images(ndarray, hwc,bgr)
    @return 
        croped_query: croped image, ndarray
        croped_gallery: croped gallery images, list of ndarray
    '''
    croped_query, joint_index = crop_query(model, query_image, threshold=threshold, expand_pixels=expand_pixels )
    croped_gallery = {}
    for i in range(len(gallery_images)):
        gallery_image = gallery_images[i]
        croped_gallery_image = crop_gallery(model,gallery_image,joint_index,expand_pixels)
        croped_gallery[i] = croped_gallery_image

    return croped_query, croped_gallery


def add_joints_to_image(image,joint_im,maxvalue_im):
    '''
    add joint and its maxvalue to the image and save to the disk
    '''
    colors = [(0,0,255),(255,0,0),(0,255,0),(100,100,100),(25,25,25),(200,200,200),(233,233,234),(111,111,111),(66,66,66)]
    joints_all = []
    new_image = image.copy()
    if len(joint_im)==16:
        group1 = [0,1,2,3,4,5]
        group2 = [10,11,12,13,14,15]
        group3 = [6,7,8,9]
        group4 = []
        group5 = []
        group6 = []
        group7 = []
        joints_all = [group1,group2,group3,group4,group5,group6,group7]
    elif len(joint_im)==17:
        group1 = [0,1,2,3,4]
        group2 = [5,6,11,12]
        group3 = [7,8,9,10]
        group4 = [13,14,15,16]
        group5 = []
        group6 = []
        group7 = []
        joints_all = [group1,group2,group3,group4,group5,group6,group7]
    for i in range(len(joints_all)):
        group = joints_all[i]
        for j in group:
            u = int(joint_im[j,0])
            v = int(joint_im[j,1])
            new_image = cv2.circle(new_image,(u,v),2,colors[i],-1)
            new_image = cv2.putText(new_image,'{%.2f}'%maxvalue_im[j],(u+1,v-1),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,0,255),1)
    
    return new_image

def test_joints(model,image_path,output_dir):
    print("output_dir:",output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("===========make directory:",output_dir,"================")
    image_name = image_path.split("/")[-1]
    out_name = os.path.join(output_dir,image_name)
    print(out_name)
    image = cv2.imread(image_path)
    im_h, im_w,_ = image.shape
    im_input, im_scale = im_preprocessing_for_skeleton(image)
   
    im_scale_h, im_scale_w = im_scale
    # print(im_input.shape,im_h,im_w,im_scale)

    im_input = np.expand_dims(im_input, axis=0)
    # get 2d joint on resized input image
    joint, maxvalue = get_skeleton(model,im_input)
    joint_im = joint[0]
    maxvalue_im = maxvalue[0,:,0]
    # print(joint_im.shape,maxvalue_im.shape)
    # get final joint on im image
    joint_im[:,0] *= im_scale_w
    joint_im[:,1] *= im_scale_h

    image = cv2.resize(image,(im_w*4,im_h*4))
    joint_im *= 4
    new_image = add_joints_to_image(image,joint_im,maxvalue_im)
    cv2.imwrite(out_name,new_image)

def test_crop_image(model,query_path,gallery_paths,output_dir):
    print("output_dir:",output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("===========make directory:",output_dir,"================")
    query_image = cv2.imread(query_path)
    gallery_images = []
    for i in range(len(gallery_paths)):
        gallery_image = cv2.imread(gallery_paths[i])
        gallery_images.append(gallery_image)
    croped_query,croped_gallerys = crop_query_galley(model,query_image,gallery_images,expand_pixels=(2,4,1000,1000))
    
    h,w,_ = query_image.shape
    h1,w1,_ = croped_query.shape
    im = np.zeros((h,w*2+10,3))
    im[0:h,0:w,:] = query_image
    im[0:h1,w+10:w1+w+10,:] = croped_query
    cv2.imwrite(os.path.join(output_dir,'000_query.jpg'),im)
    # cv2.imwrite(os.path.join(output_dir,'query_image.jpg'),query_image)
    # cv2.imwrite(os.path.join(output_dir,'croped_query.jpg'),croped_query)

    for i in range(len(gallery_paths)):
        h,w,_ = gallery_images[i].shape
        h1,w1,_ = croped_gallerys[i].shape
        im = np.zeros((h,w*2+10,3))
        im[0:h,0:w,:] = gallery_images[i]
        im[0:h1,w+10:w1+w+10,:] = croped_gallerys[i]
        cv2.imwrite(os.path.join(output_dir,'gallery_image_{}.jpg'.format(i)),im)
        # cv2.imwrite(os.path.join(output_dir,'gallery_image_{}.jpg'.format(i)),gallery_images[i])
        # cv2.imwrite(os.path.join(output_dir,'croped_gallery{}.jpg'.format(i)),croped_gallerys[i])
