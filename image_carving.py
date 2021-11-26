# USAGE:
# python image_carving.py -im <input_image_path> -out <output_image_path> [-dy DY] [-dx DX] [-vis]

# Importing the required libraries
from scipy import ndimage as ndi
import warnings
from numba import jit
import argparse
import numpy as np
import cv2

warnings.filterwarnings("ignore")

############### UTILITY CODE #########################

def convert_type(image,to):
    if to=="uint8":
        converted = image.astype(np.uint8)
    elif to=="float64":
        converted = image.astype(np.float64)
    return converted


def visualize_util(vis):
    cv2.imshow("Visualizer", vis)
    cv2.waitKey(1)

def visualize(im, mask=None, should_rotate=False):
    vis = convert_type(im,"uint8")


    if mask is not None:
        condition = (mask == False)
        vis[np.asarray(condition).nonzero()] = np.array([255, 200, 180]) # seam visualization color (BGR)

    vis = rotate_image(vis, False)

    if not should_rotate:
        vis = rotate_image(vis, True)

    visualize_util(vis)
    return vis




def rotate_image(image, rightward):
    if rightward:
        return np.rot90(image, 1)    
    else:
        return np.rot90(image, 3)    
 

def resize(image, width):
    h, w, z = image.shape
    w = float(w)
    num = h * width 
    den = num / w
    den = int(den)
    dim = (width, den)
    return cv2.resize(image, dim)

###############  ENERGY FUNCTION  #########################

# Simple gradient magnitude energy map.
def energy_map(im):
    edge_filter = np.array([1, 0, -1])
    xgrad = ndi.convolve1d(im, edge_filter, axis=1, mode='wrap')
    ygrad = ndi.convolve1d(im, edge_filter, axis=0, mode='wrap')
    xgrad = xgrad**2
    ygrad = ygrad**2
    grad_mag = np.sqrt(np.sum(xgrad, axis=2) + np.sum(ygrad, axis=2))
    return grad_mag


##############  SEAM HELPER FUNCTIONS  ##########################


def add_seam_util(output, im, row, col, channel_num, p, bit):
    if bit == 0:
        output[row, col + 1:, channel_num] = im[row, col:, channel_num] 
        for col_iter in range(col):
            output[row, col_iter, channel_num] = im[row, col_iter, channel_num]
        output[row, col, channel_num] = p
    else:
        output[row, col + 1, channel_num] = p
        output[row, col, channel_num] = im[row, col, channel_num]
        output[row, col + 1:, channel_num] = im[row, col:, channel_num]

@jit
def add_seam(im, seam_idx):
    h, w, z = im.shape
    w = w + 1
    dim = 3
    size_tuple = (h,w,dim)
    output = np.zeros(size_tuple)
    row = h - 1 
    while row>=0:
        col = seam_idx[row]
        channel_num = 0 
        while channel_num < dim:
            if col != 0:
                p = np.average([im[row, col - 1, channel_num],im[row, col, channel_num]])
                add_seam_util(output, im, row, col, channel_num, p, 0)
            else:
                p = np.average([im[row, col, channel_num], im[row, col + 1, channel_num]])
                add_seam_util(output, im, row, col, channel_num, p, 1)
                
            channel_num += 1 
        row = row - 1 

    return output


@jit
def remove_seam(image, mask):
    h, w, z = image.shape
    boolmask_3D = np.stack([mask] * 3, axis=2)
    w = w - 1
    dim = 3
    size_tuple = (h, w, dim)
    final_image = image[boolmask_3D].reshape(size_tuple)
    return final_image 


def return_min_index(arg_one,arg_two,arg_three=float('inf')):
    arr = [arg_one,arg_two,arg_three]
    min_val = min(arr)
    index = arr.index(min_val)  
    return np.int64(index)

@jit
def get_minimum_seam(im):
    h, w, z = im.shape
    M = energy_map(im)

    backtrack = np.zeros_like(M, dtype=np.int)

    
    for i in range(1, h):
        j = 0
        idx = np.argmin(M[i - 1, j:j + 2])
        min_energy = M[i-1, idx + j]
        backtrack[i, j] = idx + j
        M[i, j] += min_energy
        for j in range(1, w):
            idx = np.argmin(M[i - 1, j - 1:j + 2])
            min_energy = M[i - 1, idx + j - 1]
            backtrack[i, j] = idx + j - 1

            M[i, j] += min_energy



    boolmask = np.ones((h, w), dtype=np.bool)
    col = np.argmin(M[-1])
    seam_idx = []

    row = h 
    while row>0:
        row = row - 1
        boolmask[row, col] = False
        seam_idx.append(col)
        col = backtrack[row, col]
        
    seam_idx = seam_idx[::-1]
    seam_idx = np.array(seam_idx)
    return seam_idx, boolmask


def seams_removal(image, num_remove, vis=False, should_rotate=False):
    while num_remove>0:
        num_remove -= 1
        __, mask = get_minimum_seam(image)
        if vis:
            visualize(image, mask, should_rotate=should_rotate)
            image = remove_seam(image, mask)
        else:
            image = remove_seam(image, mask)
    return image

def seams_insertion(image, num_add, should_visualize=False, should_rotate = False):
    seams_record = []
    temp_im = image.copy()
    
    oper = num_add
    while oper:
        seam_idx, boolmask = get_minimum_seam(temp_im)
        
        if should_visualize:
            visualize(temp_im, boolmask, should_rotate=should_rotate)
        
        seams_record.append(seam_idx)
        temp_im = remove_seam(temp_im, boolmask)
        oper = oper - 1

    seams_record = seams_record[::-1]

    to_add = num_add
    while to_add:

        to_add = to_add - 1
        seam = seams_record.pop(-1)

        image = add_seam(image, seam)
        if should_visualize:
            visualize(image, should_rotate=should_rotate)

        # update the remaining seam indices
        for remaining_seam in seams_record:
            condition = remaining_seam >= seam
            remaining_seam[np.asarray(condition).nonzero()] += 2
    return image

def seam_carve_x(final_image,vis):
    if dx>0:
        function_to_call = seams_insertion
    else:
        function_to_call = seams_removal
    final_image = function_to_call(final_image, abs(dx), vis)
    return final_image

def seam_carve_y(final_image, vis):
    if dy < 0:
        function_to_call = seams_removal
    else:   
        function_to_call = seams_insertion
    final_image = rotate_image(final_image, True)
    final_image = function_to_call(final_image, abs(dy), vis, should_rotate=True)
    final_image = rotate_image(final_image, False)
    return final_image

def seam_carve(image, dy, dx, vis=False):
    image = convert_type(image,"float64")
    h, w, z = image.shape
    assert h + dy > 0 and w + dx > 0 and dy <= h and dx <= w
    final_image = image
    if dx:
        final_image = seam_carve_x(final_image , vis)
    if dy:
        final_image = seam_carve_y(final_image , vis)
    return final_image


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-vis", action='store_true', help="")
    ap.add_argument("-dy", help="", type=int, default=0)
    ap.add_argument("-im", required=True, help="")
    ap.add_argument("-out", required=True, help="")
    ap.add_argument("-dx", type=int, default=0,help="")
    args = vars(ap.parse_args())
    image = cv2.imread(args["im"])
    dy, dx = args["dy"], args["dx"]
    should_visualize = args["vis"]
    assert dy is not None and dx is not None
    final_image = seam_carve(image, dy, dx, should_visualize)
    cv2.imwrite(args["out"], final_image)
