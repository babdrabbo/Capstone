#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 19:31:35 2021

@author: basm
"""

#%% imports

import task
import numpy as np
from priors import priors
   
#%% Task 67e8384a
def solve_67e8384a(inimg):
    outimg = priors.img_create(inimg.shape[0]*2, inimg.shape[1]*2)
    
    overlay = inimg
    priors.img_overlay(outimg, overlay, pos=(0, 0))
    overlay = np.flipud(overlay)
    priors.img_overlay(outimg, overlay, pos=(inimg.shape[0], 0))
    overlay = np.fliplr(overlay)
    priors.img_overlay(outimg, overlay, pos=(inimg.shape[0], inimg.shape[1]))
    overlay = np.flipud(overlay)
    priors.img_overlay(outimg, overlay, pos=(0, inimg.shape[1]))
    
    return outimg

task.test_task('67e8384a', solve_67e8384a)

#%% Task 2013d3e2
def solve_2013d3e2(inimg):
    obj = priors.img_subimg(inimg, * priors.img_interest_area(inimg))
    return obj[:obj.shape[0]//2, :obj.shape[1]//2]

task.test_task('2013d3e2', solve_2013d3e2)

#%% Task 5ad4f10b
def solve_5ad4f10b(img):
    
    def _divs(n, m):
        return [1] if m==1 else (_divs(n, m-1) + ([m] if n % m == 0 else []))

    def divs(n):
        for x in _divs(n, n-1): yield x

    bgc = priors.img_major_color(img)
    colors = priors.img_colors(img)
    colors.remove(bgc)
    colors = sorted(list(map(lambda c: (c, priors.img_color_density(img, c)), colors)), key=lambda e: e[1], reverse=True)
    obj_color = colors[0][0]
    pigment = colors[1][0]

    fimg = priors.img_filter(img, obj_color)
    obj = priors.img_subimg(fimg, *priors.img_interest_area(fimg))

    tile_sides = list(set(list(divs(obj.shape[0])) + list(divs(obj.shape[1]))))
    new_obj = []
    for t in reversed(tile_sides):
        new_obj = priors.img_create(obj.shape[0]//t, obj.shape[1]//t)
        unicolor = True
        for i in range(0, obj.shape[0], t):
            for j in range(0, obj.shape[1], t):
                h = priors.img_hist(priors.img_subimg(obj, (i, j), (i+t-1, j+t-1)))
                unicolor = len(h) == 1
                if unicolor:
                    uc = list(h.keys())[0]
                    new_obj[(i//t), (j//t)] = pigment if uc == obj_color else bgc
                else: break
            if not unicolor: break
        if unicolor: break

    return new_obj

task.test_task('5ad4f10b', solve_5ad4f10b)

#%% Task c8cbb738
def solve_c8cbb738(img):
    bgc = priors.img_major_color(img)
    objs = priors.img_unicolor_objs(img)
    objs = list(map(lambda obj: priors.img_clear_color(obj, bgc), objs))
    objs_sizes = (obj.shape for obj in objs)
    combined_size = tuple(map(lambda t: max(t), zip(*objs_sizes)))
    outimg = priors.img_create(combined_size[0], combined_size[1], bgc)
    for obj in objs:
        centering_pos = tuple((np.array(outimg.shape) - np.array(obj.shape)) // 2)
        priors.img_overlay(outimg, obj, centering_pos)
    return outimg

task.test_task('c8cbb738', solve_c8cbb738)

#%% Task 681b3aeb
def solve_681b3aeb(img):
    
    def conv_overlay(img1, img2):
        for i in range(-img2.shape[0], img1.shape[0] + 1):
            for j in range(-img2.shape[1], img1.shape[1] + 1):
                index = np.array((i, j))
                f = np.where((index < 0), (img2.shape + index), (img1.shape - index))
                intersection = np.array(list(map(min, zip(img1.shape, img2.shape, f))))
                out_size = np.array(img1.shape) + np.array(img2.shape) - intersection
                outimg = priors.img_create(*out_size, color=-1)
                pos1 = tuple(np.where(index < 0, -index, 0))
                pos2 = tuple(np.where(index < 0, 0, index))
                priors.img_overlay(outimg, img1, pos1)
                priors.img_overlay(outimg, img2, pos2)
                yield outimg

    def does_match(img, objs):
        if(len(priors.img_hist(img).keys()) > 2):
            return False
        else:
            for obj in objs:
                colors = priors.img_hist(obj)
                colors.pop(-1, 0)
                obj_color = list(colors.keys())[0]
                fimg = priors.img_filter(img, color=obj_color, bg_color=-1)
                fobj = priors.img_subimg(fimg, *priors.img_interest_area(fimg, bgc=-1))
                if(not np.array_equal(obj, fobj)):
                    return False

        return True
    
    bgc = priors.img_major_color(img)
    objs = priors.img_unicolor_objs(img)
    objs = list(map(lambda obj: priors.img_clear_color(obj, bgc), objs))
    for im in conv_overlay(*objs):
        if(does_match(im, objs)):
            return im
        
task.test_task('681b3aeb', solve_681b3aeb)


#%% Task 6d75e8bb
def solve_6d75e8bb(img):
    outimg = np.copy(img)
    fill_color = 2
    bgc = priors.img_major_color(img)
    bb = priors.img_interest_area(img)
    obj = priors.img_subimg(img, *bb)
    obj = np.where(obj == bgc, fill_color, obj)
    priors.img_overlay(outimg, obj, bb[0])
    
    return outimg

task.test_task('6d75e8bb', solve_6d75e8bb)

#%% Test

task.test_task('6d75e8bb', solve_6d75e8bb)

