# -*- coding:utf8 -*-
# File   : rect.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/22/17
# 
# This file is part of TensorArtist


def ltwh2tblr(l, t, w, h):
    return t, t + h - 1, l, l + w - 1


def tblr2ltwh(t, b, l, r):
    return l, t, b - t + 1, r - l + 1


def area(rect):
    return max(0, rect[1] - rect[0] + 1) * max(0, rect[3] - rect[2] + 1)


def intersection(i, j):
    return max(i[0], j[0]), min(i[1], j[1]), max(i[2], j[2]), min(i[3], j[3])


def iou(i, j):
    inter = area(intersection(i, j))
    return float(inter) / (area(i) + area(j) - inter)


def ioself(i, j):
    inter = area(intersection(i, j))
    return float(inter) / area(i)


def nms(rect_list, threshold, sort_key_func=lambda x: x[1]):
    rect_list = sorted(rect_list, key=sort_key_func, reverse=True)
    length = len(rect_list)
    ignored = [False] * length

    for i in range(length):
        if not ignored[i]:
            for j in range(i + 1, length):
                if iou(rect_list[i]['box'], rect_list[j]['box']) > threshold:
                    rect_list[i]['confidence'] += rect_list[j]['confidence']
                    ignored[j] = True

    result = list()
    for i in range(length):
        if not ignored[i]:
            result.append(rect_list[i])
    return result

