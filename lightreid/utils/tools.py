"""
@author:    Guan'an Wang
@contact:   guan.wang0706@gmail.com
"""

import time
import os

def time_now():
    '''return current time in format of 2000-01-01 12:01:01'''
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

def os_walk(folder_dir):
    for root, dirs, files in os.walk(folder_dir):
        files = sorted(files, reverse=True)
        dirs = sorted(dirs, reverse=True)
        return root, dirs, files