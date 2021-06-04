"""
@author:    Guan'an Wang
@contact:   guan.wang0706@gmail.com
"""

from lightreid.utils.tools import time_now

class Logging:

    def __init__(self, log_file, record_time=False):
        """
        Args:
            log_file(str): file to save log, e.g. /path/to/log_file.txt
            record_time(bool): record time if true
        """
        self.log_file = log_file
        self.record_time = record_time 

    def __call__(self, *args, **kwargs):
        line = ''
        # record inputs
        for value in args:
            line += '' + str(value) + ' '
        for key, value in kwargs.items():
            line += ' [' + str(key) + ': ' + str(value) + '] '
        # record time
        if self.record_time:
            line = "Time: " + time_now() + ' ' + line
        # save to file and print in terminal
        with open(self.log_file, 'a') as f:
            f.writelines(line+'\n')
        print(line)
