"""
@author:    Guan'an Wang
@contact:   guan.wang0706@gmail.com
"""

class Logging:

    def __init__(self, log_file):
        '''/path/to/log_file.txt'''
        self.log_file = log_file

    def __call__(self, *args, **kwargs):
        line = ''
        for value in args:
            line += '' + str(value) + ' '
        for key, value in kwargs.items():
            line += ' [' + str(key) + ': ' + str(value) + '] '
        with open(self.log_file, 'a') as f:
            f.writelines(line+'\n')
        print(line)
