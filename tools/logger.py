class Logger:

    def __init__(self, log_file):
        '''/path/to/log_file.txt'''
        self.log_file = log_file

    def __call__(self, input):
        input = str(input)
        with open(self.log_file, 'a') as f:
            f.writelines(input+'\n')
        print(input)