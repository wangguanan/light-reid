
class Logger:

    def __init__(self, logger_path):
        self.logger_path = logger_path

    def __call__(self, input):
        input = str(input)
        with open(self.logger_path, 'a') as f:
            f.writelines(input+'\n')
        print(input)