class bcolor:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Logger:
    def __init__(self,do_info=True,do_warn=True):
        self.do_info = do_info
        self.do_warn = do_warn
    def info(self,msg):
        if self.do_info:
            print(f"{bcolor.OKGREEN}INFO: {bcolor.ENDC}{msg}")
    def warn(self,msg):
        if self.do_warn:
            print(f"{bcolor.WARNING}WARNING: {bcolor.ENDC}{msg}")