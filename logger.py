class bcolor:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
def color_print(msg,color,newline=True):
        if newline:
            print(f"{color}{msg}{bcolor.ENDC}")
        else:
            print(f"{color}{msg}{bcolor.ENDC}",end='')
class Logger:
    def __init__(self,do_info=True,do_warn=True,do_debug=False):
        self.do_info = do_info
        self.do_warn = do_warn
        self.do_debug = do_debug
    
    def info(self,msg,newline=True):
        if self.do_info:
            color_print(msg,bcolor.OKGREEN,newline)
    def warn(self,msg):
        if self.do_warn:
            color_print(msg,bcolor.WARNING)
    def debug(self,msg,newline=True):
        if self.do_debug:
            color_print(msg,bcolor.OKBLUE,newline)
    def error(self,msg):
        color_print(msg,bcolor.FAIL)
        exit