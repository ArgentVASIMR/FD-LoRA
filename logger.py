class bcolor:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
def color_print(type,msg,color,newline=True):
        if newline:
            print(f"{color}{type}:{bcolor.ENDC} {msg}")
        else:
            print(f"{color}{type}:{bcolor.ENDC} {msg}",end='')
class Logger:
    def __init__(self,do_info=True,do_warn=True,do_debug=False):
        self.do_info = do_info
        self.do_warn = do_warn
        self.do_debug = do_debug
        print(f"\n-- antrogent logger --")
    
    def info(self,msg,newline=True):
        if self.do_info:
            color_print('INFO',msg,bcolor.OKGREEN,newline)
    def warn(self,msg):
        if self.do_warn:
            color_print('WARNING',msg,bcolor.WARNING)
    def debug(self,msg,newline=True):
        if self.do_debug:
            color_print('DEBUG',msg,bcolor.OKBLUE,newline)
    def error(self,msg):
        color_print('ERROR',msg,bcolor.FAIL)
        exit()
    def debug_list(self,msg:list):
        if self.do_debug:
            for i in msg:
                color_print('DEBUG',i,bcolor.OKBLUE,newline=False)
            print()