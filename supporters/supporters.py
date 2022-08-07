import sys


# classs to define clour shades for different terminal outputs
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# class to define transcript functionality with console and log file output
class Transcript(object):

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

# function to start console and log file transcription
def start(filename):
    """Start transcript, appending print output to given filename"""
    sys.stdout = Transcript(filename)


# function to stop console and log file transcription
def stop():
    """Stop transcript and return print functionality to normal"""
    sys.stdout.logfile.close()
    sys.stdout = sys.stdout.terminal