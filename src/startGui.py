import sys

from Gui import Gui

def main(optionsFlag):

    gui = Gui(optionsFlag)

if __name__ == '__main__':
    if len(sys.argv)>1:
        optionsFlag = int(sys.argv[1])
    else:
        optionsFlag = 0
    main(optionsFlag)