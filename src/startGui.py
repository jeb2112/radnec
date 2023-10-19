import sys
import argparse

from Gui import Gui

def main(optionsFlag,debug=False):

    gui = Gui(optionsFlag, debug=debug)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',action = 'store_true',default=False)
    parser.add_argument('--options',default='0')
    args = parser.parse_args()
    optionsFlag = int(args.options)

    main(optionsFlag,debug=args.debug)