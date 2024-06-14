import sys,os
import argparse
import tk_async_execute as tae

from src.Gui import Gui

# def main(optionsFlag,debug=False):
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',action = 'store_true',default=False)
    parser.add_argument('--options',default='0')
    args = parser.parse_args()
    optionsFlag = int(args.options)

    tae.start()
    gui = Gui(optionsFlag, debug=args.debug)

if __name__ == '__main__':

    main()
