#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
train part-of-speech model from data set
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2019-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from argparse import ArgumentParser, Namespace
import logging

from khaiii.train.trainer import Trainer


#############
# functions #
#############
def run(args: Namespace):
    """
    run function which is the start point of program
    Args:
        args:  program arguments (config)
    """
    Trainer(args).train()


########
# main #
########
def main():
    """
    main function processes only argument parsing
    """
    parser = ArgumentParser(description='train model from data')
    parser.add_argument('-i', '--in-pfx', help='input data path prefix', metavar='NAME',
                        required=True)
    parser.add_argument('--rsc-src', help='resource source dir <default: ../rsc/src>',
                        metavar='DIR', default='../rsc/src')
    parser.add_argument('--logdir', help='tensorboard log dir <default: ./logdir>', metavar='DIR',
                        default='./logdir')
    parser.add_argument('--window', help='left/right character window length <default: 4>',
                        metavar='INT', type=int, default=4)
    parser.add_argument('--spc-dropout', help='space(word delimiter) dropout rate <default: 0.1>',
                        metavar='REAL', type=float, default=0.1)
    parser.add_argument('--cutoff', help='cutoff <default: 2>', metavar='INT', type=int, default=2)
    parser.add_argument('--embed-dim', help='embedding dimension <default: 35>', metavar='INT',
                        type=int, default=35)
    parser.add_argument('--learning-rate', help='learning rate <default: 0.001>', metavar='REAL',
                        type=float, default=0.001)
    parser.add_argument('--lr-decay', help='learning rate decay <default: 0.9>', metavar='REAL',
                        type=float, default=0.9)
    parser.add_argument('--batch-size', help='batch size <default: 500>', metavar='INT', type=int,
                        default=500)
    parser.add_argument('--patience', help='maximum patience count to revert model <default: 10>',
                        metavar='INT', type=int, default=10)
    parser.add_argument('--gpu-num', help='GPU number to use <default: -1 for CPU>', metavar='INT',
                        type=int, default=-1)
    parser.add_argument('--debug', help='enable debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    run(args)


if __name__ == '__main__':
    main()
