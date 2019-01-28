#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
apply patch to original Sejong corpus
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2019-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from argparse import ArgumentParser, Namespace
import logging
import os
import shutil

from khaiii.munjong import libpatch


#############
# functions #
#############
def run(args: Namespace):
    """
    run function which is the start point of program
    Args:
        args:  program arguments
    """
    if not os.path.exists(args.modified):
        logging.info('creating modified corpus dir: %s', args.modified)
        os.mkdir(args.modified)

    for name in sorted(os.listdir(args.original)):
        if not name.endswith('.txt'):
            continue
        org_path = '%s/%s' % (args.original, name)
        mod_path = '%s/%s' % (args.modified, name)
        patch_path = '%s/%s.patch' % (args.patch, name[:-len('.txt')])
        if os.path.exists(patch_path):
            logging.info('[%s] + [%s] = [%s]', org_path, patch_path, mod_path)
            libpatch.apply(org_path, args.org_enc, patch_path, mod_path, args.mod_enc)
        else:
            logging.info('[%s] = [%s]', org_path, mod_path)
            shutil.copyfile(org_path, mod_path)


########
# main #
########
def main():
    """
    main function processes only argument parsing
    """
    parser = ArgumentParser(description='apply patch to original Sejong corpus')
    parser.add_argument('-o', '--original', help='original corpus dir', metavar='DIR',
                        required=True)
    parser.add_argument('-p', '--patch', help='patch dir', metavar='DIR', required=True)
    parser.add_argument('-m', '--modified', help='modified corpus output dir', metavar='DIR',
                        required=True)
    parser.add_argument('--org-enc', help='original corpus encoding <default: UTF-16>',
                        metavar='ENCODING', default='UTF-16')
    parser.add_argument('--mod-enc', help='modified corpus encoding <default: UTF-8>',
                        metavar='ENCODING', default='UTF-8')
    parser.add_argument('--debug', help='enable debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    run(args)


if __name__ == '__main__':
    main()
