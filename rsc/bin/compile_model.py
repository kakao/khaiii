#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
compile trained model for C/C++ decoder
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2019-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from argparse import ArgumentParser, Namespace
import json
import logging
import os
import pathlib
import pickle
from typing import Tuple

from khaiii.resource.resource import Resource


#############
# functions #
#############
def load_cfg_rsc(rsc_src: str, model_size: str) -> Tuple[Namespace, Resource]:
    """
    load config and resource from source directory
    Args:
        rsc_src:  source directory
        model_size:  model size (base|large)
    Returns:
        config
        resource
    """
    file_path = '{}/{}.config.json'.format(rsc_src, model_size)
    cfg_dic = json.load(open(file_path, 'r', encoding='UTF-8'))
    logging.info('config: %s', json.dumps(cfg_dic, indent=2))
    cfg = Namespace()
    for key, val in cfg_dic.items():
        setattr(cfg, key, val)
    cwd = os.path.realpath(os.getcwd())
    train_dir = os.path.realpath('{}/..'.format(rsc_src))
    if cwd != train_dir:
        os.chdir(train_dir)
    rsc = Resource(cfg)
    if cwd != train_dir:
        os.chdir(cwd)
    return cfg, rsc


def _write_config(cfg: Namespace, rsc: Resource, rsc_dir: str):
    """
    write config file
    Args:
        cfg:  config
        rsc:  resource
        rsc_dir:  target resource directory
    """
    cfg_dic = {}
    cfg_dic['window'] = cfg.window
    cfg_dic['vocab_size'] = len(rsc.vocab_in)
    cfg_dic['embed_dim'] = cfg.embed_dim
    cfg_dic['hidden_dim'] = cfg.hidden_dim
    cfg_dic['class_num'] = len(rsc.vocab_out)
    cfg_dic['conv_kernels'] = [2, 3, 4, 5]
    pathlib.Path(rsc_dir).mkdir(parents=True, exist_ok=True)
    config_json = '{}/config.json'.format(rsc_dir)
    with open(config_json, 'w', encoding='UTF-8') as fout:
        json.dump(cfg_dic, fout, indent=2, sort_keys=True)


def _write_embedding(embedding_data: dict, path: str):
    """
    write character embedding
    Args:
        rsc:  resource
        embedding_data:  pickled embedding data
        path:  output file path
    """
    with open(path, 'wb') as fout:
        embedding_data['chars'].tofile(fout)    # [input vocab(char)] * 4(wchar_t)
        for weight in embedding_data['weights']:
            weight.tofile(fout)    # [input vocab(char)] * embed_dim * 4(float)


def _write_linear(linear_data: dict, path: str):
    """
    write weight and bias with given layer name prefix
    Args:
        linear_data:  pickled linear layer data
        path:  output file path
    """
    with open(path, 'wb') as fout:
        for weight in linear_data['weight']:
            weight.tofile(fout)    # output * input * 4
        if 'bias' in linear_data:
            linear_data['bias'].tofile(fout)    # output * 4


def _write_conv(kernel_size: int, conv_data: dict, path: str):
    """
    write convolution module weight and bias with given layer name prefix and suffix(index)
    Args:
        kernel_size:  kernel size
        conv_data:  pickled 'convs' data
        path:  output file path
    """
    with open(path, 'wb') as fout:
        for channel in conv_data[kernel_size]['channel']:
            channel.tofile(fout)
        if 'bias' in conv_data[kernel_size]:
            conv_data[kernel_size]['bias'].tofile(fout)


def _write_data(data: dict, rsc_dir: str):
    """
    write data file
    Args:
        data:  pickled data
        rsc_dir:  target resource directory
    """
    _write_embedding(data['embedding'], '{}/embed.bin'.format(rsc_dir))

    for kernel in data['convs'].keys():
        # weight: [output chan(embed_dim)] * kernel * [input chan(embed_dim)] * 4
        # bias: [output chan] * 4
        _write_conv(kernel, data['convs'], '{}/conv.{}.fil'.format(rsc_dir, kernel))

    # weight: hidden_dim * [cnn layers * output chan(embed_dim)] * 4
    # bias: hidden_dim * 4
    _write_linear(data['conv2hidden'], '{}/cnv2hdn.lin'.format(rsc_dir))

    # weight: [output vocab(tag)] * hidden_dim * 4
    # bias: [output vocab(tag)] * 4
    _write_linear(data['hidden2tag'], '{}/hdn2tag.lin'.format(rsc_dir))


def run(args: Namespace):
    """
    run function which is the start point of program
    Args:
        args:  program arguments
    """
    cfg, rsc = load_cfg_rsc(args.rsc_src, args.model_size)
    data = pickle.load(open('{}/{}.model.pickle'.format(args.rsc_src, args.model_size), 'rb'))
    _write_config(cfg, rsc, args.rsc_dir)
    _write_data(data, args.rsc_dir)


########
# main #
########
def main():
    """
    main function processes only argument parsing
    """
    parser = ArgumentParser(description='part-of-speech tagger')
    parser.add_argument('--model-size', help='model size <default: base>',
                        metavar='SIZE', default='base')
    parser.add_argument('--rsc-src', help='source directory (model) <default: ./src>',
                        metavar='DIR', default='./src')
    parser.add_argument('--rsc-dir', help='target directory (output) <default: ./share/khaiii>',
                        metavar='DIR', default='./share/khaiii')
    parser.add_argument('--debug', help='enable debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    run(args)


if __name__ == '__main__':
    main()
