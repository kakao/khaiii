#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
compile trained model for C/C++ decoder
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2018-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
import argparse
from array import array
import json
import logging
import os
import pathlib
import re

import torch

from resource import Resource    # pylint: disable=wrong-import-order


#############
# functions #
#############
def _load_cfg_rsc(rsc_src, model_size):
    """
    load config and resource from source directory
    Args:
        rsc_src:  source directory
        model_size:  model size (base|large)
    Returns:
        (config, resource) pair
    """
    file_path = '{}/{}.config.json'.format(rsc_src, model_size)
    cfg_dic = json.load(open(file_path, 'r', encoding='UTF-8'))
    logging.info('config: %s', json.dumps(cfg_dic, indent=2))
    cfg = argparse.Namespace()
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


def _validate_state_dict(cfg, rsc, state_dict):
    """
    validate state dict with config and resource
    Args:
        cfg:  config
        rsc:  resource
        state_dict:  state dic
    Returns:
        whether is valid or not
    """
    def _assert(expected, actual, msg):
        """
        assert expected equals actual. if not raise exception
        Args:
            expected:  expected value
            actual:  actual value
            msg:  exception message if both are not equal
        """
        if expected != actual:
            raise ValueError((msg + ': expected: {}, but actual: {}').format(expected, actual))

    for name, tensor in state_dict.items():
        logging.info('%s: %s', name, tensor.size())
        if name == 'embedder.embedding.weight':
            _assert(len(rsc.vocab_in), tensor.size(0), 'invalid input vocab')
            _assert(cfg.embed_dim, tensor.size(1), 'invalid embedding dim')
        if re.match(r'convs\..\.weight', name):
            _assert(cfg.embed_dim, tensor.size(1), 'invalid conv input channel dim')
        if name == 'hidden2tag.bias':
            _assert(len(rsc.vocab_out), tensor.size(0), 'invalid output bias dim')


def _write_config(cfg, rsc, rsc_dir):
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


def _write_embedding(rsc, state_dict, fout):
    """
    write character embedding
    Args:
        rsc:  resource
        state_dict:  state dictionary of model
        fout:  output file
    """
    chars = [0, ] * 5    # 5 special characters
    for idx in range(5, len(rsc.vocab_in)):
        chars.append(ord(rsc.vocab_in[idx]))
    array('i', chars).tofile(fout)    # [input vocab(char)] * 4(wchar_t)

    embedding = state_dict['embedder.embedding.weight']
    for row in embedding:
        array('f', row).tofile(fout)    # [input vocab(char)] * embed_dim * 4(float)


def _write_linear(name_pfx, state_dict, path):
    """
    write weight and bias with given layer name prefix
    Args:
        name_pfx:  layer name prefix
        state_dict:  state dictionary of model
        path:  output file path
    """
    with open(path, 'wb') as fout:
        weight = state_dict['{}.weight'.format(name_pfx)]
        for row in weight:
            array('f', row).tofile(fout)    # output * input * 4
        bias_name = '{}.bias'.format(name_pfx)
        if bias_name in state_dict:
            bias = state_dict[bias_name]
            array('f', bias).tofile(fout)    # output * 4


def _write_conv(name_pfx, kernel_size, state_dict, path):
    """
    write convolution module weight and bias with given layer name prefix and suffix(index)
    Args:
        name_pfx:  layer name prefix
        kernel_size:  kernel size
        state_dict:  state dictionary of model
        path:  output file path
    """
    with open(path, 'wb') as fout:
        weight = state_dict['{}.{}.weight'.format(name_pfx, kernel_size - 2)]
        weight_t = weight.transpose(1, 2)    # [output chan] * [input chan] * kernel
        #                                      => [output chan] * kernel * [input chan]
        for ch_out in weight_t:
            array('f', ch_out.contiguous().view(-1)).tofile(fout)
        bias_name = '{}.{}.bias'.format(name_pfx, kernel_size - 2)
        if bias_name in state_dict:
            bias = state_dict[bias_name]
            array('f', bias).tofile(fout)    # [output chan] * 4


def _write_data(rsc, state_dict, rsc_dir):
    """
    write data file
    Args:
        rsc:  resource
        state_dict:  state dictionary of model
        rsc_dir:  target resource directory
    """
    with open('{}/embed.bin'.format(rsc_dir), 'wb') as fout:
        # key: [input vocab(char)] * 4(float)
        # val: [input vocab(char)] * embed_dim * 4(float)
        _write_embedding(rsc, state_dict, fout)

    for kernel in range(2, 6):
        # weight: [output chan(embed_dim)] * kernel * [input chan(embed_dim)] * 4
        # bias: [output chan] * 4
        _write_conv('convs', kernel, state_dict, '{}/conv.{}.fil'.format(rsc_dir, kernel))
    # weight: hidden_dim * [cnn layers * output chan(embed_dim)] * 4
    # bias: hidden_dim * 4
    _write_linear('conv2hidden', state_dict, '{}/cnv2hdn.lin'.format(rsc_dir))

    # weight: [output vocab(tag)] * hidden_dim * 4
    # bias: [output vocab(tag)] * 4
    _write_linear('hidden2tag', state_dict, '{}/hdn2tag.lin'.format(rsc_dir))


def run(args):
    """
    run function which is the start point of program
    Args:
        args:  program arguments
    """
    cfg, rsc = _load_cfg_rsc(args.rsc_src, args.model_size)
    state_dict = torch.load('{}/{}.model.state'.format(args.rsc_src, args.model_size),
                            map_location=lambda storage, loc: storage)
    _validate_state_dict(cfg, rsc, state_dict)
    _write_config(cfg, rsc, args.rsc_dir)
    _write_data(rsc, state_dict, args.rsc_dir)


########
# main #
########
def main():
    """
    main function processes only argument parsing
    """
    parser = argparse.ArgumentParser(description='part-of-speech tagger')
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
