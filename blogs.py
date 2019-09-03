#!/usr/bin/env python3
from argparse import ArgumentParser
from glob import glob
from os import makedirs
from os.path import join

import numpy as np

from common import min_sequence_length, pad, sequence_length, read_xml, tokenize

parser = ArgumentParser()
parser.add_argument('-d', '--data', default='data')
args = parser.parse_args()

in_dir = join(args.data, 'raw/blogs/')
in_paths = glob(in_dir + "*xml")
out_dir = join(args.data, 'proc/blogs/')

x = []
y = []
for path in in_paths:
    if 'male' not in path:
        continue
    xml = read_xml(path)
    posts = xml.xpath(".//post/text()")
    is_female = 'female' in path
    for post in posts:
        encoding = tokenize(post)
        if encoding.shape[0] <= min_sequence_length:
            continue
        if encoding.shape[0] < sequence_length:
            encoding = pad(encoding, sequence_length)
        label = 1 if is_female else 0
        x.append(encoding[:sequence_length])
        y.append(label)

makedirs(out_dir, exist_ok=True)
np.save(join(out_dir, 'x.npy'), np.vstack(x))
np.save(join(out_dir, 'y.npy'), np.array(y))
