#!/usr/bin/env python3
from argparse import ArgumentParser
from glob import glob
from os import makedirs
from os.path import basename, join

import numpy as np

from common import interpret_html_texts, read_xml

parser = ArgumentParser()
parser.add_argument('-d', '--data', default='data')
parser.add_argument('-l', '--lang', default='en', choices=['en', 'es'])
parser.add_argument('-m', '--mode', default='tr', choices=['tr', 'te'])
args = parser.parse_args()

text_dir = join(args.data, 'raw/pan13_' + args.mode, args.lang)
text_paths = glob(text_dir + "/*.xml")
label_file = join(args.data, 'raw/pan13_' + args.mode + '/truth-' + args.lang + '.txt')
out_dir = join(args.data, 'proc/pan13_' + args.mode + '_' + args.lang + '/')

x = []
y = []
with open(label_file, 'r') as file:
    lines = file.readlines()
    female_ids = [line.split(':::')[0] for line in lines if 'female' in line]
for path in text_paths:
    xml = read_xml(path)
    texts = xml.xpath(".//conversation/text()")
    id = basename(path).split('_')[0]
    is_female = id in female_ids
    xs, ys = interpret_html_texts(texts, is_female)
    x.extend(xs)
    y.extend(ys)

makedirs(out_dir, exist_ok=True)
np.save(join(out_dir, 'x.npy'), np.vstack(x))
np.save(join(out_dir, 'y.npy'), np.array(y))
