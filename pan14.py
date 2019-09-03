#!/usr/bin/env python3
from argparse import ArgumentParser
from glob import glob
from os import makedirs
from os.path import basename, join, splitext

import numpy as np
from lxml import etree as ET

from common import interpret_html_texts

parser = ArgumentParser()
parser.add_argument('-d', '--data', default='data')
parser.add_argument('-l', '--lang', default='en', choices=['en', 'es'])
args = parser.parse_args()
language = 'spanish' if args.lang == 'es' else 'english'

text_dir = join(args.data, 'raw/pan14_' + args.lang)
text_paths = glob(text_dir + "/*.xml")
label_file = join(args.data, 'raw/pan14_' + args.lang + '/truth.txt')
out_dir = join(args.data, 'proc/pan14_' + args.lang)

x = []
y = []
with open(label_file, 'r') as file:
    lines = file.readlines()
    female_ids = [line.split(':::')[0] for line in lines if 'female' in line.lower()]

for path in text_paths:
    with open(path, 'r', encoding='ascii', errors='ignore') as file:
        blog = file.read()
    xml = ET.fromstring(blog)
    texts = xml.xpath('//document/text()')
    id = splitext(basename(path))[0]
    is_female = id in female_ids
    xs, ys = interpret_html_texts(texts, is_female)
    x.extend(xs)
    y.extend(ys)

makedirs(out_dir, exist_ok=True)
np.save(join(out_dir, 'x.npy'), np.vstack(x))
np.save(join(out_dir, 'y.npy'), np.array(y))
