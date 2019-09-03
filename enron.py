#!/usr/bin/env python3
from argparse import ArgumentParser
from email import message_from_file
from glob import glob
from hashlib import md5
from os import makedirs
from os.path import isdir, join
from re import sub, IGNORECASE

import numpy as np
from gender_guesser.detector import Detector
from pandas import read_csv

from common import min_sequence_length, pad, sequence_length, tokenize

parser = ArgumentParser()
parser.add_argument('-d', '--data', default='data')
args = parser.parse_args()

data_path = args.data
dataset_path = join(data_path, 'raw', 'enron')
in_paths = glob(join(dataset_path, '**/*sent*/**'), recursive=True)
roles_path = join(data_path, 'raw', 'enron_roles.txt')
out_dir = join(data_path, 'proc', 'enron')
checksums = []
detector = Detector()
x = []
y = []
roles = read_csv(roles_path, sep='\t', skiprows=[139], header=None)

for path in in_paths:
    if isdir(path):
        continue
    with open(path, 'r', encoding='iso8859') as file:
        mail = message_from_file(file)
    if 'X-From' not in mail.keys():
        continue
    names = None
    if '@' in mail.get('From'):
        mail_addr = mail.get('From').split('@')[0]
        role = roles[roles[0] == mail_addr]
        if len(role) > 0:
            full_name = role[1].item().split('  ')[0]
            names = full_name.split(' ')
    if names is None:
        names = mail.get('X-From').split(' ')
    sender_first_name = names[0]
    gender = detector.get_gender(sender_first_name)
    if gender not in ['female', 'mostly_female', 'mostly_male', 'male']:  # ignore androgynous names etc.
        continue
    checksum = md5(mail.as_bytes()).hexdigest()
    if checksum in checksums:  # do not process duplicate mails
        continue
    text = mail.get_payload()
    if 'Forwarded by' in text:  # discard forwarded emails for obvious authorship disambiguation
        continue
    reply_indices = [text.find(indicator) for indicator in ['\nFrom:', '\nTo:', '- Original Message -', '-----------']]
    reply_indices = [index for index in reply_indices if index > -1]
    if len(reply_indices) > 0:  # remove older messages, signatures, etc.
        text = text[:min(reply_indices) - 5]
    lines = text.split('\n')
    non_quote_lines = [line for line in lines if not line.startswith('>')]  # remove quotes
    text = '\n'.join(non_quote_lines)
    for name in names:  # remove own name as indicator for gender (e.g. in signature)
        if len(name) > 2:
            text = sub(name, '', text, flags=IGNORECASE)
    text = text.replace('?', '')  # replace question mark, as it is also used for unknown characters and redacted text
    encoding = tokenize(text)
    if encoding.shape[0] <= min_sequence_length:
        continue
    if encoding.shape[0] < sequence_length:
        encoding = pad(encoding, sequence_length)
    x.append(encoding[:sequence_length])
    label = 1 if 'female' in gender else 0
    y.append(label)
    checksums.append(checksum)

x = np.vstack(x)
y = np.array(y)

makedirs(out_dir, exist_ok=True)
np.save(join(out_dir, 'x'), x)
np.save(join(out_dir, 'y'), y)
