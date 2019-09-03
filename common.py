from re import sub, MULTILINE, IGNORECASE
from string import ascii_lowercase, digits

import numpy as np
from lxml.etree import ParserError
from lxml.html import fromstring

sequence_length = 1000
min_sequence_length = 500  # data with lower than 500 chars is discarded
tokens = ascii_lowercase + '!"#%&\'()/:@^~ *+,-.;?' + digits
pad_char = '\0'
tokens = pad_char + tokens
char_to_int = {t: i for i, t in enumerate(tokens)}


def clean_text(text):
    text = text.lower()
    text = sub(r'http\S+', '', text, flags=MULTILINE)  # remove http urls
    text = sub(r'www\.[^ ]+', '', text)  # remove www.* urls
    text = text.replace('urllink', '')
    text = text.replace('(Taken with Instagram)', '')
    text = ' '.join(text.split())  # substitutes multiple whitespaces with single whitespace
    return text


def encode_text(text):
    text = clean_text(text)
    ints = [char_to_int[c] for c in text if c in tokens]
    encoding = one_hot_encode(ints, len(tokens))
    return encoding


def one_hot_encode(ints, number_classes):
    return np.eye(number_classes, dtype='uint8')[ints]


def pad(x, length):
    if len(x.shape) == 2:
        zeros = np.zeros((length, x.shape[1]), dtype=x.dtype)
    else:
        zeros = np.zeros((length,), dtype=x.dtype)
    zeros[:x.shape[0]] = x
    return zeros


def tokenize(text):
    text = clean_text(text)
    ints = [char_to_int[c] for c in text if c in tokens]
    return np.array(ints, dtype='uint8')


def read_xml(path):
    with open(path, 'r', encoding='ascii', errors='ignore') as file:
        blog = file.read()
    blog = "".join(c for c in blog if ord(c) < 128)
    xml = fromstring(blog)
    return xml


def interpret_html_texts(texts, is_female):
    x = []
    y = []
    for text in texts:
        text = text.replace('>;', '>')  # remove "extra" semicolon after named entities
        text = sub(r'<br\s*\/?>', ' ', text, flags=IGNORECASE)  # replace line breaks with a single white space
        text = sub(r'<p\s*>', ' <p>', text, flags=IGNORECASE)  # replace paragraph starts with a single white space
        text = sub(r'& nbsp;', ' ', text, flags=IGNORECASE)  # replace paragraph starts with a single white space
        if text.isspace():
            continue
        try:
            inner_text = fromstring(text).text_content()
        except ParserError as e:
            print('Blog post parsing error: ', e)
            continue
        encoding = tokenize(inner_text)
        if encoding.shape[0] <= min_sequence_length:
            continue
        if encoding.shape[0] < sequence_length:
            encoding = pad(encoding, sequence_length)
        label = 1 if is_female else 0
        x.append(encoding[:sequence_length])
        y.append(label)
    return x, y
