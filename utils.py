# -*- coding: utf-8 -*-

import re
from nltk.corpus import stopwords
stopwords = stopwords.words('english')


def text_cleaner(text):

    text = re.sub(r"@\w*", " ", str(text)).strip()  # removing username
    text = re.sub(r'https?://[A-Za-z0-9./]+', " ",
                  str(text)).strip()  # removing links
    text = re.sub(r'[^a-zA-Z]', " ", str(text)).strip()  # removing sp_char
    tw = []

    for text in text.split():
        if text not in stopwords:
            if not tw.startwith('@') and tw != 'RT':
                tw.append(text)
    tw = re.sub(r"\s+", '-', ' '.join(tw))
    return tw
