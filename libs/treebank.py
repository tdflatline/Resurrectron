# Natural Language Toolkit: Tokenizers
#
# Copyright (C) 2001-2010 NLTK Project
# Author: Edward Loper <edloper@gradient.cis.upenn.edu>
# URL: <http://nltk.sourceforge.net>
# For license information, see LICENSE.TXT

"""
A regular-expression based word tokenizer that tokenizes sentences
using the conventions used by the Penn Treebank.
"""

import curses
import re
from nltk.tokenize.api import *

######################################################################
#{ Regexp-based treebank tokenizer
######################################################################
# (n.b., this isn't derived from RegexpTokenizer)

class TreebankWordTokenizer(TokenizerI):
    """
    A word tokenizer that tokenizes sentences using the conventions
    used by the Penn Treebank.  Contractions, such as "can't", are
    split in to two tokens.  E.g.:

      - can't S{->} ca n't
      - he'll S{->} he 'll
      - weren't S{-} were n't

    This tokenizer assumes that the text has already been segmented into
    sentences.  Any periods -- apart from those at the end of a string --
    are assumed to be part of the word they are attached to (e.g. for
    abbreviations, etc), and are not separately tokenized. 
    """
    # List of contractions adapted from Robert MacIntyre's tokenizer.
    CONTRACTIONS2 = [re.compile(r"(?i)(.)('ll|'re|'ve|n't|'s|'m|'d)\b"),
                     re.compile(r"(?i)\b(Mor)('n)\b"),
                     re.compile(r"(?i)\b(D)('ye)\b"),
                     re.compile(r"(?i)\b(up)(2)\b"),
                     re.compile(r"(?i)\b(Gim)(me)\b"),
                     re.compile(r"(?i)\b(can)(not)\b"),
                     re.compile(r"(?i)\b(Gon)(na)\b"),
                     re.compile(r"(?i)\b(Got)(ta)\b"),
                     re.compile(r"(?i)\b(Get)(cha)\b"),
                     re.compile(r"(?i)\b(Get)(ya)\b"),
                     re.compile(r"(?i)\b(Wan)(na)\b"),
                     re.compile(r"(?i)\b(Lem)(me)\b"),
                     re.compile(r"(?i)\b(You)(d)\b"),
                     re.compile(r"(?i)\b(You)(ll)\b"),
                     re.compile(r"(?i)\b(I)(d)\b"),
                     re.compile(r"(?i)\b(You)(re)\b"),
                     re.compile(r"(?i)\b(You)(re)\b"),
                     re.compile(r"(?i)\b(You)(ve)\b"),
                     re.compile(r"(?i)\b(ca)(nt)\b"),
                     re.compile(r"(?i)\b(T)(is)\b"),
                     re.compile(r"(?i)\b(T)(was)\b"),
                     re.compile(r"(?i)\b(I)(ve)\b"),
                     re.compile(r"(?i)\b(no)(thx)\b"),
                     re.compile(r"(?i)\b(Wha)(chu)\b"), # Hack...
                     re.compile(r"(?i)\b(Wha)(cha)\b"), # Hack...
                     re.compile(r"(?i)\b(What)(chu)\b"), # Hack...
                     re.compile(r"(?i)\b(What)(cha)\b"), # Hack...
                     re.compile(r"(?i)\b(Whad)(ya)\b"), # Hack...
                     re.compile(r"\b(I)(m)\b"),
                     re.compile(r"\b(I)(ll)\b"),
                     re.compile(r"(?i)\b(wo)(nt)\b")
                     ]
    CONTRACTIONS3 = [ re.compile(r"(?i)\b(Whad)(dd)(ya)\b")]
    
    def tokenize(self, text):
        for regexp in self.CONTRACTIONS2:
            text = regexp.sub(r'\1 \2', text)
        for regexp in self.CONTRACTIONS3:
            text = regexp.sub(r'\1 \2 \3', text)

        # Separate most punctuation. XXX: Hrmm.. hash tags...
        text = re.sub(r"([^\w\.\'\-\/,&\:\*\@\#])", r' \1 ', text)

        # Separate :'s if they are not followed by // (urls)
        text = re.sub(r'(\:)([^\/][^\/])', r' \1 \2 ', text)

        # Separate periods into consecutive groups
        text = re.sub(r'([\.]+[\s$])', r' \1 ', text)

        # Separate commas if they're followed by space.
        # (E.g., don't separate 2,500)
        text = re.sub(r"(,\s)", r' \1', text)

        # Separate single quotes if they're followed by a space.
        text = re.sub(r"('\s)", r' \1', text)

        return text.split()

    SNUGGLE_PUNCTS = set([ "$", "#", "@" ])
    NONSNUGGLE_PUNCTS = set([ "&", "-", "+" ]) # could also be suprsnuggle...

    # XXX: Still having problems with NONSNUGGLE's
    def detokenize(self, tokens):
        tok_len = len(tokens)
        i = 0
        text = ""
        while i < tok_len-1:
            # FIXME: Super ghetto, but not critical path and I'm lazy.
            if len(self.tokenize(tokens[i]+tokens[i+1])) == 2 and \
                tokens[i+1] not in self.SNUGGLE_PUNCTS and \
                tokens[i+1] not in self.NONSNUGGLE_PUNCTS:
                text += tokens[i]+tokens[i+1]
                i+=2
            else:
                text += tokens[i]
                i+=1
            if i < tok_len:
              ispunct = True
              for c in xrange(len(tokens[i])):
                if not curses.ascii.ispunct(tokens[i][c]) or \
                   tokens[i] in self.SNUGGLE_PUNCTS:
                  ispunct = False
              if ispunct:
                text += tokens[i]+" "
                i+=1
              else:
                text += " "

        if i < tok_len: text += tokens[i]
        return text
