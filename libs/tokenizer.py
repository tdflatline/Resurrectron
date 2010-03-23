#!/usr/bin/python
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

import curses.ascii
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
                     re.compile(r"(?i)\b(sort)(a)\b"),
                     re.compile(r"(?i)\b(o)(rly)\b"),
                     re.compile(r"(?i)\b(ya)(rly)\b"),
                     re.compile(r"\b(I)(m)\b"),
                     re.compile(r"\b(I)(ll)\b"),
                     re.compile(r"(?i)\b(wo)(nt)\b"),
                     # FIXME: "Dunno" -> do not know... need to use 3 tuples :(
                     re.compile(r"(?i)\b(Wha)(chu)\b"), # Hack...
                     re.compile(r"(?i)\b(Wha)(cha)\b"), # Hack...
                     re.compile(r"(?i)\b(What)(chu)\b"), # Hack...
                     re.compile(r"(?i)\b(What)(cha)\b"), # Hack...
                     re.compile(r"(?i)\b(Whad)(ya)\b") # Hack...
                     ]
    CONTRACTIONS3 = [ re.compile(r"(?i)\b(Whad)(dd)(ya)\b")]

    # FIXME: re.compile these...
    TITLES = ["mr", "mrs", "ms", "dr", "fr", "lt", "sgt", "rev"]

    def tokenize(self, text):
        for regexp in self.CONTRACTIONS2:
            text = regexp.sub(r'\1 \2', text)
        for regexp in self.CONTRACTIONS3:
            text = regexp.sub(r'\1 \2 \3', text)

        # Separate most punctuation.
        text = re.sub(r"([^\w\.\-\'\/,\:\*\@\#\s])", r' \1 ', text)

        # Separate :'s if they are not followed by numbers or spaces.
        # (urls or smilies)...
        text = re.sub(r'(\:)([\s0-9]|$)', r' \1 \2', text)

        # Separate - if they have non-alphas between them
        text = re.sub(r'(\W)-(\W)', r'\1 - \2', text)

        # Separate commas if they're followed by space.
        # (E.g., don't separate 2,500)
        text = re.sub(r"(,\s)", r' \1', text)

        # Handle "St." special. Watch for end of sentence.
        text = re.sub(r"((?:^| )[Ss][Tt])[\.]([\s]+[^A-Z0-9])", r"\1=\2", text)

        # Separate titles into consecutive groups. We know this
        # is OK to do, because "=" would have been separated earlier.
        for t in self.TITLES:
          text = re.sub(r"(?i)((?:^| )"+t+r")[\.]( |$)", r"\1=\2", text)

        # Split periods into groups.
        text = re.sub(r'([\.]+(?:[\s]|$))', r' \1 ', text)
        text = re.sub(r'([\.]{2,})', r' \1 ', text)

        # Subsititute titles back.
        for t in self.TITLES + ['st']:
          text = re.sub(r"(?i)((?:^| )"+t+r")=", r"\1.", text)

        # Separate single quotes if they're followed by a space.
        #text = re.sub(r"('\s)", r' \1', text)

        return text.split()

    PREFIX_PUNCTS = set([ "$", "#", "@", "-" ])
    INFIX_PUNCTS = set([ "&", "+", "=" ]) # could also be suprsnuggle...

    def detokenize(self, orig_tokens):
        tokens = []
        for t in orig_tokens: tokens.extend(t.split())

        tok_len = len(tokens)
        i = 0
        text = ""
        while i < tok_len-1:
            ltok = tokens[i+1].lower()
            found_title = False
            for t in self.TITLES + ['st']:
              if ltok == t+".":
                found_title = True
                break
            if found_title:
              text += tokens[i]+" "
              i+=1
              continue

            # Join up times
            if i < tok_len-2 and \
              (re.match("[\d]\+:\+[\d]",
                  tokens[i]+"+"+tokens[i+1]+"+"+tokens[i+2]) or\
                 tokens[i+1] in self.INFIX_PUNCTS or \
               re.match("[\w]\+-\+[\w]",
                  tokens[i]+"+"+tokens[i+1]+"+"+tokens[i+2])):
                text += tokens[i]+tokens[i+1]+tokens[i+2]
                i+=3
            # FIXME: Super ghetto, but not critical path and I'm lazy.
            elif len(self.tokenize(tokens[i]))+len(self.tokenize(tokens[i+1])) == 2 and\
                 len(self.tokenize(tokens[i]+tokens[i+1])) >= 2 and \
                tokens[i+1] not in self.PREFIX_PUNCTS:
                text += tokens[i]+tokens[i+1]
                i+=2
            else:
                text += tokens[i]
                i+=1
            if i < tok_len:
              ispunct = True
              for c in xrange(len(tokens[i])):
                if not curses.ascii.ispunct(tokens[i][c]) or \
                   tokens[i] in self.PREFIX_PUNCTS:
                  ispunct = False
              if ispunct:
                text += tokens[i]+" "
                i+=1
              else:
                text += " "

        if i < tok_len: text += tokens[i]
        return text

toker = TreebankWordTokenizer()
def word_tokenize(text):
  return toker.tokenize(text)

def word_detokenize(tokens):
  return toker.detokenize(tokens)

def main():
  tok = TreebankWordTokenizer()
  print tok.tokenize("Hi a-ok.... g+money I&I. You..like? http://www.foo.com? www.foo.com. www.goo.com/ ")
  print tok.detokenize(["25%", "predict", "a lot", "of", "G-Unit", "45%", "fashion"])
  print tok.tokenize("I&u are+teh best: @ 7:30 pm 1:30. a=hi.")
  print tok.detokenize(tok.tokenize("I&u are+teh best: @ 7:30 pm 1:30. a=hi."))
  print tok.tokenize("Mr. Johnson's ate a Dr.'s hotdog on Main St. USA :p.")
  print tok.detokenize(tok.tokenize("The Mr. Johnson ate a Dr.'s hotdog on Main St. USA"))


if __name__ == "__main__":
  main()
