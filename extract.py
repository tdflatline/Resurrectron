#!/usr/bin/python

# Extractor: Extracts a soul from a harvested corpus

try:
  import psyco
  psyco.full()
except: pass

import nltk
import simplejson as json
import os
import re
import cPickle as pickle
import random
import curses.ascii
import traceback
#import twitter

import libs.AGFL

from libs.SpeechModels import TokenNormalizer, PhraseGenerator
from libs.tokenizer import word_tokenize, word_detokenize

# Temporarily strip out stuff we left in (esp for AGFL)
# ["#", "*", "@", "/"]
def agfl_fix(tokens, nltk_tags):
  fixed = False
  # XXX: Add these to normalizer? Or just invert them after parsing
  # XXX: Standalone numbers too?
  for t in xrange(len(nltk_tags)):
    tokens[t] = re.sub(r"\.\.[\.]+", "...", tokens[t])
    if nltk_tags[t][0] == "'s":
      if nltk_tags[t][1] == "VBZ":
        tokens[t] = "is"
      elif nltk_tags[t][1] == "POS": # Evil Hack. XXX: Undo?
        tokens[t-1] += "s"
        nltk_tags[t-1] = (nltk_tags[t-1][0]+"s", nltk_tags[t-1][1])
        fixed = True
  if fixed:
    nltk_tags.remove(("'s", "POS"))
    tokens.remove("'s")

# These tend to be pretty solid.
nltk_ep4_map = {
  "PRP$" : "POSSPRON", # pronoun, possessive
  "RB" : "ADVB(modf)", # adverb
  "RBR" : "ADVB(comp)", # adverb, comparative
  "RBS" : "ADVB(supl)", # adverb, superlative
  "RP" : "PARTICLE(none)", # particle
  "JJ" : "ADJE(abso)", # adjective or numeral, ordinal 
  "JJR" : "ADJE(comp)", # adjective, comparative
  "JJS" : "ADJE(supl)", # adjective, superlative
  "CC" : "CON(coo)", # conjunction, coordinating
  "IN" : "CON(sub)", # preposition or conjunction, subordinating
  "UH" : "INTERJ", # interjection
  "CD" : "NUM(card)", # numeral, cardinal
  "." : "-."
}

# These tags tend not to work very well with NLTK.
# Leave them (and others) as their NLTK equivs to
# let the HMM compensate a bit. Left here
# for reference.
nltk_ep4_map_junky = {
  # NLTK is too generous with tagging things as nouns
  "NN" : "NOUN(sing)", # noun, common, singular or mass
  "NNP" : "NOUN(sing)", # noun, proper, singular
  "NNPS" : "NOUN(plur)", # noun, proper, plural
  "NNS" : "NOUN(plur)", # noun, common, plural

  # These need some review..
  "VB" : "VERBI(NONE,none,cplx)", # verb, base form
  "VBD" : "VERBP(NONE,none,cplx)", # verb, past tense
  "VBG" : "VERBG(NONE,none,cplx)", # verb, present participle or gerund
  "VBN" : "VERBP(NONE,none,trav)", # verb, past participle
  "VBP" : "VERBV(NONE,none,cplx)", # verb, present tense, not 3rd person singular
  "VBZ" : "VERBS(None,none,cplx)",# verb, present tense, 3rd person singular
  "PRP" : "PERSPRON(nltk,nltk,nltk)" # pronoun, personal
}

def agfl_repair(agfl_tags, nltk_tags):
  # This is somewhat hackish
  for a in xrange(len(agfl_tags)):
    if not agfl_tags[a][1] or agfl_tags[a][1] == "WORD" or \
          agfl_tags[a][1].isspace():
      closest = -1
      for n in xrange(len(nltk_tags)):
        if agfl_tags[a][0] == nltk_tags[n][0]:
          if closest > 0:
            print "Two tags found for "+agfl_tags[a][0]
            if abs(a-n) > abs(a-closest):
              continue # Farther away. Skip
          closest = n
          if nltk_tags[n][1] in nltk_ep4_map:
            agfl_tags[a] = (nltk_tags[n][0], nltk_ep4_map[nltk_tags[n][1]])
          else:
            print nltk_tags[n][1]+" not in nltk map!"
            agfl_tags[a] = (nltk_tags[n][0], nltk_tags[n][1])

agfl = libs.AGFL.AGFLWrapper()
def pos_tag(tokens):
  if agfl.agfl_ok():
    nltk_tags = nltk.pos_tag(tokens)
    agfl_fix(tokens, nltk_tags)
    detoked = word_detokenize(tokens)
    sentences = nltk.sent_tokenize(detoked)
    all_tags = []
    for s in sentences:
      if not s:
        print "Empty string for: "+str(tokens)
        continue
      print "Parsing: |"+s+"|"
      agfl_tree = agfl.parse_sentence(s)
      if not agfl_tree:
        print "Parse fail for |"+s+"|"
        return None # Hrmm. use partials? Prob not
      else:
        tags = agfl_tree.pos_tag()
        if tags:
          all_tags.extend(tags)
        else:
          print "Tag fail for |"+s+"|"
          return None
    agfl_repair(all_tags, nltk_tags)
    return all_tags
  else:
    print "AGFL not being used" # XXX: Kill this log
    return nltk.pos_tag(tokens)

# XXX: "@kategardiner www.sexpigeon.org"
class CorpusSoul:
  def __init__(self, directory):
    self.normalizer = TokenNormalizer()
    tagged_tweets = []
    for root, dirs, files in os.walk(directory):
      for f in files:
        # .jtwt: json-encoded twitter tweets, 1 per line
        # TODO: Add @msgs to this user as hidden text
        if f.endswith(".jtwt"):
          fl = open(root+"/"+f, "r")
          for jtweet in fl.readlines():
            tweet = json.loads(jtweet)
            txt = tweet['text'].encode('ascii', 'ignore')
            if re.search("( |^)RT(:| )", txt, re.IGNORECASE): continue
            if txt[0] == '@': txt = re.sub('^@[\S]+ ', '', txt)
            tagged_tweet = pos_tag(self.normalizer.normalize_tokens(word_tokenize(txt)))
            if tagged_tweet: tagged_tweets.append(tagged_tweet)
            print "Loaded tweet #"+str(len(tagged_tweets)) #+"/"+str(len(files))
        # .twt: plain-text tweets, 1 per line
        elif f.endswith(".twt"):
          fl = open(root+"/"+f, "r")
          for tweet in fl.readlines():
            txt = tweet.encode('ascii', 'ignore')
            if txt.startswith('RT'): continue
            if txt[0] == '@': txt = re.sub('^@[\S]+ ', '', txt)
            tagged_tweet = pos_tag(self.normalizer.normalize_tokens(word_tokenize(txt)))
            if tagged_tweet: tagged_tweets.append(tagged_tweet)
            print "Loaded tweet #"+str(len(tagged_tweets)) #+"/"+str(len(files))
          pass
        # .post: long-winded material (blog/mailinglist posts, essays, articles, etc)
        elif f.endswith(".post"):
          pass
        # .irclog: irc log files. irssi format.
        elif f.endswith(".irclog"):
          pass
        # .4sq: foursquare data
        elif f.endswith(".4sq"):
          pass

    for t in tagged_tweets:
      words = map(lambda x: x[0], t)
      # Need to compute finals scores for normalzier to be able to denormalize
      self.normalizer.score_tokens(words)

    self.tagged_tweets = tagged_tweets

# Lousy hmm can't be pickled
class SoulWriter:
  def __init__(self):
    pass

  @classmethod
  def write(cls, soul, f):
    voice = soul.voice
    soul.voice = None
    pickle.dump(soul, f)
    soul.voice = voice

  @classmethod
  def load(cls, f):
    soul = pickle.load(f)
    soul.voice = PhraseGenerator(soul.tagged_tweets, soul.normalizer)
    return soul

def main():
  try:
    soul = pickle.load(open("target_user.soul", "r"))
  except Exception,e:
    traceback.print_exc()
    print "No soul file found. Regenerating."
    soul = CorpusSoul('target_user')
    pickle.write(soul, open("target_user.soul", "w"))

  voice = PhraseGenerator(soul.tagged_tweets, soul.normalizer)

  while True:
    query = raw_input("> ")
    if not query: query = "41"
    if query.isdigit():
      (str_result, tok_result, result) = voice.say_something()
      print str(result)
      print str_result

if __name__ == "__main__":
  main()
