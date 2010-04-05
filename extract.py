#!/usr/bin/python

# Extractor: Extracts a soul from a harvested corpus

try:
  import psyco
  psyco.full()
except: pass

import gzip
import nltk
import simplejson as json
import os
import re
import cPickle as pickle
import random
import curses.ascii
import traceback
import sys

from libs.SpeechModels import TokenNormalizer, PhraseGenerator
from libs.tokenizer import word_tokenize, word_detokenize

from libs.tagger import pos_tag
from libs.summarize import SimpleSummarizer

from ConfigParser import SafeConfigParser
config = SafeConfigParser()
config.read('settings.cfg')

class CorpusSoul:
  def __init__(self, directory):
    self.normalizer = TokenNormalizer()
    self.quote_engine_only = config.getboolean('soul', 'quote_engine_only')
    # FIXME: http://www.w3schools.com/HTML/html_entities.asp
    clean_ents = [("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&")]
    tagged_tweets = []
    self.tweet_texts = []
    self.vocab = set([])
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
            for e in clean_ents:
              txt = re.sub(e[0], e[1], txt)
            if self.quote_engine_only:
              tagged_tweets.append(txt)
            else:
              tokens = self.normalizer.normalize_tokens(word_tokenize(txt))
              if tokens:
                self.tweet_texts.append(word_detokenize(tokens))
                self.vocab.update(tokens)
                tagged_tweet = pos_tag(tokens,
                                 config.getboolean("soul","attempt_agfl"),
                                 config.getboolean("soul","reject_agfl_failures"),
                                 config.getboolean("soul","agfl_nltk_fallback"))
                if tagged_tweet: tagged_tweets.append(tagged_tweet)
            print "Loaded tweet #"+str(len(tagged_tweets)) #+"/"+str(len(files))
        # .twt: plain-text tweets, 1 per line
        elif f.endswith(".twt"):
          fl = open(root+"/"+f, "r")
          for tweet in fl.readlines():
            txt = tweet.encode('ascii', 'ignore')
            if txt.startswith('RT'): continue
            if txt[0] == '@': txt = re.sub('^@[\S]+ ', '', txt)
            for e in clean_ents:
              txt = re.sub(e[0], e[1], txt)
            if self.quote_engine_only:
              tagged_tweets.append(txt)
            else:
              tokens = self.normalizer.normalize_tokens(word_tokenize(txt))
              if tokens:
                self.tweet_texts.append(word_detokenize(tokens))
                self.vocab.update(tokens)
                tagged_tweet = pos_tag(tokens,
                              config.getboolean("soul","attempt_agfl"),
                              config.getboolean("soul","reject_agfl_failures"),
                              config.getboolean("soul","agfl_nltk_fallback"))
                if tagged_tweet: tagged_tweets.append(tagged_tweet)
            print "Loaded tweet #"+str(len(tagged_tweets)) #+"/"+str(len(files))
          pass
        # .post: long-winded material (blog/mailinglist posts, essays, articles, etc)
        elif f.endswith(".post"):
          fl = open(root+"/"+f, "r")
          post = fl.read()
          tweets = self.post_to_tweets(post)
          for txt in tweets:
            #txt = txt.encode('ascii', 'ignore')
            for e in clean_ents:
              txt = re.sub(e[0], e[1], txt)
            if self.quote_engine_only:
              tagged_tweets.append(txt)
            else:
              tokens = self.normalizer.normalize_tokens(word_tokenize(txt))
              if tokens:
                self.tweet_texts.append(word_detokenize(tokens))
                self.vocab.update(tokens)
                tagged_tweet = pos_tag(tokens,
                               config.getboolean("soul","attempt_agfl"),
                               config.getboolean("soul","reject_agfl_failures"),
                               config.getboolean("soul","agfl_nltk_fallback"))
                if tagged_tweet: tagged_tweets.append(tagged_tweet)
            print "Loaded post-tweet #"+str(len(tagged_tweets))
        # .irclog: irc log files. irssi format.
        elif f.endswith(".irclog"):
          pass
        # .4sq: foursquare data
        elif f.endswith(".4sq"):
          pass

    self.tagged_tweets = tagged_tweets

  def post_to_tweets(self, post, summarize=False):
    # We do poorly with parentheticals. Just kill them.
    post = re.sub(r"\([^\)]+\)", "", post)
    if summarize:
      summ = SimpleSummarizer()
      post = summ.summarize(post, config.getint("soul", "post_summarize_len"))
    sentences = nltk.sent_tokenize(post)
    tweets = []
    tweet = ""
    for s in sentences:
      if len(s) > config.getint("soul","post_len"): continue
      if len(tweet + s) < config.getint("soul","post_len"):
        tweet += s+" "
      else:
        if tweet: tweets.append(tweet)
        tweet = ""
    return tweets

  def cluster_tweets(self, num_clusters=3, use_topics=False):
    # XXX: move SearchableTextCollection to libs
    from resurrect import SearchableTextCollection,SearchableText
    # Hack for clusterer
    from numpy import array

    print "Scoring tweets.."
    tc = SearchableTextCollection(self.vocab)
    for tweet in self.tweet_texts:
      txt = SearchableText(tweet)
      tc.add_text(txt)
    tc.update_matrix()
    means = []
    if use_topics:
      for topic in self.cluster_topics:
        means.append(tc.score_query(SearchableText(topic)))
    else:
      for i in xrange(num_clusters):
        means.append(random.choice(tc.D))
    print "Scored tweets.."
    cluster = nltk.cluster.KMeansClusterer(num_clusters,
                 nltk.cluster.util.euclidean_distance,
                 repeats=20*num_clusters)
                 #svd_dimensions=100)
    #cluster = nltk.cluster.EMClusterer(means, svd_dimensions=100)
    clustered = cluster.cluster(tc.D, assign_clusters=True)
    print "Clustered tweets.."
    clustered_tweets = {}
    for i in xrange(len(clustered)):
      if clustered[i] not in clustered_tweets:
        clustered_tweets[clustered[i]] = []
      clustered_tweets[clustered[i]].append(self.tweet_texts[i])
    for i in clustered_tweets.iterkeys():
      print
      print "Cluster "+str(i)+": "+str(len(clustered_tweets[i]))
      for t in clustered_tweets[i]:
        print t

def main():
  try:
    print "Loading soul file..."
    soul = pickle.load(open("target_user.soul", "r"))
    print "Loaded soul file."
  except pickle.UnpicklingError:
    soul = pickle.load(gzip.GzipFile("target_user.soul", "r"))
    print "Loaded soul file."
  except KeyError:
    soul = pickle.load(gzip.GzipFile("target_user.soul", "r"))
    print "Loaded soul file."
  except IOError:
    print "No soul file found. Regenerating."
    soul = CorpusSoul('target_user')
    if config.getboolean("soul", "gzip_soul"):
      pickle.dump(soul, gzip.GzipFile("target_user.soul", "w"))
    else:
      pickle.dump(soul, open("target_user.soul", "w"))
  except Exception,e:
    traceback.print_exc()

  soul.normalizer.verify_scores()

  soul.cluster_tweets(10)

  sys.exit(0)

  voice = PhraseGenerator(soul.tagged_tweets, soul.normalizer,
                          config.getint("brain","hmm_context"),
                          config.getint("brain","hmm_offset"))

  while True:
    query = raw_input("> ")
    if not query: query = "41"
    if query.isdigit():
      (str_result, tok_result, result) = voice.say_something()
      print str(result)
      print str_result

if __name__ == "__main__":
  main()
