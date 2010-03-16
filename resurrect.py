#!/usr/bin/python

# Resurrecter: Takes a soul file and animates it.

try:
  import psyco
  psyco.full()
except: pass

import nltk
import re
import cPickle as pickle
import random
import math
import numpy
import curses.ascii
import threading
import time
import traceback
#import twitter

from libs.tokenizer import word_tokenize, word_detokenize
from libs.SpeechModels import PhraseGenerator
from extract import CorpusSoul

# Maybe pos-tag isn't that important for searching. It seems
# to make some things harder... We probably should strip tenses from
# verbs, at least
# XXX: Unused
class POSTrim:
   def __init__(self):
    self.pos_map = { "VBD":"VB", "VBG":"VB", "VBN":"VB", "VBP":"VB", "VBZ":"VB",
                       "JJR":"JJ", "JJS":"JJ",
                       "NNS":"NN", "NNPS":"NN", "NNP":"NN" }

   def trim(self, pos):
     if pos in self.pos_map:
       return self.pos_map[pos]
     else:
       return pos

# RP: particle
# CC: conjunction, coordinating
# MD: modal auxiliary
# PDT: pre-determiner
# POS: genitive marker
# TO: "to" as preposition or infinitive marker
# WDT: WH-determiner (that)
# WP: WH-pronoun (that)
# WP$: WH-pronoun, possessive (whose)
# WRB: Wh-adverb (how, why)
class QueryStripper:
  kill_pos = set(["RP", "CC", "MD", "PDT", "POS", "TO", "WDT", "WP", "WP$",
                  "WRB"])
  @classmethod
  def strip_tagged_query(cls, tagged_query):
    ret = []
    for t in tagged_query:
      if t[1] not in cls.kill_pos:
        ret.append(t)
    return ret

#  - Subsitute @msgs: I/Me->You, Mine->Yours, My->Your
class PronounInverter:
  pronoun_21_map = { "you":"i", "yours":"mine","your":"my" }
  pronoun_12_map = { "i":"you", "me":"you", "mine":"yours","my":"your" }

  # XXX: use (?i) instead of this casehack nonsense
  @classmethod
  def make_2nd_person(cls, string):
    for i in cls.pronoun_12_map.iterkeys():
      chars = list(i)
      casehack = ""
      for c in chars: casehack += "["+c.lower()+c.upper()+"]"
      string = re.sub("([^a-zA-Z]|^)"+casehack+"([^a-zA-Z]|$)",
                      "\\1"+cls.pronoun_12_map[i]+"\\2", string)
    return string

  @classmethod
  def make_1st_person(cls, string):
    for i in cls.pronoun_21_map.iterkeys():
      chars = list(i)
      casehack = ""
      for c in chars: casehack += "["+c.lower()+c.upper()+"]"
      string = re.sub("([^a-zA-Z]|^)"+casehack+"([^a-zA-Z]|$)",
                      "\\1"+cls.pronoun_21_map[i]+"\\2", string)
    return string

  @classmethod
  def invert_all(cls, string):
    split_string = string.split()
    new_string = []
    for s in split_string:
      snew = cls.make_1st_person(s)
      if snew == s:
        snew = cls.make_2nd_person(s)
      new_string.append(snew)
    return " ".join(new_string)



# FIXME: Should these be static classes? Some maybe should
# do some record keeping on what they modify for inversion..
porter = nltk.PorterStemmer()

#- Notion of "pronoun context" per person
#  - Store list of recent nouns+verbs used in queries/responses
#    - If no specific noun/verb is found, use these for query
#  - associate per @msg user
#  - Expire after a while. Exponential backoff?
class ConversationContext:
  pass

#- Knowledge Base:
#  - Subsitute @msgs: I/Me->You, Mine->Yours, My->Your
#  - Tag with sender @name as hidden text, add to list of tweets for searching
class KnowledgeBase:
  pass

#- foursquare
#  - special place query parsing
#  - Random lamentations about favorite places and being trapped in the
#    Internet
#    - Trim down location to key "important" word using tf-idf
#  - Named entity recognition and relation extraction:
#    - http://nltk.googlecode.com/svn/trunk/doc/book/ch07.html
class FourSquareData:
  pass

#- url simmilarity
#  - When given a url, give a similar one back
#  - Or add related urls to generated sentences.
#  - summarize the url
#    - http://stackoverflow.com/questions/626754/how-to-find-out-the-summarized-text-of-a-given-url-in-python-django
#      - http://tristanhavelick.com/summarize.zip
#    - feed summary into SearchableTextCollection
#      - score summary sentences based on similarity to bot's word frequency
#        tables..
class URLClassifier:
  pass

class SearchableText:
  skip_tokens = set(['"','(',')','[',']']) # XXX: use?
  def __init__(self, text, tokens=None, tagged_tokens=None, strip=False, hidden_text=""):
    if hidden_text:
      hidden_text = hidden_text.rstrip()
      if not curses.ascii.ispunct(hidden_text[-1]):
        hidden_text += ". "
      else:
        hidden_text += " "
    self.hidden_text = hidden_text

    if not curses.ascii.ispunct(text[-1]): text += "."

    self.tagged_tokens = tagged_tokens
    if tokens: self.tokens = tokens
    else: self.tokens = word_tokenize(text)
    self.text = text

    if hidden_text:
      self.tokens.extend(word_tokenize(hidden_text))

    # Include hidden text in search tokens
    pos_tags = nltk.pos_tag(self.tokens)

    if strip:
      self.search_tokens = [porter.stem(t[0]).lower() for t in
                            QueryStripper.strip_tagged_query(pos_tags)]
    else:
      self.search_tokens = [porter.stem(t[0]).lower() for t in pos_tags]
    self.vocab = set(self.search_tokens)

  def count(self, word):
    return self.search_tokens.count(word)


# Search:
#  1. Tag, tolower(), stem
#  2. if strip: strip all but nouns, adj, verbs
#  3. tf-idf to score text record from collection
#  4. TODO: If score is 0, requery for "WTF buh uh huh dunno what talking about"
#  5. FIXME: Return probabilistically based on score with some cutoff
#  6. TODO: try to avoid replying to questions with questions. score lower.

# TODO: Build the score vectors using both the tagged and untaged words
# Then if tags and words match, you get more score than just tags,
# but its not total fail if your question can't be parsed..
class SearchableTextCollection:
  def __init__(self, texts=[]):
    self._idf_cache = {}
    self.texts = texts
    if texts:
      self.needs_update = True
      self.update_matrix()

  def add_text(self, text, update=False):
    self.texts.append(text)
    self.needs_update = True
    if update: self.update_matrix()

  def remove_text(self, text, update=False):
    try:
      self.texts.remove(text)
      self.needs_update = True
      if update: self.update_matrix()
    except ValueError,e:
      print "Item not in list: "+text.text
      return

  def update_matrix(self):
    print "Building Terms..."
    terms = set()
    for t in self.texts:
      terms.update(t.vocab)
    self.vocab = list(terms)
    self.vocab.sort()
    print "Built Terms."
    self.D = []
    for doc in self.texts:
      d = []
      for dt in self.vocab:
        if dt in doc.vocab: d.append(self.tf_idf(dt, doc))
        else: d.append(0.0)
      d = numpy.array(d)
      d /= math.sqrt(numpy.dot(d,d))
      self.D.append(d)
    print "Computed score matrix."
    self.needs_update = False

  def query(self, query_string, randomize=False, only_pos=[]): #only_pos=["FW", "CD", "$", "N.*"]):
    if self.needs_update: self.update_matrix()

    if not query_string:
      print "Empty query!"
      return random.choice(self.texts)

    query_string = PronounInverter.invert_all(query_string)
    print "Inverted Query: "+query_string
    query_text = SearchableText(query_string, strip=True)

    print "Building Qvector.."
    # XXX: Hrmm, could amplify nouns and dampen adjectives and verbs:
    # Normalize to 0.75*(max_noun/max_word)
    # FIXME: If no nouns, only pronouns, use state from queries+responses
    q = []
    for dt in self.vocab:
      if dt in query_text.vocab:
        ok = not bool(only_pos)
        for p in only_pos:
          if re.match("/"+p, dt): ok = True
        if ok: q.append(self.tf_idf(dt, query_text))
        else: q.append(0.0)
      else: q.append(0.0)

    q = numpy.array(q)
    q /= math.sqrt(numpy.dot(q,q))

    print "Qvector built"
    scores = []
    tot_score = 0.0
    for d in self.D:
      score = numpy.dot(q,d)
      tot_score += score
      scores.append(score)

    print "Matrix multiplied"

    # FIXME: hrmm. This will mess with our pronoun logic
    if tot_score == 0.0:
      print "Zero score.. Recursing"
      return self.query(SearchableText("WTF buh uh huh dunno what talking about understand"),
                        randomize, only_pos)

    if randomize:
      sorted_scores = []
      for i in xrange(len(scores)):
        sorted_scores.append((scores[i], i))
      sorted_scores.sort(lambda x,y: int(y[0]*10000 - x[0]*10000))

      # Choose from top 5 scores..
      # FIXME: This is still kind of arbitrary
      top_quart = 0
      count = 0.0
      for i in xrange(5):
        top_quart += sorted_scores[i][0]
      choice = random.uniform(0, top_quart)
      for i in xrange(5):
        count += sorted_scores[i][0]
        if count >= choice:
          print "Rand score: "+str(sorted_scores[i][0])+"/"+str(sorted_scores[0][0])
          print "Choice: "+str(choice)+" count: "+str(count)+" top_quater: "+str(top_quart)
          return self.texts[sorted_scores[i][1]]
      print "WTF? No doc found: "+str(count)+" "+str(tot_score)
      return random.choice(self.texts)
    else:
      max_idx = 0
      for i in xrange(len(scores)):
        if scores[i] > scores[max_idx]:
          max_idx = i
      print "Max score: "+str(scores[max_idx])
      return self.texts[max_idx]

  def tf(self, term, text):
    """ The frequency of the term in text. """
    return float(text.count(term)) / len(text.search_tokens)

  def idf(self, term):
    """ The number of texts in the corpus divided by the
    number of texts that the term appears in. 
    If a term does not appear in the corpus, 0.0 is returned. """
    # idf values are cached for performance.
    idf = self._idf_cache.get(term)
    if idf is None: 
      matches = len(list(True for text in self.texts if term in text.vocab))
      if not matches:
        idf = 0.0
      else:
        idf = math.log(float(len(self.texts)) / matches)
      self._idf_cache[term] = idf
    return idf

  def tf_idf(self, term, text):
    return self.tf(term, text) * self.idf(term)

# The brain holds the already tweeted list and the list of pending tweets
# as a SearchableTextCollection.
#
# It also manages a separate worker thread for generating more pending tweets.
class TwitterBrain:
  def __init__(self, soul, pending_tweets=5000):
    self.pending_goal = pending_tweets
    self.pending_tweets = SearchableTextCollection()
    self.already_tweeted = []
    self.remove_tweets = []
    for t in soul.tagged_tweets:
      words = map(lambda x: x[0], t)
      self.already_tweeted.append(set(words))
    self.restart(soul)

  def restart(self, soul, pending_tweets=5000):
    self.pending_goal = pending_tweets
    self.voice = PhraseGenerator(soul.tagged_tweets, soul.normalizer)

    self.work_lock = threading.Lock()
    self._shutdown = False
    self._thread = threading.Thread(target=self.__phrase_worker)
    self._thread.start()

  def get_tweet(self, query=None):
    self.__lock()
    if not query:
      ret = random.choice(self.pending_tweets.texts)
    else:
      ret = self.pending_tweets.query(query)
    self.remove_tweets.append(ret)
    self.already_tweeted.append(set(ret.tokens))
    self.__unlock()
    return (ret.text, ret.tokens, ret.tagged_tokens)

  def __did_already_tweet(self, words, max_score=0.75):
    for t in self.already_tweeted:
      score = float(len(t & words))
      score1 = score/len(t)
      score2 = score/len(words)
      if score1 > max_score or score2 > max_score:
        #print "Too similar to old tweet.. skipping: "+\
        #         str(score1)+"/"+str(score2)
        return True
    return False

  def __phrase_worker(self):
    try:
      self.__phrase_worker2()
    except:
      print "Worker thread died."
      traceback.print_exc()
    print "Worker thread quit."

  # Needed to avoid the race condition on pickling..
  def __lock(self):
    lock = self.work_lock
    while not lock:
      lock = self.work_lock
      time.sleep(1)
    lock.acquire()

  def __unlock(self):
    self.work_lock.release()

  def __phrase_worker2(self):
    first_run = True
    while not self._shutdown:
      added_tweets = False
      if len(self.remove_tweets) > 0:
        self.__lock()
        while len(self.remove_tweets) > 0:
          self.pending_tweets.remove_text(self.remove_tweets.pop())
        self.pending_tweets.update_matrix()
        self.__unlock()

      while len(self.pending_tweets.texts) < self.pending_goal:
        (tweet,tokens,tagged_tokens) = self.voice.say_something()
        if len(tweet) > 140: continue

        self.__lock()

        if self.__did_already_tweet(set(tokens)):
          self.__unlock()
          continue

        self.pending_tweets.add_text(SearchableText(tweet,tokens,tagged_tokens),
                           update=(not first_run))
        added_tweets = True
        self.__unlock()

        if len(self.pending_tweets.texts) % 100 == 0:
          # XXX: Cleanup filename
          break # Perform other work
      if added_tweets:
        print "At tweet count "+str(len(self.pending_tweets.texts))+\
                  "/"+str(self.pending_goal)
        # XXX: Cleanup filename
        BrainReader.write(self, open("target_user.brain", "w"))
      if len(self.pending_tweets.texts) == self.pending_goal:
        first_run=False
      time.sleep(3)

# Lousy hmm can't be pickled
class BrainReader:
  @classmethod
  def write(cls, brain, f):
    (voice,thread,lock) = (brain.voice,brain._thread,brain.work_lock)
    lock.acquire()
    (brain.voice,brain._thread,brain.work_lock) = (None,None,None)
    pickle.dump(brain, f)
    (brain.voice,brain._thread,brain.work_lock) = (voice,thread,lock)
    lock.release()

  @classmethod
  def load(cls, f):
    brain = pickle.load(f)
    return brain


def main():
  brain = None
  soul = None

  try:
    soul = pickle.load(open("target_user.soul", "r"))
  except Exception,e:
    traceback.print_exc()
    print "No soul file found. Regenerating."
    soul = CorpusSoul('target_user')
    pickle.dump(soul, open("target_user.soul", "w"))

  try:
    brain = BrainReader.load(open("target_user.brain", "r"))
    brain.restart(soul)
  except Exception,e:
    traceback.print_exc()
    print "No brain file found. Regenerating."
    brain = TwitterBrain(soul)
    BrainReader.write(brain, open("target_user.brain", "w"))


  while True:
    query = raw_input("> ")
    (str_result, tokens, tagged_tokens) = brain.get_tweet(query)
    print str_result
    print str(tagged_tokens)

if __name__ == "__main__":
  main()
