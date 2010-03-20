#!/usr/bin/python

# Resurrecter: Takes a soul file and animates it.

try:
  import psyco
  psyco.full()
except:
  print "Psyco JIT not found. Queries will run MUCH slower."

import bz2
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
import cmd
import sys
import atexit
import en
#import twitter

from libs.tokenizer import word_tokenize, word_detokenize
from libs.SpeechModels import PhraseGenerator, TokenNormalizer
from extract import CorpusSoul

class POSTrim:
   pos_map = { "VBD":"VB", "VBG":"VB", "VBN":"VB", "VBP":"VB", "VBZ":"VB",
                       "JJR":"JJ", "JJS":"JJ",
                       "NNS":"NN", "NNPS":"NN", "NNP":"NN",
                       "RBR":"RB", "RBS":"RB" }
   @classmethod
   def trim(cls, pos):
     if pos in cls.pos_map:
       return cls.pos_map[pos]
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
      if t[1] not in cls.kill_pos and not curses.ascii.ispunct(t[0][0]):
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
#    - Possibly keep appending these keywords to queries
class ConversationContext:
  def __init__(self, nick, decay=0.25):
    self.nick = nick
    self.normalizer = TokenNormalizer()
    self.decay = decay
    self.memory_vector = None
    self.last_query_time = time.time()

  def prime_memory(self, vector=None):
    now = time.time()
    if now - self.last_query_time > 12*60*60:
      print "Priming stale memory query for user: "+self.nick
      self.memory_vector = vector
      self.last_query_time = now

  def decay_query(self, q_vector):
    now = time.time()
    if now - self.last_query_time > 12*60*60:
      print "Expiring stale memory query for user: "+self.nick
      self.memory_vector = None
    if self.memory_vector != None:
      self.memory_vector *= self.decay
      self.memory_vector += q_vector
    else:
      self.memory_vector = q_vector
    self.last_query_time = now
    return self.memory_vector

  def remember_query(self, q_vector):
    if self.memory_vector != None:
      self.memory_vector += q_vector
    else:
      self.memory_vector = q_vector
    return self.memory_vector

#- Knowledge Base:
#  - Subsitute @msgs: I/Me->You, Mine->Yours, My->Your
#  - Tag with sender @name as hidden text
#  - Maintain internal list of tweets for searching
#  - Compare scores to other tweet lists
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
  def __init__(self, text, tokens=None, tagged_tokens=None, strip=False, hidden_text="", generalize_terms=True):
    if hidden_text:
      hidden_text = hidden_text.rstrip()
      if not curses.ascii.ispunct(hidden_text[-1]):
        hidden_text += ". "
      else:
        hidden_text += " "
    self.hidden_text = hidden_text

    if not curses.ascii.ispunct(text[-1]): text += "."

    # XXX: We should remove the tagged tokens and possibly the tokens
    # too. This eats a lot of storage.
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

    if generalize_terms:
      # Add senses, antonyms, and hypernyms to this list with
      # http://nodebox.net/code/index.php/Linguistics
      # Also add normalized versions with en.spelling() first
      new_terms = set()
      for v in xrange(len(self.search_tokens)):
         sv = self.search_tokens[v]
         # en.spelling.correct(v)
         tag = POSTrim.trim(pos_tags[v][1])
         mod = None
         if tag == "NN": mod = en.noun
         elif tag == "JJ": mod = en.adjective
         elif tag == "VB": mod = en.verb
         elif tag == "RB": mod = en.adverb
         else: mod = en.wordnet # XXX: Too much cpu?
         if mod:
           new_terms.update(en.list.flatten(mod.senses(sv)))
           new_terms.update(en.list.flatten(mod.antonym(sv)))
           new_terms.update(en.list.flatten(mod.hypernym(sv)))
           new_terms.update(en.list.flatten(mod.hyponym(sv)))
      self.vocab.update(new_terms)
      self.search_tokens = list(self.vocab)

  def count(self, word):
    return self.search_tokens.count(word)


# Search:
#  1. Tag, tolower(), stem
#  2. if strip: strip all but nouns, adj, verbs
#  3. tf-idf to score text record from collection
#  4. If score is 0, requery for "WTF buh uh huh dunno what talking about"
#  5. Return probabilistically based on score with some cutoff

class SearchableTextCollection:
  def __init__(self, vocab, texts=[], generalize_terms=True):
    self._idf_cache = {}
    self.texts = texts
    if generalize_terms:
      vocab = set(vocab)
      new_terms = set()
      print "Generalizing vocabulary"
      for v in vocab:
         # Actually, no need to spell check here. It's probably
         # spelled right eventually, and this is SLOW!
         sv = v #en.spelling.correct(v)
         new_terms.update(sv)
         # Use parts of speech. en.wordnet is not a superset..
         for mod in [en.wordnet, en.adjective, en.noun, en.verb, en.adverb]:
           new_terms.update(en.list.flatten(mod.senses(sv)))
           new_terms.update(en.list.flatten(mod.antonym(sv)))
           new_terms.update(en.list.flatten(mod.hypernym(sv)))
           new_terms.update(en.list.flatten(mod.hyponym(sv)))
      vocab.update(new_terms)
      print "Generalized vocabulary"
    self.vocab = list(vocab)
    self.vocab.sort()
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
    print "Computing score matrix."
    self.D = []
    for doc in self.texts:
      d = []
      for dt in self.vocab:
        if dt in doc.vocab: d.append(self.tf_idf(dt, doc))
        else: d.append(0.0)
      d = numpy.array(d)
      norm = math.sqrt(numpy.dot(d,d))
      # XXX: This happens. Why?
      if norm <= 0:
        print "Zero row in matrix: "+doc.text
      else:
        d /= math.sqrt(numpy.dot(d,d))
      self.D.append(d)
    print "Computed score matrix."
    self.needs_update = False

  def score_query(self, query_string):
    if self.needs_update: self.update_matrix()
    query_string = PronounInverter.invert_all(query_string)
    print "Inverted Query: "+query_string
    query_text = SearchableText(query_string, strip=True)

    print "Building Qvector.."
    # TODO: Hrmm, could amplify nouns and dampen adjectives and verbs:
    # Normalize to 0.75*(max_noun/max_word)
    # TODO: If no nouns, only pronouns, use state from queries+responses
    q = []
    for dt in self.vocab:
      if dt in query_text.vocab:
        print "Found in text: "+str(dt)
        q.append(self.tf_idf(dt, query_text))
      else: q.append(0.0)

    q = numpy.array(q)
    # Numpy is retarded here.. Need to special case 0
    norm = math.sqrt(numpy.dot(q,q))
    if norm > 0: q /= norm
    else: print "Zero vector"
    print "Qvector built"
    return q

  def query_string(self, query_string, exclude=[], max_len=140,
                   randomize_top=1):
    if not query_string:
      print "Empty query!"
      return random.choice(self.texts)
    return self.query_vector(self.score_query(query_string), exclude, max_len,
                      randomize_top)

  def query_vector(self, query_vector, exclude=[], max_len=140,
                   randomize_top=1):
    q = query_vector

    print "Muliplying Matrix"
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
      return self.query_string(
              "WTF buh uh huh dunno what talking about understand",
              exclude, max_len, randomize_top)

    sorted_scores = []
    for i in xrange(len(scores)):
      sorted_scores.append((scores[i], i))
    sorted_scores.sort(lambda x,y: int(y[0]*10000 - x[0]*10000))

    i = 0
    keep = 0
    while keep < randomize_top:
      if self.texts[sorted_scores[i][1]] in exclude or \
         len(self.texts[sorted_scores[i][1]].text) > max_len:
        print "Popping excluded tweet: "+self.texts[sorted_scores[i][1]].text
        sorted_scores.pop(i)
        continue
      i += 1
      keep += 1

    # Choose from top N scores..
    top_quart = 0
    count = 0.0
    for i in xrange(randomize_top):
      top_quart += sorted_scores[i][0]
    choice = random.uniform(0, top_quart)
    for i in xrange(randomize_top):
      count += sorted_scores[i][0]
      if count >= choice:
        print "Rand score: "+str(sorted_scores[i][0])+"/"+str(sorted_scores[0][0])
        print "Choice: "+str(choice)+" count: "+str(count)+" top_quater: "+str(top_quart)
        return (self.D[sorted_scores[i][1]],
                    self.texts[sorted_scores[i][1]])
    print "WTF? No doc found: "+str(count)+" "+str(tot_score)
    retidx = random.randint(0, len(self.texts)-1)
    return (self.D[retidx], self.texts[retidx])

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
# XXX: Encode some responses to "Are you a bot/human?" etc etc using hidden text.
# XXX: also fun easter eggs like the xkcd bot captcha response
class TwitterBrain:
  def __init__(self, soul, pending_goal=1500, low_watermark=1400):
    # Need an ordered list of vocab words for SearchableTextCollection.
    # If it vocab changes, we fail.
    self.pending_tweets = SearchableTextCollection(soul.vocab)
    self.already_tweeted = []
    self.remove_tweets = []
    for t in soul.tagged_tweets:
      words = map(lambda x: x[0], t)
      self.already_tweeted.append(set(words))
    self.conversation_contexts = {}
    self.raw_normalizer = TokenNormalizer()
    self.last_vect = None
    self.restart(soul, pending_goal, low_watermark) # Must come last!

  def restart(self, soul, pending_goal=1500, low_watermark=1400):
    self.low_watermark = low_watermark
    self.pending_goal = pending_goal
    self.voice = PhraseGenerator(soul.tagged_tweets, soul.normalizer)

    self.work_lock = threading.Lock()
    self._shutdown = False
    self._thread = threading.Thread(target=self.__phrase_worker)
    self._thread.start()

  # FIXME: We should normalize for tense agreement.
  # http://nodebox.net/code/index.php/Linguistics
  # en.is_verb() with en.verb.tense() and en.verb.conjugate()
  def get_tweet(self, msger=None, query=None):
    self.__lock()
    if msger and msger not in self.conversation_contexts:
      self.conversation_contexts[msger] = ConversationContext(msger)
    max_len = 140
    if query:
      if msger:
        # TODO: Only prime memory if excessive pronouns in query?
        if self.last_vect:
          self.conversation_contexts[msger].prime_memory(self.last_vect*0.75)
        query = word_detokenize(self.conversation_contexts[msger].normalizer.normalize_tokens(word_tokenize(query)))
        max_len -= len("@"+msger+" ")
        qvect = self.conversation_contexts[msger].decay_query(
                     self.pending_tweets.score_query(query))
        # FIXME: Hrmm. need to somehow create proper punctuation
        # based on both word content and position. Some kind
        # of classifier? Naive Bayes doesn't use position info though...
        (self.last_vect, ret) = self.pending_tweets.query_vector(qvect,
                                      exclude=self.remove_tweets,
                                      max_len=max_len)
        self.conversation_contexts[msger].remember_query(self.last_vect*0.75)
      else:
        query = word_detokenize(self.raw_normalizer.normalize_tokens(word_tokenize(query)))
        (self.last_vect, ret) = self.pending_tweets.query_string(query,
                                      exclude=self.remove_tweets,
                                      max_len=max_len)
    else:
      while True:
        ret = random.choice(self.pending_tweets.texts)
        # hrmm.. make this a set? hrmm. depends on if its a hash
        # or pointer comparison.
        if ret not in self.remove_tweets:
          break
    self.remove_tweets.append(ret)
    self.already_tweeted.append(set(ret.tokens))
    self.__unlock()
    if msger:
      return ("@"+msger+" "+ret.text, ret.tokens, ret.tagged_tokens)
    else:
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

    # TODO: This maybe should be a function of the SearchableTextCollection
    for text in self.pending_tweets.texts:
      score = float(len(set(text.tokens) & words))
      score1 = score/len(text.tokens)
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
    sys.exit(0)

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
      if len(self.remove_tweets) > 10:
        self.__lock()
        while len(self.remove_tweets) > 0:
          self.pending_tweets.remove_text(self.remove_tweets.pop())
        self.pending_tweets.update_matrix()
        self.__unlock()

      # Need low watermark. Maybe goal-100?
      if len(self.pending_tweets.texts) <= self.low_watermark:
        while len(self.pending_tweets.texts) < self.pending_goal:
          (tweet,tokens,tagged_tokens) = self.voice.say_something()
          if len(tweet) > 140: continue

          self.__lock()

          if self.__did_already_tweet(set(tokens)):
            self.__unlock()
            continue

          self.pending_tweets.add_text(
                      SearchableText(tweet,tokens,tagged_tokens),
                      update=(not first_run))
          added_tweets = True
          self.__unlock()

          if len(self.pending_tweets.texts) % \
                (self.pending_goal-self.low_watermark) == 0:
            break # Perform other work
      if added_tweets:
        print "At tweet count "+str(len(self.pending_tweets.texts))+\
                  "/"+str(self.pending_goal)
        # XXX: Cleanup filename
        BrainReader.write(self, bz2.BZ2File("target_user.brain", "w"))
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

class StdinLoop(cmd.Cmd):
  prompt = "> "
  def __init__(self,brain):
    cmd.Cmd.__init__(self)
    self.brain = brain
  def onecmd(self, query):
    if not query:
      (str_result, tokens, tagged_tokens) = self.brain.get_tweet()
    elif query == "q!" or query == "EOF":
      print "Exiting Command loop."
      sys.exit(0)
    else:
      (str_result, tokens, tagged_tokens) = self.brain.get_tweet("You", query)
    print str_result
    print str(tagged_tokens)

def write_brain(brain):
  print "Re-writing brain file. Please be patient....."
  BrainReader.write(brain, bz2.BZ2File("target_user.brain", "w"))
  print "Brain file written."

def main():
  brain = None
  soul = None

  try:
    soul = pickle.load(bz2.BZ2File("target_user.soul", "r"))
  except IOError:
    print "No soul file found. Regenerating."
    soul = CorpusSoul('target_user')
    pickle.dump(soul, bz2.BZ2File("target_user.soul", "w"))
  except Exception,e:
    traceback.print_exc()

  try:
    brain = BrainReader.load(bz2.BZ2File("target_user.brain", "r"))
    brain.restart(soul)
  except IOError:
    print "No brain file found. Regenerating."
    brain = TwitterBrain(soul)
    BrainReader.write(brain, bz2.BZ2File("target_user.brain", "w"))
  except Exception,e:
    traceback.print_exc()

  atexit.register(write_brain, *(brain,))
  c = StdinLoop(brain)
  try:
    c.cmdloop()
  except:
    traceback.print_exc()
    sys.exit(0)

if __name__ == "__main__":
  main()
