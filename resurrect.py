#!/usr/bin/python

# Resurrecter: Takes a soul file and animates it.

try:
  import psyco
  psyco.full()
except:
  print "Psyco JIT not found. Queries will run MUCH slower."

import gzip
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
import en
import os
#import twitter

from libs.tokenizer import word_tokenize, word_detokenize
from libs.SpeechModels import PhraseGenerator, TokenNormalizer
from extract import CorpusSoul

from ConfigParser import SafeConfigParser
config = SafeConfigParser()
config.read('settings.cfg')

import easter_eggs

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
# EX: existential there
class QueryStripper:
  kill_pos = set(["RP", "CC", "MD", "PDT", "POS", "TO", "WDT", "WP", "WP$",
                  "WRB", "DT", "EX", "."])
  kill_words = set(["am", "are", "is", "was", "were"])
  @classmethod
  def strip_tagged_query(cls, tagged_query):
    ret = []
    for t in tagged_query:
      if t[0].lower() not in cls.kill_words \
         and t[1] not in cls.kill_pos and not curses.ascii.ispunct(t[0][0]):
        ret.append(t)
    return ret

#  - Subsitute @msgs: I/Me->You, Mine->Yours, My->Your
class PronounInverter:
  pronoun_21_map = { "you":"i me", "yours":"mine","your":"my" }
  pronoun_12_map = { "i":"you", "me":"you", "mine":"yours","my":"your" }

  @classmethod
  def make_2nd_person(cls, string):
    for i in cls.pronoun_12_map.iterkeys():
      string = re.sub(r"(?i)([^a-zA-Z]|^)"+i+"([^a-zA-Z]|$)",
                      "\\1"+cls.pronoun_12_map[i]+"\\2", string)
    return string

  @classmethod
  def make_1st_person(cls, string):
    for i in cls.pronoun_21_map.iterkeys():
      string = re.sub(r"(?i)([^a-zA-Z]|^)"+i+"([^a-zA-Z]|$)",
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
  def __init__(self, nick,
               decay=(1.0-config.getfloat("brain","memory_decay_rate"))):
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
    # Normalize these after summing to prevent score inflation.
    norm = math.sqrt(numpy.dot(self.memory_vector,self.memory_vector))
    self.memory_vector /= norm
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

class TextWordInfo:
  def __init__(self):
    self.vector_idx = -1 # index in the global vocab/score vector
    self.count = 0 # occurrence in parent text body
    self.pos_counts = {} # POSTrimmed nltk tag

class SearchableText:
  def __init__(self, text, tokens=None, tagged_tokens=None, strip=False,
               hidden_text="",
               generalize_terms=config.getboolean('brain', 'generalize_terms')):
    if hidden_text:
      hidden_text = hidden_text.rstrip()
      if not curses.ascii.ispunct(hidden_text[-1]):
        hidden_text += ". "
      else:
        hidden_text += " "
    self.hidden_text = hidden_text

    if not curses.ascii.ispunct(text[-1]): text += "."

    self.tagged_tokens = tagged_tokens
    if not tokens: tokens = word_tokenize(text)
    self.text = text

    if hidden_text:
      tokens.extend(word_tokenize(hidden_text))

    # Include hidden text in search tokens
    pos_tags = nltk.pos_tag(tokens)

    if strip:
      search_tokens = [porter.stem(t[0]).lower() for t in
                            QueryStripper.strip_tagged_query(pos_tags)]
    else:
      search_tokens = [porter.stem(t[0]).lower() for t in pos_tags]

    self.word_info = {}
    self.total_words = 0
    if generalize_terms:
      # Add senses, antonyms, and hypernyms to this list with
      # http://nodebox.net/code/index.php/Linguistics
      # Also add normalized versions with en.spelling() first

      # FIXME: This is biasing results. Words with lots of hyponyms are being
      # favored by TF-IDF. We need word sense disambiguation to prune this
      # down.
      # http://groups.google.com/group/nltk-users/browse_thread/thread/ad191241e5d9ee78
      for v in xrange(len(search_tokens)):
        sv = search_tokens[v]
        add_terms = set([sv])
        # en.spelling.correct(v)
        tag = POSTrim.trim(pos_tags[v][1])
        mod = None
        if tag == "NN": mod = en.noun
        elif tag == "JJ": mod = en.adjective
        elif tag == "VB": mod = en.verb
        elif tag == "RB": mod = en.adverb
        else: mod = en.wordnet
        if mod:
          #add_terms.update(en.list.flatten(mod.senses(sv)))
          add_terms.update(en.list.flatten(mod.antonym(sv)))
          add_terms.update(en.list.flatten(mod.hypernym(sv)))
          add_terms.update(en.list.flatten(mod.hyponym(sv)))

        for t in add_terms:
          if t not in self.word_info:
            self.word_info[t] = TextWordInfo()
          if tag not in self.word_info[t].pos_counts:
            self.word_info[t].pos_counts[tag] = 0
          self.word_info[t].count += 1
          self.word_info[t].pos_counts[tag] += 1
          self.total_words += 1
    else:
      for t in search_tokens:
        if t not in self.word_info:
          self.word_info[t] = TextWordInfo()
        if tag not in self.word_info[t].pos_counts:
          self.word_info[t].pos_counts[tag] = 0
        self.word_info[t].count += 1
        self.word_info[t].pos_counts[tag] += 1
        self.total_words += 1

  def tokens(self):
    # FIXME: If we decide to drop tagged_tokens, switch to saving
    # just the tokens
    if self.tagged_tokens:
      retlist = [t[0] for t in self.tagged_tokens]
    else:
      retlist = word_tokenize(self.text)
    if self.hidden_text:
      retlist.extend(word_tokenize(self.hidden_text))
    return retlist

  def count(self, word):
    if word in self.word_info: return self.word_info[word].count
    else: return 0


# Search:
#  1. Tag, tolower(), stem
#  2. if strip: strip all but nouns, adj, verbs
#  3. tf-idf to score text record from collection
#  4. If score is 0, requery for "WTF buh uh huh dunno what talking about"
#  5. Return probabilistically based on score with some cutoff

class SearchableTextCollection:
  def __init__(self, vocab, texts=[],
               generalize_terms=config.getboolean('brain', 'generalize_terms')):
    self._idf_cache = {}
    self.texts = texts
    if generalize_terms:
      # Need to stem vocab. SearchableTexts are all stemmed.
      vocab = set([porter.stem(t).lower() for t in vocab])
      new_terms = set()
      print "Generalizing vocabulary"
      for v in vocab:
        # Actually, no need to spell check here. It's probably
        # spelled right eventually, and this is SLOW!
        sv = v #en.spelling.correct(v)
        new_terms.update(sv)
        # Use parts of speech. en.wordnet is not a superset..
        for mod in [en.wordnet, en.adjective, en.noun, en.verb, en.adverb]:
          #new_terms.update(en.list.flatten(mod.senses(sv)))
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
    if not self.needs_update:
      print "No update needed"
      return
    print "Computing score matrix."
    self.D = []
    for doc in self.texts:
      d = []
      for dt in self.vocab:
        if dt in doc.word_info: d.append(self.tf_idf(dt, doc))
        else: d.append(0.0)
      d = numpy.array(d)
      norm = math.sqrt(numpy.dot(d,d))
      if norm <= 0:
        print "Zero row in matrix: "+doc.text
      else:
        d /= math.sqrt(numpy.dot(d,d))
      # Needed to downcast. Too much memory...
      self.D.append(d.astype(numpy.float32))
    print "Computed score matrix."
    self.needs_update = False

  def score_query(self, query_string):
    if self.needs_update: self.update_matrix()
    query_string = PronounInverter.invert_all(query_string)
    print "Inverted Query: "+query_string
    query_text = SearchableText(query_string, strip=True)

    print "Building Qvector.."
    # XXX: Amplify nouns and dampen adjectives and verbs:
    # 1. divide current score by count
    # 2. amplify by linear combo of ratios in config
    # 3. re-normalize
    q = []
    for dt in self.vocab:
      if dt in query_text.word_info:
        # XXX: Need to produce a tagged score for this. Some
        # kind of aux structure that links tagged word to score-index.
        # The other problem is that the SearchableText has wordnet data in it
        # too.
        q.append(self.tf_idf(dt, query_text))
      else: q.append(0.0)

    q = numpy.array(q)
    # Numpy is retarded here.. Need to special case 0
    norm = math.sqrt(numpy.dot(q,q))
    if norm > 0: q /= norm
    else: print "Zero vector"
    print "Qvector built"
    return q

  def query_string(self, query_string, exclude=[],
                   max_len=config.getint("brain","tweet_len"),
                   randomize_top=1):
    if not query_string:
      while True:
        retidx = random.randint(0, len(self.texts)-1)
        ret = self.texts[retidx]
        # hrmm.. make this a set? hrmm. depends on if its a hash
        # or pointer comparison.
        if not ret.hidden_text and ret not in exclude:
          break
      return (0, self.D[retidx], ret)

    return self.query_vector(self.score_query(query_string), exclude, max_len,
                      randomize_top)

  def query_vector(self, query_vector, exclude=[],
                   max_len=config.getint("brain","tweet_len"),
                   randomize_top=1):
    q = query_vector

    # FIXME: Could actually turn this into a proper numpy matrix multiply
    # for speed.
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

    # FIXME: Could also eliminate this sort if we decide we never
    # want to randomize.
    sorted_scores = []
    for i in xrange(len(scores)):
      sorted_scores.append((scores[i], i))
    sorted_scores.sort(lambda x,y: int(y[0]*100000000 - x[0]*100000000))

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
        self.print_score(q, self.D[sorted_scores[i][1]])
        print "Rand score: "+str(sorted_scores[i][0])+"/"+str(sorted_scores[0][0])
        print "Choice: "+str(choice)+" count: "+str(count)+" top_quater: "+str(top_quart)
        return (sorted_scores[i][0], self.D[sorted_scores[i][1]],
                    self.texts[sorted_scores[i][1]])
    print "WTF? No doc found: "+str(count)+" "+str(tot_score)
    retidx = random.randint(0, len(self.texts)-1)
    return (sorted_scores[retidx][0], self.D[retidx], self.texts[retidx])

  def print_score(self, q, resp):
    for i in xrange(len(resp)):
      if q[i]*resp[i] > 0:
        print "Found match |"+self.vocab[i]+"| score: "+str(q[i]*resp[i])

  def tf(self, term, text):
    """ The frequency of the term in text. """
    return float(text.count(term)) / text.total_words

  def idf(self, term):
    """ The number of texts in the corpus divided by the
    number of texts that the term appears in. 
    If a term does not appear in the corpus, 0.0 is returned. """
    # idf values are cached for performance.
    idf = self._idf_cache.get(term)
    if idf is None:
      # Decided to sum total counts, not just membership..
      #matches = len(list(True for text in self.texts if term in text.word_info))
      matches = sum(list(text.word_info[term].count for text in self.texts if
term in text.word_info))
      if not matches:
        idf = 0.0
      else:
        tot_words = sum(list(text.total_words for text in self.texts))
        idf = math.log(float(tot_words) / matches)
        #idf = math.log(float(len(self.texts)) / matches)
      self._idf_cache[term] = idf
    return idf

  def tf_idf(self, term, text):
    return self.tf(term, text) * self.idf(term)

# The brain holds the already tweeted list and the list of pending tweets
# as a SearchableTextCollection.
#
# It also manages a separate worker thread for generating more pending tweets.
class TwitterBrain:
  def __init__(self, soul):
    # Need an ordered list of vocab words for SearchableTextCollection.
    # If it vocab changes, we fail.
    for t in easter_eggs.xkcd: soul.vocab.update(t.word_info.iterkeys())
    self.pending_tweets = SearchableTextCollection(soul.vocab)
    for t in easter_eggs.xkcd: self.pending_tweets.add_text(t)
    self.already_tweeted = []
    self.remove_tweets = []
    for t in soul.tagged_tweets:
      words = map(lambda x: x[0], t)
      self.already_tweeted.append(set(words))
    self.conversation_contexts = {}
    self.raw_normalizer = TokenNormalizer()
    self.last_vect = None
    self.restart(soul) # Must come last!

  def restart(self, soul):
    if config.getfloat("brain","tweet_pool_multiplier") > 0:
      self.pending_goal = min(len(soul.tagged_tweets) * \
           config.getfloat("brain","tweet_pool_multiplier"),
                   config.getint("brain", "tweet_pool_max"))
    else:
      self.pending_goal = config.getint("brain", "tweet_pool_max")
    self.low_watermark = self.pending_goal - 85 # FIXME: config?
    self.voice = PhraseGenerator(soul.tagged_tweets, soul.normalizer,
                          config.getint("brain", "hmm_context"),
                          config.getint("brain", "hmm_offset"))
    if self.pending_tweets.needs_update:
      self.pending_tweets.update_matrix()
    self.work_lock = threading.Lock()
    self._shutdown = False
    self._thread = threading.Thread(target=self.__phrase_worker)
    self._thread.start()

  # TODO: We could normalize for tense agreement... might be a bad idea
  # though.
  # http://nodebox.net/code/index.php/Linguistics
  # en.is_verb() with en.verb.tense() and en.verb.conjugate()
  def get_tweet(self, msger=None, query=None, followed=False):
    self.__lock()
    if msger and msger not in self.conversation_contexts:
      self.conversation_contexts[msger] = ConversationContext(msger)
    max_len = config.getint("brain","tweet_len")
    if query and msger:
      # XXX: Only prime memory if no nouns in query, or if only nouns
      # present are also present in the last_vect.
      # If other nouns and no noun matches, ignore last_vect,
      # otherwise if other nouns dampen it by 0.25-0.5
      # otherwise, don't dampen
      if self.last_vect != None:
        self.conversation_contexts[msger].prime_memory(self.last_vect)
      query = word_detokenize(self.conversation_contexts[msger].normalizer.normalize_tokens(word_tokenize(query)))
      max_len -= len("@"+msger+" ")
      # FIXME: Hrmm, should we add followed tweets to the memory even if
      # we decide to ignore them?
      # XXX: nltk.pos_tag doesn't do so well if the first word in a question
      # is capitalized
      qvect = self.conversation_contexts[msger].decay_query(
                   self.pending_tweets.score_query(query))
      (score, last_vect, ret) = self.pending_tweets.query_vector(qvect,
                                      exclude=self.remove_tweets,
                                      max_len=max_len)

      if followed:
        min_score = config.getfloat('brain', 'min_follow_reply_score')
      else:
        min_score = config.getfloat('brain', 'min_msg_reply_score')

      if score >= min_score:
        self.last_vect = last_vect
      else:
        print "Minimum score of "+str(min_score)+" not met: "+str(score)
        print str(ret.tagged_tokens)
        print "Not responding with: "+ret.text
        return None
      self.conversation_contexts[msger].remember_query(self.last_vect)
    else:
      (score, self.last_vect, ret) = self.pending_tweets.query_string(query,
                                      exclude=self.remove_tweets,
                                      max_len=max_len)
    self.remove_tweets.append(ret)
    tokens = ret.tokens()
    self.already_tweeted.append(set(tokens))
    self.__unlock()
    print str(ret.tagged_tokens)
    if msger:
      return "@"+msger+" "+ret.text
    else:
      return ret.text

  def __did_already_tweet(self, words,
                          max_score=config.getfloat("brain","max_shared_word_ratio")):
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
      tokens = set(text.tokens())
      score = float(len(tokens & words))
      score1 = score/len(tokens)
      score2 = score/len(words)
      if score1 > max_score or score2 > max_score:
        #print "Too similar to pending tweet.. skipping: "+\
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

  # TODO: Maybe this whole tweet pool model is the wrong way to go.
  # We could try influencing the HMM's probabilities directly using a query,
  # but I'm not sure how to do that and not get utter nonsense.
  def __phrase_worker2(self):
    first_run = True
    while not self._shutdown:
      added_tweets = False
      if len(self.remove_tweets) > 90: # FIXME: config?
        self.__lock()
        while len(self.remove_tweets) > 0:
          self.pending_tweets.remove_text(self.remove_tweets.pop())
        added_tweets = True
        self.__unlock()

      # Need low watermark. Maybe goal-100?
      if len(self.pending_tweets.texts) <= self.low_watermark:
        while len(self.pending_tweets.texts) < self.pending_goal:
          self.__lock()
          (tweet,tokens,tagged_tokens) = self.voice.say_something()

          if len(tweet) > config.getint("brain", "tweet_len") \
                or self.__did_already_tweet(set(tokens)):
            self.__unlock()
            continue

          self.pending_tweets.add_text(
                      SearchableText(tweet,tokens,tagged_tokens),
                      update=(not first_run))
          print "At tweet count "+str(len(self.pending_tweets.texts))+\
                  "/"+str(self.pending_goal)
          added_tweets = True
          self.__unlock()

          if len(self.pending_tweets.texts) % 100 == 0: # FIXME: config?
            break # Perform other work

      time.sleep(2)

      if len(self.pending_tweets.texts) == self.pending_goal and added_tweets:
        print "At full tweet count "+str(self.pending_goal)
        self.__lock()
        self.pending_tweets.update_matrix()
        self.__unlock()
        first_run=False
        BrainReader.write(self, "target_user.brain")
      elif added_tweets:
        self.__lock()
        print "At tweet count "+str(len(self.pending_tweets.texts))+\
                  "/"+str(self.pending_goal)
        self.pending_tweets.update_matrix()
        self.__unlock()
        if (len(self.pending_tweets.texts) % \
               config.getint("brain","save_brain_every")) == 0:
          BrainReader.write(self, "target_user.brain")
      time.sleep(2)

# Lousy hmm can't be pickled
class BrainReader:
  @classmethod
  def write(cls, brain, fname,
            do_gzip=config.getboolean("brain", "gzip_brain")):
    brain.work_lock.acquire()
    (voice,thread,lock) = (brain.voice,brain._thread,brain.work_lock)
    (brain.voice,brain._thread,brain.work_lock) = (None,None,None)
    (D,needs_update) = (brain.pending_tweets.D,brain.pending_tweets.needs_update)
    (brain.pending_tweets.D, brain.pending_tweets.needs_update) = (None,True)
    print "Writing brain file..."
    if do_gzip: pickle.dump(brain, gzip.GzipFile(fname+".part", "w"))
    else: pickle.dump(brain, open(fname+".part", "w"))
    print "Brain file written..."
    (brain.voice,brain._thread,brain.work_lock) = (voice,thread,lock)
    (brain.pending_tweets.D, brain.pending_tweets.needs_update) = (D,needs_update)
    os.rename(fname+".part", fname)
    lock.release()

  @classmethod
  def load(cls, fname):
    try:
      print "Loading brain file. This may take some time..."
      brain = pickle.load(open(fname, "r"))
    except pickle.UnpicklingError:
      brain = pickle.load(gzip.GzipFile(fname, "r"))
    except KeyError:
      brain = pickle.load(gzip.GzipFile(fname, "r"))
    print "Brain file loaded."
    return brain

class StdinLoop(cmd.Cmd):
  prompt = "> "
  def __init__(self,brain):
    cmd.Cmd.__init__(self)
    self.brain = brain
  def onecmd(self, query):
    if not query:
      str_result = self.brain.get_tweet()
    elif query == ":w":
      BrainReader.write(self, "target_user.brain")
    elif query == ":wq":
      BrainReader.write(self, "target_user.brain")
      print "Exiting Command loop."
      sys.exit(0)
    elif query == ":q!" or query == "EOF":
      print "Exiting Command loop."
      sys.exit(0)
    else:
      str_result = self.brain.get_tweet("You", query)
    print str_result

def main():
  brain = None
  soul = None

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
    if config.getboolean("soul","gzip_soul"):
      pickle.dump(soul, gzip.GzipFile("target_user.soul", "w"))
    else:
      pickle.dump(soul, open("target_user.soul", "w"))
  except Exception,e:
    traceback.print_exc()

  soul.normalizer.verify_scores()

  try:
    brain = BrainReader.load("target_user.brain")
    brain.restart(soul)
  except IOError:
    print "No brain file found. Regenerating."
    brain = TwitterBrain(soul)
    BrainReader.write(brain, "target_user.brain")
  except Exception,e:
    traceback.print_exc()

  #for i in brain.pending_tweets.texts: print i.text
  #sys.exit(0)

  c = StdinLoop(brain)
  try:
    c.cmdloop()
  except:
    traceback.print_exc()
    sys.exit(0)

if __name__ == "__main__":
  main()
