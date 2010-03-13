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

import libs.hmm
import libs.treebank

# TODO: Use agfl for pos_tag. nltk.pos_tag kind of sucks
# Agfl sucks too, but in a different way. It's better
# with parts of speach and grammar, but worse with
# contractions. We could use nltk to normalize out the
# contractions and then use agfl to get a tagged parse
# structure. We really only care about the labled terminals.
# agfl-run -T 1 -B -b ep4ir
# We still would have to reject a lot of tweets though.
# There are a lot of them agfl parses as fragments with
# unlabeled terminals.. It also needs really clean sentences.
# We could translate the AGFL tags to nltk tags, and then
# fall back to nltk when AGFL fails.

# XXX: Consider temporarily stripping out stuff we left in (esp for AGFL)
# ["#", "*", "@", "/"], and urls -> "Noun"
def pos_tag(tokens):
  return nltk.pos_tag(tokens)

toker = libs.treebank.TreebankWordTokenizer()
def word_tokenize(text):
  return toker.tokenize(text)

def word_detokenize(tokens):
  return toker.detokenize(tokens)


# Stages:
# 1. nltk tokenize
# 2. Normalize
# 3. Attempt AGFL
# 4. If no result, nltk.pos_tag

# FIXME: Can we do anything clever with "!1!1!1" and "?//?/1!/"? Maybe a regex?
class TokenNormalizer:
  def __init__(self):
    # Tokens to remove entirely
    self.skip_tokens = set(['"','(',')','[',']'])

    # One-to-one mapping, no context
    self.mono_map = { "u":"you", "r":"are", "m":"am", "c":"see", "n":"and",
                    "h8":"hate", "<3":"love", "thx":"thanks",
                    "teh":"the", "fb":"Facebook", "2nite":"tonight",
                    "ur" :"your",
                    "ya":"you", "yah":"yes", #fishy
                    "i":"I" }

    # Make sure capital words are always capitalized for POS tag
    f = open("/usr/share/dict/words", "r")
    for word in f.readlines():
      if curses.ascii.isupper(word[0]):
        self.mono_map[word.lower()] = word

    # Store following (if matched) in map, then return tuple
    self.dual_map = { ("gon", "na") : ("going", "to"),
                     ("no", "thx") : ("no", "thanks"),
                     ("got", "ta") : ("got", "to"),
                     ("wan", "na") : ("want", "to"),
                     ("you", "d") : ("you", "should"),
                     ("you", "ll") : ("you", "will"),
                     ("i", "d") : ("I", "should"),
                     ("you", "re") : ("you", "are"),
                     ("you", "ve") : ("you", "have"),
                     ("ca", "nt") : ("can", "not"),
                     ("ca", "n't") : ("can", "not"),
                     ("wo", "n't") : ("will", "not"),
                     ("get", "cha") : ("get", "you"),
                     ("get", "ya") : ("get", "you"),
                     ("up", "2") : ("up", "to"),
                     ("t", "is") : ("it", "is"),
                     ("t", "was") : ("it", "was"),
                     ("i", "ve") : ("I", "have"),
                     ("i", "ll") : ("I", "will"),
                     ("i", "m") : ("I", "am"),
                     ("wha", "chu") : ("what", "you"), # hack...
                     ("what", "chu") : ("what", "you"), # hack...
                     ("wha", "cha") : ("what", "you"), # hack...
                     ("what", "cha") : ("what", "you"), # hack...
                     ("whad", "ya") : ("what", "you") } # hack...

    # Store preceeding term in map
    self.dual_pre = { "'re":"are", "'m": "am", "n't":"not",
                         "'d":"should", "'ll":"will", "'ve":"have",
                         "ve":"have" }

    # Store following term in map
    self.dual_post = { "gim":"give", "lem":"let" }

    self.mono_denorm = {}
    self.dual_denorm = {}

    self.mono_counts = {}
    self.mono_totals = {}

    self.dual_counts = {}
    self.dual_totals = {}

  def _normalize(self, word, next_word):
    word_orig = word
    word = word.lower()

    if next_word:
      isupper = word_orig.isupper() and next_word.isupper()
      next_word_orig = next_word
      next_word = next_word.lower()
      tup = (word, next_word)

      if tup in self.dual_map:
        norm_tup = self.dual_map[tup]
        norm_tup = (norm_tup[0].lower(), norm_tup[1].lower())

        if norm_tup not in self.dual_counts: self.dual_counts[norm_tup] = 0
        self.dual_counts[norm_tup] += 1

        # Preserve case for I
        if isupper or norm_tup[0] == "i": tupd = (word_orig, next_word_orig)
        else: tupd = tup
        if norm_tup not in self.dual_denorm:
          self.dual_denorm[norm_tup] = {}
        if tupd not in self.dual_denorm[norm_tup]:
          self.dual_denorm[norm_tup][tupd] = 1
        else: self.dual_denorm[norm_tup][tupd] += 1

        if isupper:
          return (self.dual_map[tup][0].upper(), self.dual_map[tup][1].upper())
        else: return self.dual_map[tup]

      if word in self.dual_post:
        ret = self.dual_post[word]
        norm_tup = (word, ret.lower())

        if norm_tup not in self.dual_counts: self.dual_counts[norm_tup] = 0
        self.dual_counts[norm_tup] += 1

        # Preserve case for I
        if isupper or norm_tup[0] == "i": tupd = (word_orig, next_word_orig)
        else: tupd = tup
        if norm_tup not in self.dual_denorm:
          self.dual_denorm[norm_tup] = {}
        if tupd not in self.dual_denorm[norm_tup]:
          self.dual_denorm[norm_tup][tupd] = 1
        else: self.dual_denorm[norm_tup][tupd] += 1

        if isupper: return (ret.upper(), next_word_orig)
        else: return (ret, next_word_orig)

      if next_word in self.dual_pre:
        ret = self.dual_pre[next_word]
        norm_tup = (word, ret.lower())

        if norm_tup not in self.dual_counts: self.dual_counts[norm_tup] = 0
        self.dual_counts[norm_tup] += 1

        # Preserve case for I
        if isupper or norm_tup[0] == "i": tupd = (word_orig, next_word_orig)
        else: tupd = tup
        if norm_tup not in self.dual_denorm:
          self.dual_denorm[norm_tup] = {}
        if tupd not in self.dual_denorm[norm_tup]:
          self.dual_denorm[norm_tup][tupd] = 1
        else: self.dual_denorm[norm_tup][tupd] += 1

        if isupper: return (word_orig, ret.upper())
        else: return (word_orig, ret)

    if word in self.mono_map:
      isupper = word.isupper()
      norm_word = self.mono_map[word].lower()
      if norm_word not in self.mono_counts: self.mono_counts[norm_word] = 0
      self.mono_counts[norm_word] += 1

      if word not in self.mono_denorm:
        self.mono_denorm[norm_word] = {}
      if word_orig not in self.mono_denorm[norm_word]:
        self.mono_denorm[norm_word][word_orig] = 1
      else: self.mono_denorm[word][word_orig] += 1

      if isupper: return (self.mono_map[word].upper(),)
      else: return (self.mono_map[word],)

    return (word_orig,)

  # Normalizing tags for hmm training might cause us to lose
  # some of the colloquial feel of the target...
  def _denormalize(self, word, next_word):
    word_orig = word
    word = word.lower()
    next_word = next_word.lower()
    tup = (word, next_word)

    if tup in self.dual_counts:
      norm_score = float(self.dual_counts[tup])/self.dual_totals[tup]
      if norm_score > random.random():
        return self._choose_denorm(self.dual_denorm[tup])
    if word in self.mono_counts:
      norm_score = float(self.mono_counts[word])/self.mono_totals[word]
      if norm_score > random.random():
        return (self._choose_denorm(self.mono_denorm[word]),)
    return (word_orig,)

  def _choose_denorm(self, denorm_dict):
    tot_score = 0
    for d in denorm_dict.iterkeys():
      tot_score += denorm_dict[d]
    choose = random.randint(0,tot_score)
    score = 0
    for d in denorm_dict.iterkeys():
      score += denorm_dict[d]
      if score >= choose:
        return d

  def score_tokens(self, tokens):
    for t in tokens:
      curr = t.lower()
      if curr in self.mono_counts:
        if curr not in self.mono_totals:
          self.mono_totals[curr] = 1
        else:
          self.mono_totals[curr] += 1

    tok_len = len(tokens)
    i = 0
    while i < tok_len-1:
      tup = (tokens[i].lower(), tokens[i+1].lower())
      if tup in self.dual_counts:
        if tup not in self.dual_totals:
          self.dual_totals[tup] = 1
        else:
          self.dual_totals[tup] += 1
      i+=1

  def normalize_tokens(self, tokens):
    ret_tokens = []

    tok_len = len(tokens)
    i = 0
    while i < tok_len-1:
      # XXX: Should we skip urls? or do something more clever..
      if tokens[i] in self.skip_tokens: #or "://" in tokens[i]:
        i += 1
        continue
      new_toks = self._normalize(tokens[i], tokens[i+1])
      i += len(new_toks)
      ret_tokens.extend(new_toks)
    if i < tok_len:
      if tokens[i] in self.skip_tokens: # or "://" in tokens[i]:
        i += 1
      else: ret_tokens.extend(self._normalize(tokens[i], ""))
    # Add .
    if ret_tokens and not curses.ascii.ispunct(ret_tokens[-1][-1]):
      ret_tokens.append(".")
    return ret_tokens

  def denormalize_tokens(self, tokens):
    ret_tokens = []

    tok_len = len(tokens)
    i = 0
    while i < tok_len-1:
      new_toks = self._denormalize(tokens[i], tokens[i+1])
      i += len(new_toks)
      ret_tokens.extend(new_toks)

    if i < tok_len: ret_tokens.extend(self._denormalize(tokens[i], ""))
    return ret_tokens

  @classmethod
  def UnitTest(cls, norm=None):
    # Does it make me disturbed that these are the first sentences that came
    # to mind? Somewhat troubling...
    strings = ["Hi there. Gonna getcha. I've decided you'll die tonight.",
               "r u scared yet? Ill rip our ur guts.",
               "Whatcha up2? We're gonna go on a killing spree.",
               "Holy crap dood.",
               "Are you going out?",
               "#Hi @I love/hate $0 http://yfrog.com/a3ss0sa *always* don't /you/....",
               "r u going out?"]
    if not norm: norm = TokenNormalizer()
    tokens = []
    norm_tokens = []
    for s in strings:
      t=word_tokenize(s)
      tokens.append(t)
      print s
    print ""
    for t in tokens:
      nt = norm.normalize_tokens(t)
      norm_tokens.append(nt)
      print nt
    print ""
    for nt in norm_tokens:
      norm.score_tokens(nt)
    denorm_tokens = []
    for nt in norm_tokens:
      dt = norm.denormalize_tokens(nt)
      denorm_tokens.append(dt)
      print dt
    for dt in denorm_tokens:
      print word_detokenize(dt)

# Intuition: x-i-x-x is going to work better becuase hmm predicts
# current based on previous state already
#  1. HMM-3gram (x-i-x)
#  2. HMM-4gram (x-x-i-x or x-i-x-x)
#  3. HMM-5gram (x-x-i-x-x, x-i-x-x-x, x-x-i-x-x, x-x-x-i-x)
class PhraseGenerator:
  def __init__(self, tagged_phrases, normalizer=None):
    self.normalizer = normalizer
    # 41 is the winner so far..
    self.nm_hmm = {}
    self.nm_hmm["41"] = self._build_nm_hmm(tagged_phrases, 4, 1)

  def _build_nm_hmm(self, tagged_phrases, l, o):
    print "Prepping NM-HMM"
    hmm_states = set()
    hmm_symbols = set()
    hmm_sequences = list()

    for tags in tagged_phrases:
      #and len(tags) > l-1:
      hmm_symbols.update([t[0] for t in tags])
      sequence = []

      tag = ("^ "*(o))
      for t in xrange(0, min(l-o,len(tags))):
        tag += tags[t][1]+" "
      hmm_states.update(tag)
      #print "Adding initial Tag: "+str(tag)
      sequence.append((tags[0][0], tag))

      for i in xrange(1,len(tags)-1):
        tag = ""
        for t in xrange(i-o, min(i+(l-o),len(tags)-1)):
          tag += tags[t][1]+" "

        hmm_states.update(tag)
        sequence.append((tags[i][0], tag))

      tag = ""
      for t in xrange(-min(o,len(tags)),0):
        tag+=tags[t][1]+" "
      tag += ("|"*(l-o))
      hmm_states.update(tag)
      sequence.append((tags[-1][0], tag))

      #print "Adding Final Tag: "+str(tag)
      hmm_sequences.append(sequence)

    print "Training NM-HMM with "+str(len(hmm_states))+" states, "+\
            str(len(hmm_symbols))+" syms and "+\
            str(len(hmm_sequences))+" sequences"
    hmm_trainer = libs.hmm.HiddenMarkovModelTrainer(list(hmm_states),
                                                     list(hmm_symbols))
    nm_hmm = hmm_trainer.train_supervised(hmm_sequences)
    nm_hmm.length = l
    nm_hmm.offset = o
    print "Trained NM-HMM"
    return nm_hmm

  def _nm_hmm_phrase(self, mode="41"):
    # TODO: Maybe build mode on-demand?
    while True:
      try:
        return self.nm_hmm[mode].random_sample(random, 500,
                "|"*(self.nm_hmm[mode].length-self.nm_hmm[mode].offset))
      except:
        print "Probability failure..."
        traceback.print_exc()

  def say_something(self, tokens=None):
    if not tokens: tokens = self._nm_hmm_phrase(self)
    if self.normalizer:
      return word_detokenize(self.normalizer.denormalize_tokens(
                       map(lambda x: x[0], tokens)))
    else:
      return word_detokenize(map(lambda x: x[0], tokens))

  def test_run(self):
    for mode in map(str, [30, 31, 32, 40, 41, 42, 43]):
      print "Writing "+mode
      f = open(mode+".twt", "w")
      for i in xrange(1,50):
        print "  Writing "+str(i)
        result = self._nm_hmm_phrase(mode)
        f.write(word_detokenize(result)+"\n\n")

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
            if re.search("( |^)RT ", txt, re.IGNORECASE): continue
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

    # Need to compute finals scores for normalzier to be able to denormalize
    for t in tagged_tweets:
      self.normalizer.score_tokens(map(lambda x: x[0], t))

    self.tagged_tweets = tagged_tweets
    self.voice = PhraseGenerator(tagged_tweets, self.normalizer)
    #self.tweet_collection = SearchableTextCollection(tweet_texts)
    #self.tweet_collection.generate(100)

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
    soul = SoulWriter.load(open("target_user.soul", "r"))
  except Exception,e:
    traceback.print_exc()
    print "No soul file found. Regenerating."
    soul = CorpusSoul('target_user')
    SoulWriter.write(soul, open("target_user.soul", "w"))


  while True:
    query = raw_input("> ")
    if not query: query = "41"
    if query.isdigit():
      str_result = "1"*256
      while len(str_result) > 140:
        result = soul.voice._nm_hmm_phrase(query)
        str_result = soul.voice.say_something(result)
      print str(result)
      print str_result
    else:
      result = soul.voice._nm_hmm_phrase()
      print str(result)
      print soul.voice.say_something(result)
      #result = soul.tweet_collection.query(query, randomize=True)
      #print "|"+result.orig_text+"|"

if __name__ == "__main__":
  main()
