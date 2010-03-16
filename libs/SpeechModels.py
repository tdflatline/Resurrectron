#!/usr/bin/python

import curses.ascii

import random
import traceback
#import twitter

import hmm

from tokenizer import word_tokenize, word_detokenize

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
      # TODO: Should we skip urls? or do something more clever..
      if tokens[i] in self.skip_tokens or "://" in tokens[i]:
        i += 1
        continue
      new_toks = self._normalize(tokens[i], tokens[i+1])
      i += len(new_toks)
      ret_tokens.extend(new_toks)
    if i < tok_len:
      if tokens[i] in self.skip_tokens or "://" in tokens[i]:
        i += 1
      else: ret_tokens.extend(self._normalize(tokens[i], ""))

    if ret_tokens:
      # TODO: Trailing :'s are an artifact of stripping urls.
      if ret_tokens[-1] == ":":
        ret_tokens = ret_tokens[0:-1]
      elif ret_tokens[-1][-1] == ":":
        ret_tokens[-1] = ret_tokens[-1][0:-1]
      # Add .
      if not curses.ascii.ispunct(ret_tokens[-1][-1]):
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
    hmm_trainer = hmm.HiddenMarkovModelTrainer(list(hmm_states),
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
        return

  def hack_grammar(self, tokens):
    # "a/an",
    for i in xrange(len(tokens)-1):
      if tokens[i] == "a" and tokens[i+1][0] in ["aeuio"]:
        tokens[i] = "an"
      elif tokens[i] == "an" and tokens[i+1][0] not in ["aeuio"]:
        tokens[i] = "a"

  def say_something(self, tagged_tokens=None):
    if not tagged_tokens: tagged_tokens = self._nm_hmm_phrase()
    toks = [t[0] for t in tagged_tokens]
    self.hack_grammar(toks)
    if self.normalizer:
      tokens = self.normalizer.denormalize_tokens(toks)
      something=word_detokenize(tokens)
    else:
      tokens = toks
      something=word_detokenize(toks)
    return (something, tokens, tagged_tokens)

  def test_run(self):
    for mode in map(str, [30, 31, 32, 40, 41, 42, 43]):
      print "Writing "+mode
      f = open(mode+".twt", "w")
      for i in xrange(1,50):
        print "  Writing "+str(i)
        result = self._nm_hmm_phrase(mode)
        f.write(word_detokenize(result)+"\n\n")

def main():
  TokenNormalizer.UnitTest()

if __name__ == "__main__":
  main()