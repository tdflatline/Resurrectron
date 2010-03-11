#!/usr/bin/python

try:
  import psyco
  psyco.full()
except: pass

import nltk
import simplejson as json
import os
import re
from nltk.corpus import PlaintextCorpusReader
import pickle
import random
import math
import numpy
import curses.ascii
import traceback
#import twitter


# XXX: Write test code to verify we can normalize and denormalize back to the
# original string. This might not be possible yet.
class Twictionary:
  def __init__(self):
    # XXX: Pre-tokenization:
    #   "Ive":"I've", "Im":"I'm"
    # But how to invert?
    # http://twictionary.pbworks.com/
    self.norm_d = { "u":"you", "r":"are", "m":"am", "c":"see", "n":"and",
                    "wo":"will", "ca":"can", # hacks for won't, can't
                    "h8":"hate", "<3":"love",
                    "teh":"the", "FB":"Facebook",
                    "'re":"are", "'m": "am", "n't":"not",
                    "'d":"would", "'ll":"will", "'ve":"have", "i":"I" }
    self.denorm_d = {}

    for w in self.norm_d.iterkeys():
      self.denorm_d[self.norm_d[w]] = w

    # XXX: Ambiguity.. Handle by score? Or orig_text?
    #del self.denorm_d["will"]
    #del self.denorm_d["am"]
    #del self.denorm_d["are"]

    self.norm_score = {}
    for w in self.denorm_d.iterkeys():
      self.norm_score[w] = [0.0, 0.0]

    #self.post_pos_d = { "r/V[\S]+" : "are", "r/N[\S]+" : "our/PRP$", "'s/VB"/ }

    # Make sure capital words are always capitalized for POS tag
    # FIXME: Needed?
    f = open("/usr/share/dict/words", "r")
    for word in f.readlines():
      if curses.ascii.isupper(word[0]):
        self.norm_d[word.lower()] = word

  def normalize(self, word):
    word_orig = word
    if word.isupper(): isupper = True
    else: isupper = False
    word = word.lower()
    if word in self.denorm_d:
      self.norm_score[word][0] += 1.0
    if word in self.norm_d:
      norm = self.norm_d[word]
      self.norm_score[norm][1] += 1.0
      if isupper: return norm.upper()
      else: return norm
    else:
      return word_orig

  # Normalizing tags for hmm training might cause us to lose
  # some of the colloquial feel of the target...
  # XXX: Use, but need to fix ambiguity.. Maybe we store this for each sentence
  def denormalize(self, word):
    if word in self.norm_score:
      norm_score = self.norm_score[word][0]
      norm_score = norm_score/(norm_score+self.norm_score[word][1])
      if random.random() > norm_score:
        return self.denorm_d[word]
      else:
        return word
    else:
      return word


# Maybe pos-tag isn't that important for searching. It seems
# to make some things harder... We probably should strip tenses from
# verbs, at least
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

#  - Subsitute @msgs: I/Me->You, Mine->Yours, My->Your
class PronounInverter:
  def __init__(self):
    self.pronoun_21_map = { "you":"i", "yours":"mine","your":"my" }
    self.pronoun_12_map = { "i":"you", "me":"you", "mine":"yours","my":"your" }

  def make_2nd_person(self, string):
    for i in self.pronoun_12_map.iterkeys():
      chars = list(i)
      casehack = ""
      for c in chars: casehack += "["+c.lower()+c.upper()+"]"
      string = re.sub("([^a-zA-Z]|^)"+casehack+"([^a-zA-Z]|$)",
                      "\\1"+self.pronoun_12_map[i]+"\\2", string)
    return string

  def make_1st_person(self, string):
    for i in self.pronoun_21_map.iterkeys():
      chars = list(i)
      casehack = ""
      for c in chars: casehack += "["+c.lower()+c.upper()+"]"
      string = re.sub("([^a-zA-Z]|^)"+casehack+"([^a-zA-Z]|$)",
                      "\\1"+self.pronoun_21_map[i]+"\\2", string)
    return string

  def invert_all(self, string):
    split_string = string.split()
    new_string = []
    for s in split_string:
      snew = self.make_1st_person(s)
      if snew == s:
        snew = self.make_2nd_person(s)
      new_string.append(snew)
    return " ".join(new_string)


# FIXME: Should these be static classes? Some maybe should
# do some record keeping on what they modify for inversion..
postrim = POSTrim()
porter = nltk.PorterStemmer()
twDict = Twictionary()
proverter = PronounInverter()

class TagBin:
  def __init__(self, text, tags):
    # XXX: Denormalize to get orig text?
    # XXX: Normalize tags?
    self.tags = tags
    self.orig_text = text
    self.tagged_tokens = [nltk.tag.util.tuple2str((porter.stem(t[0]).lower(),
                            postrim.trim(t[1]))) for t in self.tags]
    self.vocab = set(self.tagged_tokens)

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
def pos_tag(tokens):
  return nltk.pos_tag(tokens)

class NormalizedText(TagBin):
  skip_tokens = set(['"','(',')','[',']'])
  def __init__(self, text, hidden_text=""):
    if hidden_text:
      hidden_text = hidden_text.rstrip()
      if not hidden_text.endswith("."):
        hidden_text += ". "
      else:
        hidden_text += " "
    self.hidden_text = hidden_text
    # Hidden text will ruin our markov model. Need to avoid it for training
    tokens = []
    for w in self.scrub(nltk.word_tokenize(hidden_text+text)):
      if w: tokens.append(twDict.normalize(w))
    if not tokens: print "Empty text!"
    tags = pos_tag(tokens)
    TagBin.__init__(self, text, tags)

  # Clean up some brokenness of nltk.word_tokenize()
  # TODO: Maybe we should just make our own tokenizer... this is
  # getting a little absurd. Problem is the word_tokenize() is
  # pretty damn smart about contractions and the like
  def scrub(self, tokens):
    new_tokens = []

    i = 0
    while i < len(tokens):
      # TODO: Summarize URL contents and use them
      if tokens[i].startswith("http") and tokens[i+1] == ":" \
          and tokens[i+2].startswith("//"):
        new_tokens.append(tokens[i]+tokens[i+1]+tokens[i+2])
        i+=3
      elif tokens[i] in NormalizedText.skip_tokens:
        i+=1
      elif tokens[i] == "@" or tokens[i] == "#" or tokens[i] == "$":
        new_tokens.append(tokens[i]+tokens[i+1])
        i+=2
      elif tokens[i] == "*" and i+2 < len(tokens) and tokens[i+2] == "*":
        new_tokens.append(tokens[i]+tokens[i+1]+tokens[i+2])
        i+=3
      elif tokens[i] == "/" and i+2 < len(tokens) and tokens[i+2] == "/":
        new_tokens.append(tokens[i]+tokens[i+1]+tokens[i+2])
        i+=3
      elif tokens[i].endswith("."):
        new_tokens.append(tokens[i][:-1])
        new_tokens.append(".")
        i+=1
      elif tokens[i]:
        new_tokens.append(tokens[i])
        i+=1

    if new_tokens:
      #if len(new_tokens[0]) > 1 and not new_tokens[0].isupper():
      #  new_tokens[0] = new_tokens[0][0].lower()+new_tokens[0][1:]
      #else:
      #  new_tokens[0] = new_tokens[0][0].lower()

      if not curses.ascii.ispunct(new_tokens[-1][0]):
        new_tokens.append(".")

    return new_tokens

  def count(self, word):
    return self.tagged_tokens.count(word)


#- Notion of "pronoun context" per person
#  - Store list of recent nouns+verbs used in queries/responses
#    - If no specific noun/verb is found, use these for query
#  - Expire after a while
class ConversationContext:
  pass

#- Knowledge Base:
#  - Subsitute @msgs: I/Me->You, Mine->Yours, My->Your
#  - Tag with sender as hidden text, add to list of tweets for searching
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

# Search:
#  1. Tag, tolower(), stem
#  2. if strip: strip all but nouns, adj, verbs
#  3. tf-idf to score text record from collection
#  4. TODO: If score is 0, requery for "WTF buh uh huh dunno what talking about"
#  5. FIXME: Return probabilistically based on score with some cutoff
#  6. TODO: try to avoid replying to questions with questions. score lower.
class SearchableTextCollection:
  def __init__(self, texts, name=None):
    self._idf_cache = {}
    self.texts = texts
    self.needs_update = True
    self.update_matrix()

  def add_text(self, text, update=False):
    self.texts.append(text)
    self.needs_update = True
    if update: self.update_matrix()

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
    print "Building Qvector.."

    query_string = proverter.invert_all(query_string)
    print "Inverted Query: "+query_string
    query_text = NormalizedText(query_string)

    # XXX: Hrmm, could amplify nouns and dampen adjectives and verbs..
    # How though? Normalize to 0.75*(max_noun/max_word)
    # Insert a You/NN in all queries?
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
      return self.query(NormalizedText("WTF buh uh huh dunno what talking about understand"),
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

  def tf(self, term, text, method=None):
    """ The frequency of the term in text. """
    return float(text.count(term)) / len(text.tagged_tokens)

  def idf(self, term, method=None):
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


# Init:
#  1. Load corpus from disk into TextCollection
#  2. Tag corpus, tolower(), stem words, store into TextCollection, backlink to original
class SoulCorpus:
  def __init__(self, directory):
    tweet_texts = []
    for root, dirs, files in os.walk(directory):
      for f in files:
        # .jtwt: json-encoded twitter tweets, 1 per line
        # TODO: Add @msgs to this user as hidden text
        if f.endswith(".jtwt"):
          fl = open(root+"/"+f, "r")
          for jtweet in fl.readlines():
            tweet = json.loads(jtweet)
            txt = tweet['text'].encode('ascii', 'ignore')
            if txt.startswith('RT') or txt.startswith('rt'): continue
            if txt[0] == '@': txt = re.sub('^@[\S]+ ', '', txt)
            norm_text = NormalizedText(txt)
            if norm_text.tags: tweet_texts.append(norm_text)
            print "Loaded tweet #"+str(len(tweet_texts)) #+"/"+str(len(files))
        # .twt: plain-text tweets, 1 per line
        elif f.endswith(".twt"):
          fl = open(root+"/"+f, "r")
          for tweet in fl.readlines():
            txt = tweet.encode('ascii', 'ignore')
            if txt.startswith('RT'): continue
            if txt[0] == '@': txt = re.sub('^@[\S]+ ', '', txt)
            norm_text = NormalizedText(txt)
            if norm_text.tags: tweet_texts.append(norm_text)
            print "Loaded tweet #"+str(len(tweet_texts)) #+"/"+str(len(files))
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

    self.tweet_collection = SearchableTextCollection(tweet_texts)
    #self.tweet_collection.generate(100)


# Mutate:
#  1. For each noun, adj and verb not in query:
#     random choice:
#       A. nltk.text.Text.similar(w) from corups
#          Replace with tf-idf weighted word
#       C. tf-idf score wordnet synonym set, choose based on that
#       B. tf-idf score: http://labs.google.com/sets
class PhraseMutator:
  pass

# Generate:
#  1. HMM
#  2. HMM-Trigram
# Intuition: x-i-x-x is going to work better becuase hmm predicts
# current based on previous state already
#  3. HMM-4gram (x-x-i-x or x-i-x-x)
#  4. HMM-5gram (x-x-i-x-x, x-i-x-x-x, x-x-i-x-x, x-x-x-i-x)

# XXX: Move to soul..
# TODO: Could we classify output of this as funny vs not funny somehow?
# Maybe a naive-bayes
class PhraseGenerator:
  def __init__(self, soul):
    #self.hmm = self.build_hmm(soul)
    #self.tri_hmm = self.build_tri_hmm(soul)
    # 41 is the winner so far..
    self.nm_hmm = {}
    self.nm_hmm["41"] = self.build_nm_hmm(soul, 4, 1)

  def build_tri_hmm(self, soul):
    print "Prepping Tri-HMM"
    hmm_states = set()
    hmm_symbols = set()
    hmm_sequences = list()

    for text in soul.tweet_collection.texts:
      if not text.hidden_text and len(text.tags) > 2:
        hmm_symbols.update([t[0] for t in text.tags])
        sequence = []
        tag = "^"+text.tags[0][1]+text.tags[1][1]
        hmm_states.update(tag)
        sequence.append((text.tags[0][0], tag))
        for i in xrange(1,len(text.tags)-1):
          tag = text.tags[i-1][1] + text.tags[i][1] + text.tags[i+1][1]
          hmm_states.update(tag)
          sequence.append((text.tags[i][0], tag))
        tag = text.tags[-2][1]+text.tags[-1][1]+"|"
        hmm_states.update(tag)
        sequence.append((text.tags[-1][0], tag))
        hmm_sequences.append(sequence)

    print "Training Tri-HMM with "+str(len(hmm_states))+" states, "+\
            str(len(hmm_symbols))+" syms and "+\
            str(len(hmm_sequences))+" sequences"
    hmm_trainer = nltk.tag.hmm.HiddenMarkovModelTrainer(list(hmm_states),
                                                     list(hmm_symbols))
    tri_hmm = hmm_trainer.train_supervised(hmm_sequences)
    print "Trained Tri-HMM"
    return tri_hmm

  def build_nm_hmm(self, soul, l, o):
    print "Prepping NM-HMM"
    hmm_states = set()
    hmm_symbols = set()
    hmm_sequences = list()

    # XXX: Something weird is happening with case here...
    for text in soul.tweet_collection.texts:
      if not text.hidden_text and not "http://" in text.orig_text:
        #and len(text.tags) > l-1:
        hmm_symbols.update([t[0] for t in text.tags])
        sequence = []

        tag = ("^ "*(o))
        for t in xrange(o, min(l,len(text.tags))):
          tag += text.tags[t][1]+" "
        hmm_states.update(tag)
        #print "Adding initial Tag: "+str(tag)
        sequence.append((text.tags[0][0], tag))

        for i in xrange(1,len(text.tags)-1):
          tag = ""
          for t in xrange(i-o, min(i+(l-o),len(text.tags)-1)):
            tag += text.tags[t][1]+" "

          hmm_states.update(tag)
          sequence.append((text.tags[i][0], tag))

        tag = ""
        for t in xrange(-min(o,len(text.tags)),0):
          tag+=text.tags[t][1]+" "
        tag += ("|"*(l-o))
        hmm_states.update(tag)
        sequence.append((text.tags[-1][0], tag))

        #print "Adding Final Tag: "+str(tag)
        hmm_sequences.append(sequence)

    print "Training NM-HMM with "+str(len(hmm_states))+" states, "+\
            str(len(hmm_symbols))+" syms and "+\
            str(len(hmm_sequences))+" sequences"
    hmm_trainer = nltk.tag.hmm.HiddenMarkovModelTrainer(list(hmm_states),
                                                     list(hmm_symbols))
    nm_hmm = hmm_trainer.train_supervised(hmm_sequences)
    nm_hmm.length = l
    nm_hmm.offset = o
    print "Trained NM-HMM"
    return nm_hmm

  def build_hmm(self, soul):
    print "Prepping HMM"
    hmm_states = set()
    hmm_symbols = set()
    hmm_sequences = list()

    for text in soul.tweet_collection.texts:
      if not text.hidden_text:
        hmm_symbols.update([t[0] for t in text.tags])
        hmm_states.update([t[1] for t in text.tags])
        hmm_sequences.append(text.tags)

    print "Training HMM"
    hmm_trainer = nltk.tag.hmm.HiddenMarkovModelTrainer(list(hmm_states),
                                                     list(hmm_symbols))
    hmm = hmm_trainer.train_supervised(hmm_sequences)
    print "Trained HMM"
    return hmm

  def hmm_phrase(self):
    return self.hmm.random_sample(random, 500)

  def tri_hmm_phrase(self):
    while True:
      try:
        return self.tri_hmm.random_sample(random, 500, "|")
      except:
        print "Probability failure..."

  def nm_hmm_phrase(self, mode="41"):
    # TODO: Maybe build mode on-demand?
    while True:
      try:
        return self.nm_hmm[mode].random_sample(random, 500,
                "|"*(self.nm_hmm[mode].length-self.nm_hmm[mode].offset))
      except:
        print "Probability failure..."
        traceback.print_exc()

  # XXX: move elsewhere?
  def stringify_tags(self, tags):
    string = ""
    for t in tags:
      if t[0] and curses.ascii.ispunct(t[0][0]) \
        and t[0][0] != "#" and t[0][0] != "@":
        string += t[0]
      else:
        string += " "+t[0]
    return string

  def test_run(self):
    for mode in map(str, [30, 31, 32, 40, 41, 42, 43]):
      print "Writing "+mode
      f = open(mode+".twt", "w")
      for i in xrange(1,50):
        print "  Writing "+str(i)
        result = self.nm_hmm_phrase(mode)
        f.write(self.stringify_tags(result)+"\n\n")

def main():
  #soul = SoulCorpus('target_user')
  #pickle.dump(soul, open("target_user.soul", "w"))
  soul = pickle.load(open("target_user.soul", "r"))
  pg = PhraseGenerator(soul)

  #pg.test_run()
  #return

  while True:
    query = raw_input("> ")
    if not query: query = "41"
    if query.isdigit():
      str_result = "1"*256
      while len(str_result) > 140:
        result = pg.nm_hmm_phrase(query)
        str_result = pg.stringify_tags(result)
      print str(result)
      print str_result
    else:
      result = soul.tweet_collection.query(query, randomize=True)
      print "|"+result.orig_text+"|"

  #soul.tweet_collection.generate(100)

if __name__ == "__main__":
  main()
