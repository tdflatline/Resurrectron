#!/usr/bin/python

import cPickle as pickle
#import resurrect
from resurrect import *
#from resurrect import BrainReader,TwitterBrain
from extract import CorpusSoul
import traceback
import sys
import gzip

# XXX: Make this more user friendly.

def main():
  brain = None
  soul = None
  soul_file = sys.argv[1]
  brain_file = sys.argv[2]

  try:
    print "Loading soul file..."
    soul = pickle.load(open(soul_file, "r"))
    print "Loaded soul file."
  except pickle.UnpicklingError:
    soul = pickle.load(gzip.GzipFile(soul_file, "r"))
    print "Loaded soul file."
  except KeyError:
    soul = pickle.load(gzip.GzipFile(soul_file, "r"))
    print "Loaded soul file."
  except IOError:
    print "No soul file found..."
    sys.exit(0)
  except Exception,e:
    traceback.print_exc()

  soul.normalizer.verify_scores()

  try:
    brain = BrainReader.load(brain_file)
    brain.restart(soul)
  except IOError:
    print "No brain file found..."
    sys.exit(0)
  except Exception,e:
    traceback.print_exc()

  for i in brain.pending_tweets.texts: print i.text
  sys.exit(0)

if __name__ == "__main__":
  main()
