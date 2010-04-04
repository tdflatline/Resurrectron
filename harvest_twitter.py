#!/usr/bin/python

# Corpus harvester: Used for gathering promising corpuses to resurrect.
#
# Uses: python-twitter (http://github.com/tweetr/python-twitter) 
#

from libs import twitterapi
import sys

def main():
    api = twitterapi.Api()

    if sys.argv[1] is not None:
        username = sys.argv[1]
    else:
        print "Usage: ./harvest_twitter <username>"
        sys.exit()

    statuses = api.GetUserTimeline(user=username, count=3200)

    fp = open('%s.jtwt' % username,'w')
    for s in statuses:
        fp.write(s.AsJsonString())
        fp.write("\n")
    fp.close()

if __name__ == "__main__":
    main()
