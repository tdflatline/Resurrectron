#!/usr/bin/python

# XXX: This is not terribly user friendly.

import nltk
import sys
from libs import Arc90

def main():
  url = sys.argv[1]
  num_pages = int(sys.argv[2])
  posts_per_page = int(sys.argv[3])

  for i in xrange(num_pages):
    posts = Arc90.fetchLink(url+str(i), posts_per_page)
    for p in xrange(len(posts)):
      text_string = nltk.util.clean_html(posts[p])
      f = file("./target_user/"+str(i)+"-"+str(p)+".post", "w")
      f.write(text_string)

if __name__ == "__main__":
  main()

