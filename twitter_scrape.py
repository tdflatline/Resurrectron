#!/usr/bin/python

# Uses: http://pypi.python.org/pypi/twitter/1.2.1
#
# This script should be rewritten, its pretty fucking clunky.
# If we do that, we should use the python-twitter 0.6 api

import re, sys, string, time
import twitter.api
from twitter import TwitterError
from urllib2 import HTTPError, URLError
import simplejson
import json
import traceback

def save_tweet(tweet):
    global fp
    j = simplejson.dumps(tweet)
    fp.write(j)
    fp.write('\n')

def get_rate_limit():
    while(1):
        try:
            rate = api.account.rate_limit_status()
            rate = rate['remaining_hits']
            return rate
        except:
            traceback.print_exc()


def get_new_account():
    global api

    if len(account_list) > 0:
        account = account_list.pop()
    else:
        print "Ran out of accounts, restart next hour"
        sys.exit(1)

    password = 'lolwut'
    api = twitter.api.Twitter(account,password)

    print "Switched account to %s" %account
    return

def check_rate_limit():
    while True:
        rate = get_rate_limit()
        if rate < 2:
            print "Fetching new account..."
            get_new_account()
        else:
            break

    return

def fetch_tweets(user):
    from_user = user
    curr_page = 0

    print "Fetching tweets for %s" %(from_user)
    while(1):
        #check_rate_limit()
        print "  Page %s" %curr_page
        try:
            tweets = api.statuses.user_timeline(screen_name=from_user,count=200,page=curr_page)
        except TwitterError, e:
            print "Failed for user %s, (%s)" %(from_user,e)
            mark_user_checked(from_user)
            break
        except:
            print "Failed to get page.."
            continue

        if len(tweets) == 0:
            break
        else:                
            map(save_tweet,tweets)
            curr_page += 1
    return

def get_api_account_list():
    global account_list
    query = "select screen_name from twitter_spam_traps"
    db_cursor.execute(query)

    twitter_accounts = db_cursor.fetchall()

    for account in twitter_accounts:
        account_list.append(account['screen_name'])

def main():
    global api
    global fp
    api = twitter.api.Twitter()

    account_list = []

    fp = open('target_user.jtwt','w')

    #get_api_account_list()

    fetch_tweets('target_user')

if __name__ == "__main__":
    main()



