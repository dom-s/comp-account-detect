def get_user_info(tweets):
    user_id = None
    comp_id = None
    comp_begin = None
    comp_end = None
    comp_begin_day = None
    comp_end_day = None

    last_date = None

    day = 0
    for tweet in tweets:
        tweet = tweet.strip().split('\t')
        user = tweet[0]
        comp = int(tweet[1]) if tweet[1] != 'None' else None
        date = tweet[2]
        day += 1
        if user_id is None:
            user_id = user
        if comp is not None and comp_id is None:
            comp_id = comp
            comp_begin = last_date
            comp_begin_day = day
        if comp is None and comp_id is not None and comp_end is None:
            comp_end = date
            comp_end_day = day

        last_date = date

    return user_id, comp_id, comp_begin, comp_end, comp_begin_day, comp_end_day