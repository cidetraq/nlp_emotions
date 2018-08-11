import pandas as pd

data=pd.read_csv('fb_news_posts_20k.csv')
data.head()

data['total_reactions']=data['react_angry']+data['react_haha']+data['react_like']+data['react_love']+data['react_sad']+data['react_wow']
subset=data.iloc[0:10]

import operator

def keyword_totals():
    keywords=dict()
    for index, row in subset.iterrows():
        message=row['message']
        words=message.split(' ')
        for word in words:
            word=word.strip('/./()!/@/#//-:')
            word=word.replace("'", "")
            word=word.replace('"', '')
            word=word.replace('()', '')
            if word not in keywords:
                keywords[word]=[row['total_reactions'], row['shares'], 1]
            else:
                keywords[word][0]+=row['total_reactions']
                keywords[word][1]+=row['shares']
                keywords[word][2]+=1
    keywords_by_reactions=sorted(list(keywords.items()), key=lambda x: x[1][0], reverse=True)
    keywords_by_shares=sorted(list(keywords.items()), key=lambda x: x[1][1], reverse=True)
    keywords_by_frequency=sorted(list(keywords.items()), key=lambda x: x[1][2], reverse=True)
    return [keywords_by_reactions, keywords_by_shares, keywords_by_frequency]
new_keywords=keyword_totals()

def avg_keyword():
    for keyword in new_keywords[0]:
        avg_reactions=keyword[1][0]/keyword[1][2]
        keyword=list(keyword)
        keyword[1].append(avg_reactions)
    return new_keywords

new_keywords=avg_keyword()
keywords_by_avg_reactions=sorted(new_keywords[0], key=lambda x: x[1][3], reverse=True)


