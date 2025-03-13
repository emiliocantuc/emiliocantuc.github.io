import requests, time, random, json, os
from bs4 import BeautifulSoup
import urllib.parse

import numpy as np

from utils import get_embedding, stringify_event

def get_week_event_urls(n_weeks, baseURL = 'https://events.umich.edu/week/'):
    # Gets urls of weekly endpoints starting at the current week and going forward n_weeks

    urls = []
    currentURL = baseURL
    r = requests.get(currentURL)
    b = BeautifulSoup(r.text, 'html.parser')

    while n_weeks > 0:

        n_weeks -= 1

        # Find link to '/week/...' with 'Prev' in text
        week_links = b.find_all('a', href = lambda x: x and '/week/' in x)
        next_link = None
        for link in week_links:
            if link and 'Next' in link.text:
                next_link = link.get('href')
                break
        if next_link is None: break

        currentURL = urllib.parse.urljoin(baseURL, next_link)
        r = requests.get(currentURL)
        b = BeautifulSoup(r.text, 'html.parser')
        urls.append(currentURL)
    
    return urls

if __name__ == '__main__':

    start_week = '2024-09-01' # First week of the (fall 2024) semester 

    urls = get_week_event_urls(4*4, baseURL = f'https://events.umich.edu/week/{start_week}') # About 4 months
    print(urls)

    # Scrap events for the urls we got
    for i, url in enumerate(urls):

        out_dir = f'data/semester_data/{url.split("/")[-1]}'
        os.makedirs(out_dir, exist_ok = True)
        r = requests.get(url + '/json?v=2')
        events = r.json()

        # Save
        with open(f'{out_dir}/events.json', 'w+') as f:
            json.dump(events, f, indent = 2)
        
        # Embed
        to_embed = [stringify_event(e) for e in events]
        embeddings = get_embedding(to_embed, os.environ['OAI_KEY'])
        
        if i % 10 == 0: time.sleep(random.random() * 5)

        E = np.array([np.array(e) for e in embeddings])
        np.save(f'{out_dir}/embs.npy', E)

        print(url, f'got {len(events)} events')
        print(f'Got embeddings shaped: {E.shape}')
  
    print('Done!')
