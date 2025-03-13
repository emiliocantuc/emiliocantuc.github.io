import os, requests, json, time, base64
from datetime import datetime

from google import genai
from google.genai import types

from dotenv import load_dotenv
load_dotenv()

day_of_week = lambda datetime_str: datetime.strptime(datetime_str, '%Y%m%dT%H%M%S').strftime('%A')
gget = lambda e, k, v: e.get(k, None) if e.get(k, None) is not None else v

def stringify_event(e, desc_len=800):
    limit = lambda s, n: s if len(s) < n else s[:n-3] + '...'
    sponsors = [gget(s, 'group_name', '') for s in gget(e, 'sponsors', {})]
    maize_group = e.get('maizepages_import', {}).get('maizepages_group_name')
    if maize_group: sponsors.append(maize_group)
    o = [
        f"{e['event_type']}:{limit(gget(e, 'combined_title', ''), 200)}",
        f"{limit(gget(e, 'description', ''), desc_len)}",
        f"Where:{e['location_name']}",
        f"When:{day_of_week(e['datetime_start'])} {e['time_start'].split(':')[0]}",
    ]
    if sponsors: o.append(f"Sponsors:{', '.join(sponsors)}")
    return '\n'.join(o)

def get_embedding(l, key = None, model = 'text-embedding-3-small', dimensions = None):
    """
    Calls OpenAPI's embedding API to embed string `s`.
    l: list of strings to embed
    key: OpenAPI key
    model: OpenAPI model to use
    """
    if key is None: key = os.environ['OAI_KEY']
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + key,
    }

    json_data = {
        'input': l,
        'model': model,
    }
    if dimensions is not None: json_data['dimensions'] = dimensions

    response = requests.post('https://api.openai.com/v1/embeddings', headers=headers, json=json_data)

    if response.status_code != 200 or 'embedding' not in response.text:
        raise Exception(f'Error getting embedding: {response.text}')
    
    return [i['embedding'] for i in response.json()['data']]