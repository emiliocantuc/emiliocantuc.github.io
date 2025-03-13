import os, json, random
import numpy as np
import pandas as pd
from itertools import product

from google import genai
from google.genai import types

from utils import get_embedding, stringify_event

EMB_DIM = 1536

def init_usr_emb(size = EMB_DIM):
    emb = np.random.normal(size = size)
    return emb / np.linalg.norm(emb)

class User:
    def __init__(self, usr_id, init_usr_emb_fn = init_usr_emb):
        self.usr_id = usr_id
        self.usr_emb = init_usr_emb_fn()
        self.interests = {}
    
    def add_interests(self, interests, interest_embs = None):
        assert isinstance(interests, list) and isinstance(interests[0], str)
        if interest_embs is None: interest_emb = get_embedding(interests)
        for i, e in zip(interests, interest_emb): self.interests[i] = e


class Recommender:

    def __init__(self, events, event_embs, vote_EMA_alpha, interest_weight, mmr_relevance_lambda, dedup_threshold = 0.99):
        self.events = events
        self.event_embs = event_embs
        self.vote_EMA_alpha = vote_EMA_alpha
        self.interest_weight = interest_weight
        self.mmr_relevance_lambda = mmr_relevance_lambda
        self.dedup_threshold = dedup_threshold

    def recommend(self, user, n):
        # Calculate similarities
        event_event_sims = self.event_embs @ self.event_embs.T
        usr_event_sims = (user.usr_emb[None, :] @ self.event_embs.T).flatten()

        # For every event, the distance to the closest user interest
        interest_event_max_sims = np.zeros_like(usr_event_sims)
        if user.interests:
            interests_embs = np.vstack(list(user.interests.values()))
            interest_event_max_sims = np.max(interests_embs @ self.event_embs.T, axis = 0)
        
        # print(interest_event_max_sims.shape)

        # How good relevant events are (user behavior + interests, higher better)
        event_scores = (1 - self.interest_weight) * usr_event_sims + self.interest_weight * interest_event_max_sims
        # print(event_scores.shape)

        # Deduplicate
        n_events = usr_event_sims.shape[0]
        for i in range(n_events):
            for j in range(i + 1, n_events):
                if event_event_sims[i, j] > self.dedup_threshold:
                    # We remove the later event (j > i)
                    event_scores[j] = -np.inf
                    # print('removed ',j, 'too similar to ',i)

        # MMR
        selected = [event_scores.argmax()]
        for _ in range(n - 1):
            mmr_scores = {}
            
            for e_ix in range(n_events):
                if e_ix in selected: continue
                lamb = self.mmr_relevance_lambda
                relevance = event_scores[e_ix]
                max_sim_to_selected = max([event_event_sims[e_ix, s] for s in selected])
                mmr_score = lamb * relevance + (1 - lamb) * -max_sim_to_selected
                mmr_scores[e_ix] = mmr_score

            selected.append(max(mmr_scores, key = mmr_scores.get))

        # print(selected)
        return [self.events[i] for i in selected]


    def record_feedback(self, usr, event_id, rating):
        # Update usr_emb with weighted EMA
        w = {'up': 1., 'down': -1., 'cal': 2.}[rating]
        event_emb = self.event_embs[event_id]
        alpha = self.vote_EMA_alpha
        usr.usr_emb = alpha * usr.usr_emb + (1 - alpha) * w * event_emb
        usr.usr_emb /= np.linalg.norm(usr.usr_emb)


system_prompt = lambda persona_str, n_weeks: f'''
Asume the following persona and use it to answer all of the following questions.

Persona:
{persona_str}

You will help evaluate a recommender system for events. You will be given a list of {n_weeks} events each "week" and asked to rate them.
'''

def weekly_prompt(recommended_events):
  formated_events = '\n'.join([f"---\nEvent {i}:\n" + stringify_event(e, 200) for i,e in enumerate(recommended_events)])
  return f'''Here are the events for this week:

  {formated_events}

  For each event, provide a "vote":
  - "up" if you are interested in the event and would consider attending
  - "cal" if you would actively try to add it to your calendar and attend
  - "down" if you are not interested and would not attend
  - "neutral" if you are indifferent and have no strong opinion

  Additionally, rate these weekly recommendations as a whole on a scale of 1-10, where:
  - 1 = "Completely dissatisfied; none of these events appeal to me"
  - 5 = "Neutral; some events are okay, but the list isn't exciting"
  - 10 = "Extremely satisfied; most or all events are highly appealing"

  Consider factors like how well the events match your (persona's) interests, the variety, and overall excitement.

  Return your ratings in JSON format, for example:
  {{
    "event_ratings": [
      {{"event_id": 0, "vote": "down/neutral/up/cal"}},
      {{"event_id": 1, "vote": "down/neutral/up/cal"}},
      ...
    ],
    "weekly_satisfaction": ?
  }}
  '''

def eval_recomender(persona_json, persona_str, vote_EMA_alpha, interest_weight, mmr_relevance_lambda, dedup_threshold, n_recommendations):

    n_weeks = 11 # save a little on requests

    # Form user
    usr = User(0)
    assert 'interests' in persona_json, 'No interests in persona: ' + persona_json
    usr.add_interests(persona_json['interests'])

    evaluations = []

    # Simulate multiple weeks
    for i, week_str in enumerate(sorted(os.listdir('data/semester_data'))[:n_weeks]):

        # Load week data
        with open(f'data/semester_data/{week_str}/events.json', 'r') as f: week_events = json.load(f)
        week_embs = np.load(f'data/semester_data/{week_str}/embs.npy')

        # Get recommendations for that week
        print(f'\tweek: {i}, # events: {len(week_events)}')
        r = Recommender(week_events, week_embs, vote_EMA_alpha, interest_weight, mmr_relevance_lambda, dedup_threshold)
        recommended_events = r.recommend(usr, n_recommendations)

        # Ask Gemini to evaluate
        client = genai.Client(api_key = os.environ.get('GEMINI_API_KEY'))
        model = 'gemini-2.0-flash-lite'
        contents = [types.Content(role = 'user', parts = [types.Part.from_text(text = weekly_prompt(recommended_events))])]
        gen_config = types.GenerateContentConfig(
            temperature = 1,
            top_p = 0.95,
            top_k = 40,
            max_output_tokens = 8192,
            response_mime_type = 'application/json',
            system_instruction = [types.Part.from_text(text = system_prompt(persona_str, n_recommendations))],
        )
        response = client.models.generate_content(model = model, contents = contents, config = gen_config)
        evaluation = json.loads(response.text)

        # Record feedback
        for rating in evaluation['event_ratings']:
            if rating['vote'] in ['up', 'down', 'cal']:
                r.record_feedback(usr, rating['event_id'], rating['vote'])
                
            elif rating['vote'] != 'neutral': print('Invalid vote:', rating)

        evaluation['week'] = i
        evaluations.append(evaluation)
    
    return evaluations


if __name__ == '__main__':

    import dotenv; dotenv.load_dotenv()

    n_recommendations = 30
    dedup_threshold = 0.99
    n_personas = 10

    with open('data/personas.json', 'r') as f: personas = json.load(f)
    # random.shuffle(personas)

    df = []

    for i_p, persona_json in enumerate(personas[:n_personas]):
        persona_str = '\n'.join([f'- {k}: {v}' for k, v in persona_json.items()])
        print(f'Persona {i_p+1}/{n_personas}:\n{persona_str}')    

        for i, (vote_EMA_alpha, interest_weight, mmr_relevance_lambda) in enumerate(product(
            [0.75, 0.8, 0.85, 0.9, 0.95, 0.99],
            np.linspace(0.1, 0.9, 5),
            np.linspace(0.1, 0.9, 5)
            )):

            print(f'{i} / {6*5*5}', vote_EMA_alpha, interest_weight, mmr_relevance_lambda)
            evaluations = eval_recomender(
                persona_json, persona_str, vote_EMA_alpha, interest_weight,
                mmr_relevance_lambda, dedup_threshold, n_recommendations
            )
            for evaluation in evaluations:
                df.append({
                    'week': evaluation['week'],
                    'persona_id': i_p,
                    'persona': persona_json,
                    'vote_EMA_alpha': vote_EMA_alpha,
                    'interest_weight': interest_weight,
                    'mmr_relevance_lambda': mmr_relevance_lambda,
                    'weekly_satisfaction': evaluation['weekly_satisfaction'],
                    **{d['event_id']:d['vote'] for d in evaluation['event_ratings']}
                })

        df = pd.DataFrame(df)
        df.to_csv('data/evaluations.csv', index=False)