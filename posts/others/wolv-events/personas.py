import base64
import os, json, time
from google import genai
from google.genai import types
from dotenv import load_dotenv
import pandas as pd

if __name__ == "__main__":
    load_dotenv()

    TOTAL_PERSONAS = 10_000

    N_PER_QUERY = 10 # 8k / 500 = 16, but being safe
    N_QUERIES = TOTAL_PERSONAS // N_PER_QUERY
    N_PAST_RESPONSES = int(1e6 / (500 * N_PER_QUERY)) # 1M max context length and 500ish tokens per response
    N_PAST_RESPONSES //= 2 # Try not to saturate the context too much


    prompt = f"""Generate {N_PER_QUERY} diverse college student personas, each with a detailed backstory representative of the U.S. college population. Ensure variety in age, gender, ethnicity, socioeconomic background, major, and institution type (e.g., community college, public university, private college). For each persona, include:
- Basic demographics (name, age, gender, ethnicity, hometown).
- Academic details (level (undergrad/graduate), year, major, type of school).
- A 3-5 sentence backstory covering their motivations, challenges, and a unique personal trait or hobby.
- Three specific interests (e.g., genres of media, activities, academic topics). Tie these to their backstory but add some diversity (people are multifaceted).
- A short-term goal tied to their college experience.

Make the personas distinct yet plausible, reflecting common student experiences like financial pressures, cultural adjustments, or career exploration, with interests that are specific enough to generate tailored recommendations.

Return the personas in JSON format. Do not generate the same or very similar personas multiple times.
"""

    personas = []
    past_responses = []
    
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    model = "gemini-2.0-flash-lite"  # As of March 10, 2025

    i = 0
    while i < N_QUERIES:

        if i % 10 == 0:
            time.sleep(30)  # Avoid rate limits
            with open("personas.json", "w+") as f: json.dump(personas, f, indent=4)

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            ),
        ]
        
        for resp in past_responses[-N_PAST_RESPONSES:]:
            contents.append(types.Content(
                role="model",
                parts=[types.Part.from_text(text=resp)],
            ))
        
        contents.append(types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=f"Generate the next {N_PER_QUERY} personas."),
            ],
        ))

        generate_content_config = types.GenerateContentConfig(
            temperature=1.5,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
            response_mime_type="application/json",
            seed=i,
        )

        try:

            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=generate_content_config,
            )
        
            ps = json.loads(response.text)
            personas.extend(ps)
            past_responses.append(response.text)
            
            print(f"Generated batch {i+1} / {N_QUERIES}:", len(ps))
            i += 1
        
        except Exception as e:
            print(e)
            time.sleep(60*10)
        

    # Save as JSON
    with open("personas.json", "w+") as f:
        json.dump(personas, f, indent=4)