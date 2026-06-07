from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ── Model ──────────────────────────────────────────────────────────────────────
model = ChatOpenAI(model="gpt-4.1", temperature=0)

# ── Prompt Template ────────────────────────────────────────────────────────────
prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a professional Movie Information Extraction Assistant.

Your task:
Extract useful structured information from a movie paragraph and present it in the exact format below.

Rules:
- Do NOT add explanations
- Do NOT add extra commentary
- Follow the exact format
- If information is missing → write NULL
- Keep summary short (2-3 lines max)
- Do NOT guess unknown facts

Output Format:

Movie Title      : 
Release Year     : 
Genre            : 
Director         : 
Main Cast        : 
Setting/Location : 
Plot             : 
Themes           : 
Ratings          : 
Notable Features : 
Short Summary    : """
    ),
    (
        "human",
        "Extract all useful movie information from the following paragraph:\n\n{paragraph}"
    )
])

# ── Chain ──────────────────────────────────────────────────────────────────────
chain = prompt_template | model

# ── Example paragraph ─────────────────────────────────────────────────────────
paragraph = """
Interstellar is a visually stunning science fiction epic directed by Christopher Nolan.
Released in 2014, the film stars Matthew McConaughey, Anne Hathaway, Jessica Chastain,
and Michael Caine. The story revolves around a group of astronauts who travel through a
wormhole near Saturn in search of a new home for humanity as Earth faces environmental
collapse. The movie was widely appreciated for its emotional depth, scientific accuracy,
and Hans Zimmer's powerful soundtrack. It holds a rating of 8.6 on IMDb and is often
considered one of the greatest sci-fi films of the 21st century.
"""

# ── Run ────────────────────────────────────────────────────────────────────────
response = chain.invoke({"paragraph": paragraph})

print(response.content)
