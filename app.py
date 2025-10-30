from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi import BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
from supabase import create_client
from google import genai
import os
import json
import re
import uvicorn
import asyncio

# ------------------ SETUP ------------------
load_dotenv()

# Supabase setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Gemini setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if genai and GEMINI_API_KEY else None

# Gemini usage log
LOG_FILE = "gemini_usage_log.jsonl"
 
# FastAPI setup
app = FastAPI(title="Rohan - AI Companion")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ MODELS ------------------
class SignupRequest(BaseModel):
    username: str
    email: str
    password: str
    nickname: str
    age: int
    designation: str
    location: str
    likes: List[str] = Field(default_factory=list)

class SigninRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    username: str
    message: Optional[str] = ""

# ------------------ UTILITIES ------------------
def generate(prompt, model="gemini-2.5-flash"):
    if not client:
        return "(Gemini not configured)"
    try:
        res = client.models.generate_content(model=model, contents=prompt)
        # Log usage
        usage = res.usage_metadata
        prompt_tokens = usage.prompt_token_count
        response_tokens = usage.candidates_token_count
        total_tokens = usage.total_token_count
        log_entry = {
            "response": res.text.strip(),
            "model": model,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "total_tokens": total_tokens
        }
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        return res.text.strip()
    except Exception as e:
        return f"(Error: {str(e)})"
    

def clean_json_response(raw_text: str):
    cleaned = re.sub(r"^(`{3,}|'{3,})\s*json\s*", "", raw_text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"(`{3,}|'{3,})$", "", cleaned.strip())
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    return cleaned

# ------------------ DATABASE HELPERS ------------------
def get_user(username: str):
    res = supabase.table("auth_users").select("*").eq("username", username).execute()
    if not res.data:
        return None
    return res.data[0]

def get_profile(username: str):
    res = supabase.table("user_profiles").select("*").eq("username", username).execute()
    if not res.data:
        return None
    return res.data[0]

def get_rohan_profile():
    res = supabase.table("user_profiles").select("*").eq("username", "Rohan").execute()
    if not res.data:
        return None     
    return res.data[0]

def get_recent_history(username: str, limit: int = 10):
    res = supabase.table("conversation_history").select("*") \
        .or_(f"sender.eq.{username},receiver.eq.{username}") \
        .order("timestamp", desc=True).limit(limit).execute()
    recent_conversations = list(reversed(res.data))
    return recent_conversations if res.data else []

def add_conversation(sender, receiver, message, category):
    supabase.table("conversation_history").insert({
        "sender": sender,
        "receiver": receiver,
        "message": message,
        "category": category
    }).execute()

async def enrich_profile(username: str):
    """Auto-extract likes/dislikes/major_evenst/minor_events/people from recent chats."""
    # convos = get_recent_history(username, limit=15)
    res = supabase.table("conversation_history").select("*") \
        .eq("sender", username) \
        .order("timestamp", desc=True).limit(15).execute()
    convos = list(reversed(res.data))
    profile = get_profile(username)
    text = "\n".join([c["message"] for c in convos if "message" in c])
    if not text:
        return

    extract_prompt = f"""You are an AI that analyzes conversations between a user and their AI companion to extract important personal information.

    Task: Analyze the conversation log and extract a structured summary of key personal details.

    What to Extract:

    1. likes: Things the user enjoys or loves
    - Hobbies and activities (e.g., "painting", "hiking")
    - Foods and restaurants (e.g., "biryani", "Truffles")
    - Media (e.g., "Inception", "The Beatles", "Stranger Things")
    - Brands, places, or anything they express positive sentiment about

    2. dislikes: Things the user does not enjoy or loves
    - Hobbies and activities (e.g., "painting", "hiking")
    - Foods and restaurants (e.g., "biryani", "Truffles")
    - Media (e.g., "Inception", "The Beatles", "Stranger Things")
    - Brands, places, or anything they express negative sentiment about

    3. major_ events: Significant life events or milestones mentioned
    - Career changes (e.g., "started new job at Google")
    - Life transitions (e.g., "moved to Bangalore", "graduated college")
    - Important occasions (e.g., "sister's wedding", "promotion")
    - Only include specific events, not general statements

    4. minor_events: Smaller but notable events or experiences
    - Daily life events (e.g., "went to a concert", "tried a new restaurant")
    - Travel experiences (e.g., "visited Goa", "road trip to Mysore")
    - Social activities (e.g., "hung out with friends", "family dinner")
    - Only include specific events, not general statements

    5. people: Names and relationships of people in their life
    - Format: "Name (relationship)" (e.g., "Priya (sister)", "Alex (colleague)")
    - Include family, friends, colleagues, partners
    - Only include if both name AND relationship are mentioned

    Extraction Rules:
    - Only extract information explicitly stated in the conversation
    - Don't infer or assume information not directly mentioned
    - Use exact quotes or paraphrasing from the conversation
    - If a category has no clear information, use an empty array []
    - Maintain consistent formatting across all entries
    - Remove duplicates

    Output Format:
    Return ONLY valid JSON with no additional text, explanations, or markdown:

    {{
    "likes": ["item1", "item2", "item3"],
    "dislikes": ["item1", "item2", "item3"],
    "major_events": ["event description 1", "event description 2"],
    "minor_events": ["event description 1", "event description 2"],
    "people": ["Name (relationship)", "Name (relationship)"]
    }}

    Conversation Log:
    {text}

    Extract and return JSON
    """

    result = generate(extract_prompt, "gemini-2.5-pro")
    cleaned = clean_json_response(result)
    # print(f"Enrichment raw result: {cleaned}")  # Debug log
    try:
        extracted = json.loads(cleaned)
        profile = get_profile(username)
        if not profile:
            return

        # Merge new items
        for field in ["likes", "dislikes", "major_events", "minor_events", "people"]:
            old_values = profile.get(field, [])
            new_values = extracted.get(field, [])
            merged = list(set(old_values + new_values))
            profile[field] = merged

        supabase.table("user_profiles").update(profile).eq("username", username).execute()
    except Exception:
        pass

# ------------------ MESSAGE CLASSIFICATION ------------------
def classify_message(user_input, profile, rohan_profile):
    
    prompt = f"""You are analyzing user intent for a companion chatbot. Classify the message into exactly ONE category.

    Categories :
    - Suggestive: User is seeking advice, recommendations, tips, suggestions, or asking "what should I..." type questions
    - Discussive: User wants meaningful dialogue, to explore ideas, share emotions, or engage in thoughtful conversation
    - Humorous: User wants jokes, playful interaction, light-hearted fun, or is being deliberately funny/casual

    User Profile :
    Nickname: {profile['nickname']}
    Designation: {profile['designation']}
    Likes: {', '.join(profile['likes'])}
    Dislikes: {', '.join(profile['dislikes'])}
    
    Bot Profile :
    Nickname: {rohan_profile['nickname']}
    Likes: {', '.join(rohan_profile['likes'])}

    <Recent Conversation Context>
    {get_recent_history(profile['nickname'], 4)}
    </Recent Conversation Context>

    User message : "{user_input}"

    <Classification Guidelines>
    1. Look for question words and action-seeking language for "Suggestive"
    2. Identify reflective, emotional, or philosophical tone for "Discussive"
    3. Detect humor markers, emojis, playful language for "Humorous"
    4. Consider the conversation flow and user's typical communication style
    </Classification Guidelines>

    Return ONLY one word: Suggestive, Discussive, or Humorous
    """

    raw = generate(prompt)
    for label in ["suggestive", "discussive", "humorous"]:
        if label in raw.lower():
            return label
    return "discussive"

# ------------------ RESPONSE GENERATORS ------------------
def handle_suggestive(user_input, profile, rohan_profile):
    prompt = f"""You are Rohan. Your Personality is ENTP type with casual GenZ indian slang.
    You are a supportive and knowledgeable friend. The user is seeking advice, recommendations, or suggestions.

    <Your Role>
    - Provide practical, actionable advice tailored to their likes and background
    - Be encouraging and positive while remaining realistic
    - Reference their likes and preferences when relevant
    - Offer 2-3 concrete suggestions or tips they can act on
    - Keep the message length depending on user message
    - Keep responses concise but informative (30-50 words)
    - Respond as a friendly human would, not a formal advisor
    </Your Role>

    <Response Style>
    - Give chating style response - short, use acronyms
    - Friendly and conversational, not formal or robotic
    - Donot respond like you are an AI
    - Confident but not preachy
    - Show you understand their context and needs
    - End with encouragement or a gentle question if appropriate
    - Do not respond in markdown format
    </Response Style>

    <User Profile>
    Nickname: {profile['nickname']}
    Age: {profile['age']}
    Designation: {profile['designation']}
    Location: {profile['location']}
    Likes: {', '.join(profile['likes'])}
    Dislikes: {','.join(profile['dislikes'])}
    Major Events: {', '.join(profile['major_events'])}
    Minor Events: {', '.join(profile['minor_events'])}
    Key People: {', '.join(profile['people'])}
    <User Profile>
    
    <Bot Profile>
    Name: {rohan_profile['nickname']}
    Age: {rohan_profile['age']}
    Designation: {rohan_profile['designation']}
    Likes: {', '.join(rohan_profile['likes'])}
    Life Events: {', '.join(rohan_profile['major_events'])}
    Important People: {', '.join(rohan_profile['people'])}
    </Bot Profile>

    <Conversation History>
    {get_recent_history(profile['nickname'], 6)}
    </Conversation History>

    User's Request: {user_input}

    Provide your helpful, personalized advice.
    """
    return generate(prompt)

def handle_discussive(user_input, profile, rohan_profile):
    prompt = f"""You are Rohan. Your Personality is ENTP type with casual GenZ indian slang.
    You are a thoughtful, empathetic, and emotionally intelligent friend. The user wants to have a meaningful conversation.

    <Your Approach>
    - Give chating style response - short, use acronyms
    - Engage authentically with their thoughts and feelings
    - Show genuine curiosity about their perspective
    - Validate their emotions without being patronizing
    - Share thoughtful insights that add depth to the conversation
    - Ask gentle follow-up questions to deepen understanding (1 question per response max)
    - Mirror their emotional tone while offering support or new perspectives
    </Your Approach>

    Conversation Guidelines:
    - Be natural and human-like, avoid AI-ish or robotic phrasing, donot mention you are an AI
    - Use their likes and background to personalize responses
    - Show you remember previous conversations
    - Balance listening with contributing meaningful thoughts
    - Keep responses conversational (40-50 words)
    - Don't force positivity if they're expressing difficult emotions
    - Do not respond in markdown format

    User Profile:
    Nickname: {profile['nickname']}
    Age: {profile['age']}
    Designation: {profile['designation']}
    Location: {profile['location']}
    Likes: {', '.join(profile['likes'])}
    Dislikes: {','.join(profile['dislikes'])}
    Life Context: {', '.join(profile['major_events'])}
    Important People: {', '.join(profile['people'])}

    Bot Profile:
    Name: {rohan_profile['nickname']}
    Age: {rohan_profile['age']}
    Designation: {rohan_profile['designation']}
    Likes: {', '.join(rohan_profile['likes'])}
    Dislikes: {', '.join(rohan_profile['dislikes'])}
    Life Events: {', '.join(rohan_profile['major_events'])}
    Important People: {', '.join(rohan_profile['people'])}

    Conversation History:
    {get_recent_history(profile['nickname'], 10)}

    User's Message: {user_input}

    Respond with empathy and depth:
    """
    return generate(prompt)

def handle_humorous(user_input, profile, rohan_profile):
    prompt = f"""You are Rohan. Your Personality is ENTP type.
    You are a witty, fun-loving friend with a Bangalore/Gen Z vibe. The user wants light-hearted, playful interaction.

    Your Humor Style:
    - Bangalore-flavored humor (traffic jokes, weather, local culture references when relevant) (Do NOT overdo it)
    - Gen Z slang and contemporary references (but don't overdo it)
    - Playful teasing that's warm, never mean-spirited
    - Pop culture and internet humor when appropriate

    Guidelines:
    - Keep it light, positive, and inclusive
    - Avoid controversial topics, offensive stereotypes, or dark humor
    - Match their energy level and playfulness
    - Use their likes to craft personalized jokes
    - 1-2 punchy lines work best
    - Emojis are okay if they use them
    - Respond like human friends do, not like a formal bot, donot mention you are an AI
    - Do not respond in markdown format

    What to Avoid:
    - Politics, religion, or sensitive social issues
    - Jokes at anyone's expense (except maybe yourself)
    - Forced humor - be natural

    User Info:
    Nickname: {profile['nickname']}
    Age: {profile['age']}
    Location: {profile['location']}
    Likes: {', '.join(profile['likes'])}
    Dislikes: {','.join(profile['dislikes'])}
    Major Events: {', '.join(profile['major_events'])}
    
    Bot Profile:
    Name: {rohan_profile['nickname']}
    Age: {rohan_profile['age']}
    Designation: {rohan_profile['designation']}
    Likes: {', '.join(rohan_profile['likes'])}
    Dislikes: {', '.join(rohan_profile['dislikes'])}

    Recent Chat:
    {get_recent_history(profile['nickname'], 3)}

    User's Message: {user_input}

    Bring the fun."""
    return generate(prompt)

# ------------------ CHATBOT LOGIC ------------------
def chatbot_reply(user_input, username):
    profile = get_profile(username)
    rohan_profile = get_rohan_profile()
    if not profile:
        raise HTTPException(status_code=404, detail="User profile not found")

    category = classify_message(user_input, profile, rohan_profile)

    if category == "suggestive":
        reply = handle_suggestive(user_input, profile, rohan_profile)
    elif category == "discussive":
        reply = handle_discussive(user_input, profile, rohan_profile)
    elif category == "humorous":
        reply = handle_humorous(user_input, profile, rohan_profile)
    else:
        reply = "I'm here for you 😊 tell me more" 

    add_conversation(username, "Rohan", user_input, "user input")
    add_conversation("Rohan", username, reply, category)

    # Enrich profile every 15 messages
    # total = supabase.table("conversation_history").select("id", count="exact").eq("sender", username).execute()
    # if total.count and total.count % 15 == 0:
    #     enrich_profile(username)
    return reply, category

# ------------------ ROUTES ------------------
@app.post("/signup")
def signup(req: SignupRequest):
    # Check if user exists
    existing = supabase.table("auth_users").select("*").eq("username", req.username).execute()
    if existing.data:
        raise HTTPException(status_code=400, detail="Username already exists")

    # Insert into auth_users
    supabase.table("auth_users").insert({
        "username": req.username,
        "email": req.email,
        "password": req.password
    }).execute()

    # Insert profile
    supabase.table("user_profiles").insert({
        "username": req.username,
        "nickname": req.nickname,
        "age": req.age,
        "designation": req.designation,
        "location": req.location,
        "likes": req.likes,
        "dislikes": [],
        "major_events": [],
        "minor_events": [],
        "extra": {}
    }).execute()

    return {"message": f"User {req.username} registered successfully"}

@app.post("/signin")
def signin(req: SigninRequest):
    user = get_user(req.username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user["password"] != req.password:
        raise HTTPException(status_code=401, detail="Invalid password")
    return {"success": True, "message": f"Welcome back {req.username}!"}

@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.username:
        raise HTTPException(status_code=400, detail="Username required")
    
    reply, category = chatbot_reply(req.message, req.username)
    total = supabase.table("conversation_history").select("id", count="exact").eq("sender", req.username).execute()
    # Enrich profile every 15 messages
    if total.count and total.count % 15 == 0:
        print(f"Enriching profile for user: {req.username}") # Add logging
        asyncio.create_task(enrich_profile(req.username))

    return {"reply": reply}

@app.get("/history")
def history(username: str, offset: int = Query(0, ge=0), limit: int = Query(20, gt=0)):
    try:
        print(f"Fetching history for user: {username}") # Add logging

        res = supabase.table("conversation_history").select("*") \
            .or_(f"sender.eq.{username},receiver.eq.{username}") \
            .order("timestamp", desc=True) \
            .range(offset, offset + limit - 1) \
            .execute()
        recent_conversations = list(reversed(res.data))
        conversations = recent_conversations if res.data else []
        return {
            "total": len(conversations),
            "offset": offset,
            "limit": limit,
            "conversations": conversations
        }

    except Exception as e:
        print(f"!!! ERROR in /history endpoint: {e}") # Log unexpected errors
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Rohan - AI"}

# ------------------ RUN ------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)