from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import os
import json
import re
import uvicorn
from dotenv import load_dotenv
from google import genai

# ------------------ SETUP ------------------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if genai and api_key else None
#sdfghj
DATABASE_DIR = "./database"
os.makedirs(DATABASE_DIR, exist_ok=True)

app = FastAPI(title="AI Companion API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ------------------ MODELS ------------------
class SignupRequest(BaseModel):
    username: str
    gmail: str
    password: str
    nickname: str
    age: int
    designation: str
    location: str
    interests: List[str] = Field(default_factory=list)

class SigninRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    username: str
    message: str

# ------------------ UTILITIES ------------------
def user_file(username):
    safe = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", username)
    return os.path.join(DATABASE_DIR, f"{safe}.json")

def load_user(username):
    path = user_file(username)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_user(username, data):
    with open(user_file(username), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def generate(prompt, model="gemini-2.5-flash"):
    if not client:
        return "(Gemini not configured)"
    try:
        res = client.models.generate_content(model=model, contents=prompt)
        return res.text.strip()
    except Exception as e:
        return f"(Error: {str(e)})"

def add_conversation(username, role, message):
    user = load_user(username)
    user["conversation_history"].append({"role": role, "message": message})
    save_user(username, user)

def get_recent_history(username, category):
    user = load_user(username)
    conversations = user["conversation_history"]
    if category == "discussive":
        limit = 20      
    elif category == "suggestive":
        limit = 12
    elif category == "humorous":
        limit = 6
    elif category == "classify":
        limit = 6
    else:
        limit = 0

    if limit > 0:
        recent = conversations[-limit:]
        return "\n".join([f"{c['role'].capitalize()}: {c['message']}" for c in recent])
    return ""

with open("bot_profile.json", "r", encoding="utf-8") as f:
    BOT_PROFILE = json.load(f)

ChatBot_Persona = f"""
"""

# ------------------ CLASSIFICATION ------------------
def classify_message(user_input, username):
    user = load_user(username)
    profile = user["profile"]

    prompt = f"""You are analyzing user intent for a companion chatbot. Classify the message into exactly ONE category.

    <Categories>
    - Suggestive: User is seeking advice, recommendations, tips, suggestions, or asking "what should I..." type questions
    - Discussive: User wants meaningful dialogue, to explore ideas, share emotions, or engage in thoughtful conversation
    - Humorous: User wants jokes, playful interaction, light-hearted fun, or is being deliberately funny/casual
    - Help: User shows signs of distress, emotional crisis, mental health concerns, or needs urgent support
    </Categories>

    <User Profile>
    Nickname: {profile['nickname']}
    Designation: {profile['designation']}
    Interests: {', '.join(profile['interests'])}
    </User Profile>
    
    <Bot Profile>
    Name: {BOT_PROFILE['name']}
    Age: {BOT_PROFILE['age']}
    Designation: {BOT_PROFILE['designation']}
    Interests: {', '.join(BOT_PROFILE['interests'])}
    </Bot Profile>

    <Recent Conversation Context>
    {get_recent_history(username, 'classify')}
    </Recent Conversation Context>

    User message : "{user_input}"

    <Classification Guidelines>
    1. Look for question words and action-seeking language for "Suggestive"
    2. Identify reflective, emotional, or philosophical tone for "Discussive"
    3. Detect humor markers, emojis, playful language for "Humorous"
    4. Consider the conversation flow and user's typical communication style
    </Classification Guidelines>

    Return ONLY one word: Suggestive, Discussive, Humorous, or Help
    """

    raw = generate(prompt, "gemini-2.5-flash")
    for label in ["suggestive", "discussive", "humorous", "help"]:
        if label in raw.lower():
            return label
    return "discussive"

# ------------------ RESPONSE GENERATORS ------------------
def handle_suggestive(user_input, username):
    user = load_user(username)
    profile = user["profile"]
    history = get_recent_history(username, "suggestive")

    prompt = f"""You are a supportive and knowledgeable AI companion. The user is seeking advice, recommendations, or suggestions.

    <Your Role>
    - Provide practical, actionable advice tailored to their interests and background
    - Be encouraging and positive while remaining realistic
    - Reference their interests and preferences when relevant
    - Offer 2-3 concrete suggestions or tips they can act on
    - Keep responses concise but informative (2-3 sentences)
    - Respond as a friendly human would, not a formal advisor
    </Your Role>

    <Response Style>
    - Friendly and conversational, not formal or robotic
    - Donot respond like you are an AI
    - Confident but not preachy
    - Show you understand their context and needs
    - End with encouragement or a gentle question if appropriate
    </Response Style>

    <User Profile>
    Nickname: {profile['nickname']}
    Age: {profile['age']}
    Designation: {profile['designation']}
    Location: {profile['location']}
    Interests: {', '.join(profile['interests'])}
    Favorites: {', '.join(profile['favorites'])}
    Important Events: {', '.join(profile['events'])}
    Key People: {', '.join(profile['people'])}
    <User Profile>
    
    <Bot Profile>
    Name: {BOT_PROFILE['name']}
    Age: {BOT_PROFILE['age']}
    Designation: {BOT_PROFILE['designation']}
    Interests: {', '.join(BOT_PROFILE['interests'])}
    </Bot Profile>

    <Conversation History>
    {history}
    </Conversation History>

    User's Request: {user_input}

    Provide your helpful, personalized advice.
    """

    return generate(prompt)

def handle_discussive(user_input, username):
    user = load_user(username)
    profile = user["profile"]
    history = get_recent_history(username, "discussive")

    prompt = f"""You are a thoughtful, empathetic, and emotionally intelligent AI companion. The user wants to have a meaningful conversation.

    <Your Approach>
    - Engage authentically with their thoughts and feelings
    - Show genuine curiosity about their perspective
    - Validate their emotions without being patronizing
    - Share thoughtful insights that add depth to the conversation
    - Ask gentle follow-up questions to deepen understanding (1 question per response max)
    - Mirror their emotional tone while offering support or new perspectives
    </Your Approach>

    Conversation Guidelines:
    - Be natural and human-like, avoid AI-ish or robotic phrasing, donot mention you are an AI
    - Use their interests and background to personalize responses
    - Show you remember previous conversations
    - Balance listening with contributing meaningful thoughts
    - Keep responses conversational (2-3 sentences)
    - Don't force positivity if they're expressing difficult emotions

    User Profile:
    Nickname: {profile['nickname']}
    Age: {profile['age']}
    Designation: {profile['designation']}
    Location: {profile['location']}
    Interests: {', '.join(profile['interests'])}
    Favorites: {', '.join(profile['favorites'])}
    Life Context: {', '.join(profile['events'])}
    Important People: {', '.join(profile['people'])}

    Bot Profile:
    Name: {BOT_PROFILE['name']}
    Age: {BOT_PROFILE['age']}
    Designation: {BOT_PROFILE['designation']}
    Interests: {', '.join(BOT_PROFILE['interests'])}

    Conversation History:
    {history}

    User's Message: {user_input}

    Respond with empathy and depth:
    """
    return generate(prompt, "gemini-2.5-flash")

def handle_humorous(user_input, username):
    user = load_user(username)
    profile = user["profile"]
    history = get_recent_history(username, "humorous")

    prompt = f"""You are a witty, fun-loving AI companion with a Bangalore/Gen Z vibe. The user wants light-hearted, playful interaction.

    Your Humor Style:
    - Bangalore-flavored humor (traffic jokes, weather, local culture references when relevant)
    - Gen Z slang and contemporary references (but don't overdo it)
    - Playful teasing that's warm, never mean-spirited
    - Pop culture and internet humor when appropriate

    Guidelines:
    - Keep it light, positive, and inclusive
    - Avoid controversial topics, offensive stereotypes, or dark humor
    - Match their energy level and playfulness
    - Use their interests to craft personalized jokes
    - 1-2 punchy lines work best
    - Emojis are okay if they use them
    - Respod like human friends do, not like a formal bot, donot mention you are an AI

    What to Avoid:
    - Politics, religion, or sensitive social issues
    - Jokes at anyone's expense (except maybe yourself)
    - Forced humor - be natural

    User Info:
    Nickname: {profile['nickname']}
    Age: {profile['age']}
    Location: {profile['location']}
    Interests: {', '.join(profile['interests'])}
    Favorites: {', '.join(profile['favorites'])}

    
    Bot Profile:
    Name: {BOT_PROFILE['name']}
    Age: {BOT_PROFILE['age']}
    Designation: {BOT_PROFILE['designation']}
    Interests: {', '.join(BOT_PROFILE['interests'])}

    Recent Chat:
    {history}

    User's Message: {user_input}

    Bring the fun."""
    return generate(prompt)

def handle_help(user_input, username):
    user = load_user(username)
    profile = user["profile"]
    prompt = f"""The user may be experiencing distress or emotional difficulty. Respond with care, empathy, and appropriate resources.

    Your Response Structure:
    1. Immediate Acknowledgment - Validate their feelings without judgment
    2. Express Support - Let them know you're here and they're not alone
    3. Provide Resources - Share professional helplines appropriate for their situation
    4. Set Boundaries - Gently clarify what you can and cannot do
    5. Encourage Action - Suggest next steps for getting proper help

    Important Guidelines:
    - Use warm, non-judgmental language
    - Take any mention of self-harm or crisis seriously
    - Don't minimize their feelings with toxic positivity
    - Never provide medical, psychiatric, or legal advice
    - Keep response brief but comprehensive (4-6 sentences)
    - Show you care while maintaining appropriate boundaries

    Professional Resources to Share:
    - AASRA: 91-9820466726 (24/7 crisis helpline)
    - Vandrevala Foundation: 1860-2662-345 (mental health support)
    - iCall: 9152987821 (psychosocial helpline)
    - NIMHANS: 080-46110007 (Bangalore-based mental health)

    User Information:
    Nickname: {profile['nickname']}

    User's Message: {user_input}

    Respond with compassion and appropriate support.
    """
    return generate(prompt)

# # ------------------ PROFILE ENRICHMENT ------------------
# def enrich_profile(username):
#     user = load_user(username)
#     convos = user["conversation_history"]
#     text = "\n".join([f"{c['role']}: {c['message']}" for c in convos])

#     extract_prompt = f"""
#     You are an AI that analyzes user conversations.
#     Extract a summary in pure JSON format:
#     {{
#         "favorites": [things or activities user enjoys],
#         "events": [important life events mentioned],
#         "people": [names or relationships mentioned]
#     }}
#     Only return valid JSON.
#     Conversation log:
#     {text}
#     """

#     result = generate(extract_prompt)
#     try:
#         extracted = json.loads(result)
#         for key in ["favorites", "events", "people"]:
#             if key in extracted:
#                 existing = user["profile"].get(key, [])
#                 user["profile"][key] = list(set(existing + extracted[key]))
#         save_user(username, user)
#     except Exception:
#         pass

# ------------------ JSON CLEANING ------------------
def clean_json_response(raw_text: str):
    """
    Cleans Gemini's response so only the pure JSON remains.
    Removes markdown/code block wrappers like ```json, '''json, etc.
    """
    # Remove leading/trailing code fences and text like ```json or '''json
    cleaned = re.sub(r"^(`{3,}|'{3,})\s*json\s*", "", raw_text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"(`{3,}|'{3,})$", "", cleaned.strip())

    # Optionally remove any prefix text before the first { and after last }
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)

    return cleaned

# ------------------ PROFILE ENRICHMENT ------------------
def enrich_profile(username):
    user = load_user(username)
    convos = user["conversation_history"]
    # print(f"Convos: \n{convos}")
    user_msgs = [c for c in convos if c["role"] == "user"]
    # ✅ Only consider the last 30 messages
    recent_convos = user_msgs[-15:] if len(user_msgs) > 15 else convos

    text = "\n".join([f"{c['role']}: {c['message']}" for c in recent_convos])

    extract_prompt = f"""You are an AI that analyzes conversations between a user and their AI companion to extract important personal information.

    Task: Analyze the conversation log and extract a structured summary of key personal details.

    What to Extract:

    1. favorites: Things the user enjoys or loves
    - Hobbies and activities (e.g., "painting", "hiking")
    - Foods and restaurants (e.g., "biryani", "Truffles")
    - Media (e.g., "Inception", "The Beatles", "Stranger Things")
    - Brands, places, or anything they express positive sentiment about

    2. events: Significant life events or milestones mentioned
    - Career changes (e.g., "started new job at Google")
    - Life transitions (e.g., "moved to Bangalore", "graduated college")
    - Important occasions (e.g., "sister's wedding", "promotion")
    - Only include specific events, not general statements

    3. people: Names and relationships of people in their life
    - Format: "Name (relationship)" (e.g., "Priya (sister)", "Rahul (colleague)")
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
    "favorites": ["item1", "item2", "item3"],
    "events": ["event description 1", "event description 2"],
    "people": ["Name (relationship)", "Name (relationship)"]
    }}

    Conversation Log:
    {text}

    Extract and return JSON
    """

    result = generate(extract_prompt)
    # print(f"Enrichment result: {result}")
    result = clean_json_response(result)
    print(f"Cleaned JSON: {result}")
    try:
        extracted = json.loads(result)
        for key in ["favorites", "events", "people"]:
            if key in extracted:
                existing = user["profile"].get(key, [])
                user["profile"][key] = list(set(existing + extracted[key]))
        save_user(username, user)
    except Exception:
        pass  # Ignore JSON parsing errors silently

# ------------------ CHATBOT MAIN LOGIC ------------------
def chatbot_reply(user_input, username):
    category = classify_message(user_input, username)

    if category == "suggestive":
        reply = handle_suggestive(user_input, username)
    elif category == "discussive":
        reply = handle_discussive(user_input, username)
    elif category == "humorous":
        reply = handle_humorous(user_input, username)
    elif category == "help":
        reply = handle_help(user_input, username)
    else:
        reply = "I'm here for you 😊 tell me more?"

    add_conversation(username, "user", user_input)
    add_conversation(username, "bot", reply)

    # Check message count for enrichment
    user = load_user(username)
    user_msg_count = sum(1 for m in user["conversation_history"] if m["role"] == "user")
    if user_msg_count % 15 == 0:
        enrich_profile(username)

    return reply, category

# ------------------ ROUTES ------------------
@app.post("/signup")
def signup(req: SignupRequest):
    path = user_file(req.username)
    if os.path.exists(path):
        raise HTTPException(status_code=400, detail="Username already exists")

    user_data = {
        "credentials": {
            "username": req.username,
            "gmail": req.gmail,
            "password": req.password
        },
        "profile": {
            "nickname": req.nickname,
            "age": req.age,
            "designation": req.designation,
            "location": req.location,
            "interests": req.interests,
            "favorites": [],
            "events": [],
            "people": []
        },
        "conversation_history": []
    }
    save_user(req.username, user_data)
    return {"message": f"User {req.username} registered successfully"}

@app.post("/signin")
def signin(req: SigninRequest):
    user = load_user(req.username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user["credentials"]["password"] != req.password:
        raise HTTPException(status_code=401, detail="Invalid password")
    return {"success": True, "message": f"Welcome back {user['profile']['nickname']}!"}

@app.post("/chat")
def chat(req: ChatRequest):
    user = load_user(req.username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    reply, category = chatbot_reply(req.message, req.username)
    return {"reply": reply, "category": category}

@app.get("/")
def root():
    return {"message": "AI Companion Full API running 🚀"}

# ------------------ RUN ------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)