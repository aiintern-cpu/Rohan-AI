from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
from dotenv import load_dotenv
from supabase import create_client
from google import genai
from neo4j import GraphDatabase
import os
import json
import re
import uvicorn
import asyncio
from threading import Lock
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import datetime, timedelta, timezone
# ------------------ CONFIG ------------------
load_dotenv()

# --- JWT CONFIG ---
SECRET_KEY = "super-secret-key-change-this"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="signin")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if genai and GEMINI_API_KEY else None

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASS = os.getenv("NEO4J_PASS")
NEO4J_DB = os.getenv("NEO4J_DB", "rohan-db")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
neo4j_lock = Lock()

LOG_FILE = "gemini_usage_log.jsonl"

# ------------------ FASTAPI ------------------
app = FastAPI(title="Rohan - AI Companion (Supabase + Neo4j)")
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

class TokenData(BaseModel):
    username: str | None = None

# ------------------ UTILITIES ------------------
def generate(prompt: str, model: str = "gemini-2.5-pro") -> str:
    if not client:
        return "(Gemini not configured)"
    try:
        res = client.models.generate_content(model=model, contents=prompt)
        usage = getattr(res, "usage_metadata", None)
        if usage:
            prompt_tokens = getattr(usage, "prompt_token_count", None)
            response_tokens = getattr(usage, "candidates_token_count", None)
            total_tokens = getattr(usage, "total_token_count", None)
            log_entry = {
                "response": getattr(res, "text", "").strip()[:50],
                "model": model,
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "total_tokens": total_tokens
            }
            try:
                with open(LOG_FILE, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
            except Exception:
                pass
        return getattr(res, "text", "").strip()
    except Exception:
        return f"Error : {str(Exception)}"
    

# --- Helper to create token ---
def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# --- Helper to verify token and get current user ---
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return username


# def create_access_token(data: dict, expires_delta: timedelta | None = None):
#     to_encode = data.copy()
#     expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
#     to_encode.update({"exp": expire})
#     encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
#     return encoded_jwt


def clean_json_response(raw_text: str) -> str:
    cleaned = re.sub(r"^(`{3,}|'{3,})\s*json\s*", "", raw_text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"(`{3,}|'{3,})$", "", cleaned.strip())
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    return cleaned

def sanitize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Remove empty or invalid entries before pushing to Neo4j"""
    def valid_dict_list(lst, key):
        return [item for item in lst if isinstance(item, dict) and item.get(key)]

    if "likes" in payload:
        payload["likes"] = valid_dict_list(payload["likes"], "name")
    if "dislikes" in payload:
        payload["dislikes"] = valid_dict_list(payload["dislikes"], "name")
    if "major_events" in payload:
        payload["major_events"] = valid_dict_list(payload["major_events"], "event_name")
    if "people" in payload:
        payload["people"] = valid_dict_list(payload["people"], "name")

    return payload


# ------------------ NEO4J CYLPHER HELPERS ------------------
def create_user_profile_tx(tx, user_data: Dict[str, Any]):
    print("user_data in create_user_profile_tx:", user_data)
    query = """
    MERGE (u:User {username: $username})
    SET u.nickname = $nickname,
        u.age = $age,
        u.designation = $designation,
        u.location = $location
    WITH u
    UNWIND $likes AS like
      MERGE (i:Interest {name: like.name})
      SET i.type = like.type
      MERGE (u)-[:LIKES]->(i)
    WITH u
    UNWIND $dislikes AS dislike
      MERGE (d:Interest {name: dislike.name})
      SET d.type = dislike.type
      MERGE (u)-[:DISLIKES]->(d)
    """
    print("Running create_user_profile_tx with query:", query)
    tx.run(query, **user_data)

def get_user_profile_tx(tx, username: str):
    query = """
    MATCH (u:User {username: $username})
    OPTIONAL MATCH (u)-[:LIKES]->(l:Interest)
    OPTIONAL MATCH (u)-[:DISLIKES]->(d:Interest)
    OPTIONAL MATCH (u)-[:HAS_EVENT]->(e:Event)
    OPTIONAL MATCH (u)-[:KNOWS]->(p:Person)
    RETURN u.username AS username,
           u.nickname AS nickname,
           u.age AS age,
           u.designation AS designation,
           u.location AS location,
           collect(DISTINCT {name: l.name, type: l.type}) AS likes,
           collect(DISTINCT {name: d.name, type: d.type}) AS dislikes,
           collect(DISTINCT {event_name: e.name, description: e.description}) AS major_events,
           collect(DISTINCT {name: p.name, relationship: p.relationship}) AS people
    """
    return tx.run(query, username=username).single()

def update_user_profile_tx(tx, username: str, update_payload: Dict[str, Any]):
    """
    Safely updates Neo4j User node and related nodes.
    Compatible with Gemini enrichment JSON format.
    """
    set_parts = []
    params = {"username": username}

    # --- Scalar properties (User node) ---
    for key in ["nickname", "age", "designation", "location"]:
        val = update_payload.get(key)
        if val not in [None, "", []]:
            set_parts.append(f"u.{key} = ${key}")
            params[key] = val

    set_clause = "SET " + ", ".join(set_parts) if set_parts else ""

    # --- Start query ---
    query = f"""
    MATCH (u:User {{username: $username}})
    {set_clause}
    """

    # --- Likes ---
    if update_payload.get("likes"):
        valid_likes = [l for l in update_payload["likes"] if l.get("name")]
        if valid_likes:
            query += """
            WITH u
            FOREACH (like IN $likes |
                MERGE (i:Interest {name: like.name})
                SET i.type = like.type
                MERGE (u)-[:LIKES]->(i)
            )
            """
            params["likes"] = valid_likes

    # --- Dislikes ---
    if update_payload.get("dislikes"):
        valid_dislikes = [d for d in update_payload["dislikes"] if d.get("name")]
        if valid_dislikes:
            query += """
            WITH u
            FOREACH (dislike IN $dislikes |
                MERGE (d:Interest {name: dislike.name})
                SET d.type = dislike.type
                MERGE (u)-[:DISLIKES]->(d)
            )
            """
            params["dislikes"] = valid_dislikes

    # --- Major Events ---
    if update_payload.get("major_events"):
        valid_events = [e for e in update_payload["major_events"] if e.get("event_name")]
        if valid_events:
            query += """
            WITH u
            FOREACH (ev IN $major_events |
                MERGE (e:Event {name: ev.event_name})
                SET e.description = ev.description
                MERGE (u)-[:HAS_EVENT]->(e)
            )
            """
            params["major_events"] = valid_events

    # --- People ---
    if update_payload.get("people"):
        valid_people = [p for p in update_payload["people"] if p.get("name")]
        if valid_people:
            query += """
            WITH u
            FOREACH (person IN $people |
                MERGE (p:Person {name: person.name})
                SET p.relationship = person.relationship
                MERGE (u)-[:KNOWS]->(p)
            )
            """
            params["people"] = valid_people

    query += "\nRETURN count(u) AS updated;"

    print("🚀 Running update_user_profile_tx Cypher:")
    print(query)
    print("Params:", json.dumps(params, indent=2, default=str))

    tx.run(query, **params)
    print(f"✅ Profile updated in Neo4j for {username}")

# ------------------ SUPABASE HELPERS ------------------
def get_user(username: str) -> Optional[Dict[str, Any]]:
    res = supabase.table("auth_users_2").select("*").eq("username", username).execute()
    if not res.data:
        return None
    return res.data[0]

def get_rohan_profile() -> Optional[Dict[str, Any]]:
    res = supabase.table("user_profiles").select("*").eq("username", "Rohan").execute()
    if not res.data:
        return None
    return res.data[0]

def get_recent_history(username: str, limit: int = 10):
    res = supabase.table("conversation_history_2").select("*") \
        .or_(f"sender.eq.{username},receiver.eq.{username}") \
        .order("timestamp", desc=True).limit(limit).execute()
    recent_conversations = list(reversed(res.data)) if res.data else []
    return recent_conversations

def add_conversation(sender: str, receiver: str, message: str, category: str):
    supabase.table("conversation_history_2").insert({
        "sender": sender,
        "receiver": receiver,
        "message": message,
        "category": category
    }).execute()

# ------------------ NEO4J PROFILE CRUD ------------------
def create_profile_in_neo4j(user_dict: Dict[str, Any]):
    print("Creating profile in Neo4j for user_dict:", user_dict)
    likes = [{"name": l, "type": ""} for l in user_dict.get("likes", [])]
    dislikes = [{"name": d, "type": ""} for d in user_dict.get("dislikes", [])]
    payload = {
        "username": user_dict["username"],
        "nickname": user_dict["nickname"],
        "age": user_dict["age"],
        "designation": user_dict["designation"],
        "location": user_dict["location"],
        "likes": likes,
        "dislikes": dislikes
    }
    with driver.session(database=NEO4J_DB) as session:
        print("Payload for create_profile_in_neo4j:", payload)
        session.execute_write(create_user_profile_tx, payload)
        print("✅ User profile created in Neo4j!")

def get_profile(username: str) -> Optional[Dict[str, Any]]: 
    with driver.session(database=NEO4J_DB) as session:
        rec = session.execute_read(get_user_profile_tx, username)
        if not rec:
            return None
        def _normalize_list_of_maps(val):
            if not val:
                return []
            normalized = []
            for item in val:
                if hasattr(item, "items"):
                    normalized.append(dict(item))
                else:
                    normalized.append(item)
            return normalized
        likes = _normalize_list_of_maps(rec["likes"])
        dislikes = _normalize_list_of_maps(rec["dislikes"])
        major_events = _normalize_list_of_maps(rec["major_events"])
        people_maps = _normalize_list_of_maps(rec["people"])
        people = []
        for p in people_maps:
            name = p.get("name") if isinstance(p, dict) else None
            rel = p.get("relationship") if isinstance(p, dict) else ""
            if name:
                if rel:
                    people.append(f"{name} ({rel})")
                else:
                    people.append(name)
        return {
            "username": rec["username"],
            "nickname": rec["nickname"] or rec["username"],
            "age": rec["age"] or None,
            "designation": rec["designation"] or "",
            "location": rec["location"] or "",
            "likes": likes,
            "dislikes": dislikes,
            "major_events": major_events,
            "people": people
        }

def update_profile_in_neo4j(username: str, update_payload: Dict[str, Any]):
    """
    Thread-safe Neo4j update function.
    Prevents concurrent writes that could deadlock or hang the driver.
    """
    with neo4j_lock:  # 🚨 ensures only one thread writes at a time
        with driver.session(database=NEO4J_DB) as session:
            print(f"🔐 Acquired lock. Updating Neo4j profile for user: {username}")
            session.execute_write(update_user_profile_tx, username, update_payload)
            print(f"✅ Released lock after updating {username}")

# ------------------ ENRICH PROFILE (ASYNC, safe updates) ------------------

async def enrich_profile(username: str):
    """
    Extracts structured profile info using Gemini and updates Neo4j safely.
    Compatible with JSON schema:
    {
      "nickname": "string or null",
      "age": int or null,
      "designation": "string or null",
      "location": "string or null",
      "likes": [{"name": "item", "type": "hobby/food/media/brand/etc."}],
      "dislikes": [{"name": "item", "type": "hobby/food/media/brand/etc."}],
      "major_events": [{"event_name": "string", "description": "short description"}],
      "people": [{"name": "string", "relationship": "string"}]
    }
    """
    try:
        # 1️⃣ Fetch last 15 user messages
        res = supabase.table("conversation_history_2").select("*") \
            .eq("sender", username).order("timestamp", desc=True).limit(15).execute()
        convos = list(reversed(res.data)) if res.data else []
        text = "\n".join([c["message"] for c in convos if "message" in c])
        if not text:
            print(f"ℹ️ No messages found for {username}")
            return

        # 2️⃣ Extract info using Gemini
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

        3. major_events: Significant life events or milestones mentioned
        - Career changes (e.g., "started new job at Google")
        - Life transitions (e.g., "moved to Bangalore", "graduated college")
        - Important occasions (e.g., "sister's wedding", "promotion")
        - Only include specific events, not general statements

        4. people: Names and relationships of people in their life
        - Format: "Name (relationship)" (e.g., "Priya (sister)", "Alex (colleague)")
        - Include family, friends, colleagues, partners
        - Only include if both name AND relationship are mentioned

        Extraction Rules:
        - Only extract information explicitly stated in the conversation
        - Use lower case for all entries (even proper nouns)
        - Don't infer or assume information not directly mentioned
        - Use exact quotes or paraphrasing from the conversation
        - If a category has no clear information, use an empty array []
        - Maintain consistent formatting across all entries
        - Remove duplicates

        Output Format:
        Return ONLY valid JSON with no additional text, explanations, or markdown:

        Conversation Log:
        {text}

        Output JSON Schema:
        {{
        "nickname": "string or null",
        "age": int or null,
        "designation": "string or null",
        "location": "string or null",
        "likes": [{{"name": "item", "type": "hobby/food/media/brand/etc."}}],
        "dislikes": [{{"name": "item", "type": "hobby/food/media/brand/etc."}}],
        "major_events": [{{"event_name": "string", "description": "short description"}}],
        "people": [{{"name": "string", "relationship": "string"}}]
        }}

        Extract and return JSON
        """
        raw = generate(extract_prompt, "gemini-2.5-pro")
        cleaned = clean_json_response(raw)
        extracted = json.loads(cleaned)
        print("🧠 Enrichment extracted:", json.dumps(extracted, indent=2))

        # 3️⃣ Fetch existing profile
        existing = get_profile(username)
        if not existing:
            print(f"⚠️ No Neo4j profile found for {username}")
            return

        # 4️⃣ Merge new + old data
        payload: Dict[str, Any] = {}

        # Scalars
        for key in ["nickname", "age", "designation", "location"]:
            new_val = extracted.get(key)
            old_val = existing.get(key)
            if new_val not in [None, "", 0] and new_val != old_val:
                payload[key] = new_val

        # Merge lists of dicts safely
        def merge_dict_lists(old_list, new_list, key_field):
            if not isinstance(old_list, list): old_list = []
            if not isinstance(new_list, list): new_list = []
            existing_keys = {item.get(key_field) for item in old_list if isinstance(item, dict)}
            merged = old_list[:]
            for item in new_list:
                if isinstance(item, dict) and item.get(key_field) and item.get(key_field) not in existing_keys:
                    merged.append(item)
            return merged

        for field, key_field in [("likes", "name"), ("dislikes", "name"),
                                 ("major_events", "event_name"), ("people", "name")]:
            if extracted.get(field):
                merged = merge_dict_lists(existing.get(field, []), extracted[field], key_field)
                if merged != existing.get(field, []):
                    payload[field] = merged

        # 5️⃣ Sanitize & update
        payload = sanitize_payload(payload)
        if payload:
            print("🔄 Updating Neo4j with payload:", json.dumps(payload, indent=2))
            update_profile_in_neo4j(username, payload)
        else:   
            print(f"ℹ️ No new data to update for {username}")

    except Exception as e:
        print(f"❌ enrich_profile error for {username}: {e}")

# ------------------ MESSAGE CLASSIFICATION & RESPONSE ------------------
def classify_message(user_input: str, profile: Dict[str, Any], rohan_profile: Dict[str, Any]) -> str:

    likes = ', '.join([f"{like['name']}({like['type']})" for like in profile.get('likes', [])])
    prompt = f"""You are analyzing user intent for a companion chatbot. Classify the message into exactly ONE category.

    Categories :
    - Suggestive: User is seeking advice, recommendations, tips, suggestions, or asking "what should I..." type questions
    - Discussive: User wants meaningful dialogue, to explore ideas, share emotions, or engage in thoughtful conversation
    - Humorous: User wants jokes, playful interaction, light-hearted fun, or is being deliberately funny/casual

    User Profile :
    Nickname: {profile['nickname']}
    Designation: {profile['designation']}
    Likes: {', '.join([f"{like['name']}({like['type']})" for like in profile.get('likes', [])])}
    Dislikes: {', '.join([f"{dislike['name']}({dislike['type']})" for dislike in profile.get('dislikes', [])])}
    
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
    raw_l = (raw or "").lower()
    if "suggest" in raw_l:
        return "suggestive"
    if "humor" in raw_l or "humorous" in raw_l:
        return "humorous"
    return "discussive"

def handle_suggestive(user_input: str, profile: Dict[str, Any], rohan_profile: Dict[str, Any]) -> str:
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
    Likes: {', '.join([f"{like['name']}({like['type']})" for like in profile.get('likes', [])])}
    Dislikes: {', '.join([f"{dislike['name']}({dislike['type']})" for dislike in profile.get('dislikes', [])])}
    Major Events: {', '.join([f"{event['event_name']}({event['description']})" for event in profile.get('major_events', [])])}
    Key People: {', '.join([f"{person['name']}({person['relationship']})" for person in profile.get('people', [])])}
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

def handle_discussive(user_input: str, profile: Dict[str, Any], rohan_profile: Dict[str, Any]) -> str:
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
    Likes: {', '.join([f"{like['name']}({like['type']})" for like in profile.get('likes', [])])}
    Dislikes: {', '.join([f"{dislike['name']}({dislike['type']})" for dislike in profile.get('dislikes', [])])}
    Major Events: {', '.join([f"{event['event_name']}({event['description']})" for event in profile.get('major_events', [])])}
    Key People: {', '.join([f"{person['name']}({person['relationship']})" for person in profile.get('people', [])])}

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

def handle_humorous(user_input: str, profile: Dict[str, Any], rohan_profile: Dict[str, Any]) -> str:
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
    Likes: {', '.join([f"{like['name']}({like['type']})" for like in profile.get('likes', [])])}
    Dislikes: {', '.join([f"{dislike['name']}({dislike['type']})" for dislike in profile.get('dislikes', [])])}
    Major Events: {', '.join([f"{event['event_name']}({event['description']})" for event in profile.get('major_events', [])])}
    Key People: {', '.join([f"{person['name']}({person['relationship']})" for person in profile.get('people', [])])}

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

def chatbot_reply(user_input: str, username: str):
    profile = get_profile(username)
    print("profile for chatbot_reply:", profile)
    if not profile:
        raise HTTPException(status_code=404, detail="User profile not found")
    rohan_profile = get_rohan_profile() or {"nickname": "Rohan", "likes": []}
    category = classify_message(user_input, profile, rohan_profile)
    if category == "suggestive":
        reply = handle_suggestive(user_input, profile, rohan_profile)
    elif category == "humorous":
        reply = handle_humorous(user_input, profile, rohan_profile)
    else:
        reply = handle_discussive(user_input, profile, rohan_profile)
    add_conversation(username, "Rohan", user_input, "user input")
    add_conversation("Rohan", username, reply, category)
    return reply, category

# ------------------ ROUTES ------------------
@app.post("/signup")
def signup(req: SignupRequest):
    existing = supabase.table("auth_users_2").select("*").eq("username", req.username).execute()
    if existing.data:
        raise HTTPException(status_code=400, detail="Username already exists")
    supabase.table("auth_users_2").insert({
        "username": req.username,
        "email": req.email,
        "password": req.password
    }).execute()
    user_data = {
        "username": req.username,
        "nickname": req.nickname,
        "age": req.age,
        "designation": req.designation,
        "location": req.location,
        "likes": req.likes or []
    }
    try:
        print("Creating profile in Neo4j for user:", req.username)
        create_profile_in_neo4j(user_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to create user profile")
    return {"message": f"User {req.username} registered successfully"}

# @app.post("/signin")
# def signin(req: SigninRequest):
#     user = get_user(req.username)
#     if not user:
#         raise HTTPException(status_code=404, detail="User not found")
#     if user["password"] != req.password:
#         raise HTTPException(status_code=401, detail="Invalid password")
#     return {"success": True, "message": f"Welcome back {req.username}!"}

@app.post("/signin")
async def signin(form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user(form_data.username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user["password"] != form_data.password:
        raise HTTPException(status_code=401, detail="Invalid password")

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )

    return {"access_token": access_token, "token_type": "bearer"}

# @app.post("/chat")
# async def chat(req: ChatRequest):
#     if not req.username:
#         raise HTTPException(status_code=400, detail="Username required")
#     reply, category = chatbot_reply(req.message, req.username)

#     total = supabase.table("conversation_history_2").select("id", count="exact").eq("sender", req.username).execute()
#     # and total.count % 3 == 0
#     if total.count  :
#         asyncio.create_task(enrich_profile(req.username))

#     return {"reply": reply, "category": category}

@app.post("/chat")
async def chat(req: ChatRequest, current_user: str = Depends(get_current_user)):
    username = current_user  # Extracted from token
    reply, category = chatbot_reply(req.message, username)

    total = supabase.table("conversation_history_2").select("id", count="exact").eq("sender", username).execute()
    if total.count and total.count % 15 == 0:
        asyncio.create_task(enrich_profile(username))

    return {"reply": reply, "category": category}


# @app.get("/history")
# def history(username: str, offset: int = Query(0, ge=0), limit: int = Query(20, gt=0)):
#     try:
#         res = supabase.table("conversation_history_2").select("*") \
#             .or_(f"sender.eq.{username},receiver.eq.{username}") \
#             .order("timestamp", desc=True) \
#             .range(offset, offset + limit - 1) \
#             .execute()
#         recent_conversations = list(reversed(res.data)) if res.data else []
#         return {
#             "total": len(recent_conversations),
#             "offset": offset,
#             "limit": limit,
#             "conversations": recent_conversations
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def history(
    offset: int = Query(0, ge=0),
    limit: int = Query(20, gt=0),
    current_user: str = Depends(get_current_user)
):
    username = current_user
    try:
        res = supabase.table("conversation_history_2").select("*") \
            .or_(f"sender.eq.{username},receiver.eq.{username}") \
            .order("timestamp", desc=True) \
            .range(offset, offset + limit - 1) \
            .execute()
        recent_conversations = list(reversed(res.data)) if res.data else []
        return {
            "total": len(recent_conversations),
            "offset": offset,
            "limit": limit,
            "conversations": recent_conversations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/verify-token")
async def verify_token(token: str):
    """
    Verifies a JWT token manually.
    Returns decoded username if valid, error otherwise.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        exp: int = payload.get("exp")

        if username is None or exp is None:
            raise HTTPException(status_code=400, detail="Invalid token structure")

        # Check expiration
        if datetime.fromtimestamp(exp, tz=timezone.utc) < datetime.now(timezone.utc):
            raise HTTPException(status_code=401, detail="Token expired")

        return {
            "valid": True,
            "username": username,
            "expires_at": datetime.fromtimestamp(exp, tz=timezone.utc).isoformat()
        }

    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")


@app.get("/")
def root():
    return {"message": "Rohan - AI (Supabase + Neo4j)"}

# ------------------ LIFECYCLE ------------------
@app.on_event("shutdown")
def shutdown_event():
    try:
        driver.close()
    except Exception:
        pass

# ------------------ RUN ------------------
if __name__ == "__main__":
    uvicorn.run("app_jwt2:app", host="127.0.0.1", port=8000, reload=True)
