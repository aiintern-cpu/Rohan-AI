from fastapi import FastAPI, HTTPException, Query
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

# ------------------ CONFIG ------------------
load_dotenv()

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
# print("Neo4j driver created:", driver)
# print("Neo4j URI:", NEO4J_URI)
# with driver.session(database=NEO4J_DB) as session:
#     res = session.run("RETURN 1 AS test")
#     print("Neo4j test query result:", res.single())
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
        return "(Error generating response)"

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

    total = supabase.table("conversation_history_2").select("id", count="exact").eq("sender", req.username).execute()
    # and total.count % 3 == 0
    if total.count  :
        asyncio.create_task(enrich_profile(req.username))

    return {"reply": reply, "category": category}

@app.get("/history")
def history(username: str, offset: int = Query(0, ge=0), limit: int = Query(20, gt=0)):
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
    uvicorn.run("app_neo4j:app", host="127.0.0.1", port=8000, reload=True)




















# # app.py  (Upgraded: Supabase + Neo4j for user_profiles)
# from fastapi import FastAPI, HTTPException, Query
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field
# from typing import List, Optional
# from dotenv import load_dotenv
# from supabase import create_client
# from google import genai
# from neo4j import GraphDatabase
# import os
# import json
# import re
# import uvicorn
# import asyncio

# # ------------------ CONFIG ------------------
# load_dotenv()

# # Supabase setup
# SUPABASE_URL = os.getenv("SUPABASE_URL")
# SUPABASE_KEY = os.getenv("SUPABASE_KEY")
# supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# # Gemini setup (same as before)
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# client = genai.Client(api_key=GEMINI_API_KEY) if genai and GEMINI_API_KEY else None

# # Neo4j setup 
# NEO4J_URI = os.getenv("NEO4J_URI")
# NEO4J_USER = os.getenv("NEO4J_USER")
# NEO4J_PASS = os.getenv("NEO4J_PASS")
# NEO4J_DB = os.getenv("NEO4J_DB", "rohantest")

# driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# # Gemini usage log file
# LOG_FILE = "gemini_usage_log.jsonl"

# # ------------------ FASTAPI SETUP ------------------
# app = FastAPI(title="Rohan - AI Companion (Supabase + Neo4j)")
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ------------------ MODELS ------------------
# class SignupRequest(BaseModel):
#     username: str
#     email: str
#     password: str
#     nickname: str
#     age: int
#     designation: str
#     location: str
#     likes: List[str] = Field(default_factory=list)

# class SigninRequest(BaseModel):
#     username: str
#     password: str

# class ChatRequest(BaseModel):
#     username: str
#     message: Optional[str] = ""

# # ------------------ UTILITIES ------------------
# def generate(prompt, model="gemini-2.5-flash"):
#     """
#     Generate text using Gemini API with logging of usage.
#     Returns the generated text.
#     """
#     if not client:
#         return "(Gemini not configured)"
#     try:
#         res = client.models.generate_content(model=model, contents=prompt)
#         # Log usage
#         usage = res.usage_metadata
#         prompt_tokens = usage.prompt_token_count
#         response_tokens = usage.candidates_token_count
#         total_tokens = usage.total_token_count
#         log_entry = {
#             "response": res.text.strip()[:50],
#             "model": model,
#             "prompt_tokens": prompt_tokens,
#             "response_tokens": response_tokens,
#             "total_tokens": total_tokens
#         }
#         with open(LOG_FILE, "a") as f:
#             f.write(json.dumps(log_entry) + "\n")
#         return res.text.strip()
#     except Exception as e:
#         return f"(Error: {str(e)})"

# def clean_json_response(raw_text: str):
#     """
#     Cleans the raw text response from Gemini to extract valid JSON.
#     Removes markdown, extraneous text, and isolates the JSON object.
#     """
#     cleaned = re.sub(r"^(`{3,}|'{3,})\s*json\s*", "", raw_text.strip(), flags=re.IGNORECASE)
#     cleaned = re.sub(r"(`{3,}|'{3,})$", "", cleaned.strip())
#     match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
#     if match:
#         cleaned = match.group(0)
#     return cleaned

# # # ------------------ DATABASE HELPERS (Neo4j) ------------------
# # def create_user_profile_tx(tx, user_data):
# #     """
# #     Creates a User node and basic relationships for likes/dislikes/events/people.
# #     Expects user_data keys: username, nickname, age, designation, location, likes (list)
# #     """
# #     query = """
# #     MERGE (u:User {username: $username})
# #     SET u.nickname = $nickname,
# #         u.age = $age,
# #         u.designation = $designation,
# #         u.location = $location

# #     WITH u
# #     UNWIND $likes AS like
# #       MERGE (i:Interest {name: like, type: ''})
# #       MERGE (u)-[:LIKES]->(i)
# #     """
# #     tx.run(query, **user_data)

# # def get_user_profile_tx(tx, username):
# #     """
# #     Return a consistent dict-like structure for the profile.
# #     (Excludes minor_events by design.)
# #     """
# #     query = """
# #     MATCH (u:User {username: $username})
# #     OPTIONAL MATCH (u)-[:LIKES]->(l:Interest)
# #     OPTIONAL MATCH (u)-[:DISLIKES]->(d:Interest)
# #     OPTIONAL MATCH (u)-[:HAS_EVENT]->(e:Event)
# #     OPTIONAL MATCH (u)-[:KNOWS]->(p:Person)
# #     RETURN u.username AS username,
# #            u.nickname AS nickname,
# #            u.age AS age,
# #            u.designation AS designation,
# #            u.location AS location,
# #            collect(DISTINCT {l.name}) AS likes,
# #            collect(DISTINCT {d.name}) AS dislikes,
# #            collect(DISTINCT {event_name: e.name, description: e.discription}) AS major_events,
# #            collect(DISTINCT {name: p.name, relationship: p.relationship}) AS people
# #     """
# #     result = tx.run(query, username=username)
# #     rec = result.single()
# #     return rec

# # def update_user_profile_tx(tx, username, update_payload):
# #     """
# #     Merge new items into the graph for the given username.
# #     update_payload may contain keys: nickname, age, designation, location,
# #     likes (list), dislikes (list), major_events (list of {event_name,date}), people (list of {name,relationship})
# #     """
# #     # Update scalar properties first
# #     set_parts = []
# #     params = {"username": username}
# #     if "nickname" in update_payload:
# #         set_parts.append("u.nickname = $nickname")
# #         params["nickname"] = update_payload["nickname"]
# #     if "age" in update_payload:
# #         set_parts.append("u.age = $age")
# #         params["age"] = update_payload["age"]
# #     if "designation" in update_payload:
# #         set_parts.append("u.designation = $designation")
# #         params["designation"] = update_payload["designation"]
# #     if "location" in update_payload:
# #         set_parts.append("u.location = $location")
# #         params["location"] = update_payload["location"]

# #     set_clause = ""
# #     if set_parts:
# #         set_clause = "SET " + ", ".join(set_parts)

# #     # Build the Cypher with FOREACH merging collections
# #     query = f"""
# #     MATCH (u:User {{username: $username}})
# #     {set_clause}
# #     WITH u
# #     """
# #     # likes
# #     if update_payload.get("likes"):
# #         query += """
# #         FOREACH (like IN $likes |
# #           MERGE (i:Interest {name: like})
# #           MERGE (u)-[:LIKES]->(i)
# #         )
# #         WITH u
# #         """
# #         params["likes"] = update_payload["likes"]
# #     # dislikes
# #     if update_payload.get("dislikes"):
# #         query += """
# #         FOREACH (dislike IN $dislikes |
# #           MERGE (d:Interest {name: dislike})
# #           MERGE (u)-[:DISLIKES]->(d)
# #         )
# #         WITH u
# #         """
# #         params["dislikes"] = update_payload["dislikes"]
# #     # major events (list of dicts with event_name and date)
# #     if update_payload.get("major_events"):
# #         query += """
# #         FOREACH (ev IN $major_events |
# #           MERGE (e:Event {name: ev.event_name})
# #           SET e.date = ev.date
# #           MERGE (u)-[:HAS_EVENT]->(e)
# #         )
# #         WITH u
# #         """
# #         params["major_events"] = update_payload["major_events"]
# #     # people (list of dicts {name, relationship})
# #     if update_payload.get("people"):
# #         query += """
# #         FOREACH (person IN $people |
# #           MERGE (p:Person {name: person.name})
# #           SET p.relationship = person.relationship
# #           MERGE (u)-[:KNOWS]->(p)
# #         )
# #         """
# #         params["people"] = update_payload["people"]

# #     tx.run(query, **params)

# # ------------------ NEO4J HELPERS (v2: supports {name,type}, {event_name,description}) ------------------

# def create_user_profile_tx(tx, user_data):
#     """
#     Creates a User node and related nodes based on the unified schema.
#     Likes and dislikes are list of dicts: {name, type}
#     Major events: {event_name, description}
#     People: ["Name (relationship)"]
#     """
#     query = """
#     MERGE (u:User {username: $username})
#     SET u.nickname = $nickname,
#         u.age = $age,
#         u.designation = $designation,
#         u.location = $location

#     WITH u
#     UNWIND $likes AS like
#       MERGE (i:Interest {name: like.name})
#       SET i.type = like.type
#       MERGE (u)-[:LIKES]->(i)
    
#     WITH u
#     UNWIND $dislikes AS dislike
#       MERGE (d:Interest {name: dislike.name})
#       SET d.type = dislike.type
#       MERGE (u)-[:DISLIKES]->(d)
#     """
#     tx.run(query, **user_data)

# def get_user_profile_tx(tx, username):
#     query = """
#     MATCH (u:User {username: $username})
#     OPTIONAL MATCH (u)-[:LIKES]->(l:Interest)
#     OPTIONAL MATCH (u)-[:DISLIKES]->(d:Interest)
#     OPTIONAL MATCH (u)-[:HAS_EVENT]->(e:Event)
#     OPTIONAL MATCH (u)-[:KNOWS]->(p:Person)
#     RETURN u.username AS username,
#            u.nickname AS nickname,
#            u.age AS age,
#            u.designation AS designation,
#            u.location AS location,
#            collect(DISTINCT {name: l.name, type: l.type}) AS likes,
#            collect(DISTINCT {name: d.name, type: d.type}) AS dislikes,
#            collect(DISTINCT {event_name: e.name, description: e.description}) AS major_events,
#            collect(DISTINCT {name: p.name, relationship: p.relationship}) AS people
#     """
#     return tx.run(query, username=username).single()

# def update_user_profile_tx(tx, username, update_payload):
#     """
#     Update Neo4j User node + related nodes using the new format.
#     """
#     set_parts = []
#     params = {"username": username}

#     # Update scalar properties
#     for key in ["nickname", "age", "designation", "location"]:
#         if key in update_payload:
#             set_parts.append(f"u.{key} = ${key}")
#             params[key] = update_payload[key]

#     set_clause = "SET " + ", ".join(set_parts) if set_parts else ""

#     query = f"""
#     MATCH (u:User {{username: $username}})
#     {set_clause}
#     WITH u
#     """

#     # Likes (list of dicts)
#     if update_payload.get("likes"):
#         query += """
#         FOREACH (like IN $likes |
#             MERGE (i:Interest {name: like.name})
#             SET i.type = like.type
#             MERGE (u)-[:LIKES]->(i)
#         )
#         WITH u
#         """
#         params["likes"] = update_payload["likes"]

#     # Dislikes (list of dicts)
#     if update_payload.get("dislikes"):
#         query += """
#         FOREACH (dislike IN $dislikes |
#             MERGE (d:Interest {name: dislike.name})
#             SET d.type = dislike.type
#             MERGE (u)-[:DISLIKES]->(d)
#         )
#         WITH u
#         """
#         params["dislikes"] = update_payload["dislikes"]

#     # Major Events
#     if update_payload.get("major_events"):
#         query += """
#         FOREACH (ev IN $major_events |
#             MERGE (e:Event {name: ev.event_name})
#             SET e.description = ev.description
#             MERGE (u)-[:HAS_EVENT]->(e)
#         )
#         WITH u
#         """
#         params["major_events"] = update_payload["major_events"]

#     # People
#     if update_payload.get("people"):
#         query += """
#         FOREACH (person IN $people |
#             MERGE (p:Person {name: person.name})
#             SET p.relationship = person.relationship
#             MERGE (u)-[:KNOWS]->(p)
#         )
#         """
#         params["people"] = update_payload["people"]

#     tx.run(query, **params)

# def get_profile(username: str):
#     with driver.session(database=NEO4J_DB) as session:
#         rec = session.execute_read(get_user_profile_tx, username)
#         if not rec:
#             return None
#         return {
#             "username": rec["username"],
#             "nickname": rec["nickname"] or rec["username"],
#             "age": rec["age"] or 0,
#             "designation": rec["designation"] or "",
#             "location": rec["location"] or "",
#             "likes": rec["likes"] or [],
#             "dislikes": rec["dislikes"] or [],
#             "major_events": rec["major_events"] or [],
#             "people": rec["people"] or []
#         }

# def create_profile_in_neo4j(user_dict):
#     """
#     During signup: likes may be plain strings; convert to {name,type:""}
#     """
#     likes = [{"name": l, "type": ""} for l in user_dict.get("likes", [])]
#     dislikes = [{"name": d, "type": ""} for d in user_dict.get("dislikes", [])]
#     payload = {
#         "username": user_dict["username"],
#         "nickname": user_dict["nickname"],
#         "age": user_dict["age"],
#         "designation": user_dict["designation"],
#         "location": user_dict["location"],
#         "likes": likes,
#         "dislikes": dislikes
#     }
#     with driver.session(database=NEO4J_DB) as session:
#         session.execute_write(create_user_profile_tx, payload)


# # ------------------ DATABASE HELPERS (Supabase) ------------------
# def get_user(username: str):
#     res = supabase.table("auth_users_2").select("*").eq("username", username).execute()
#     if not res.data:
#         return None
#     return res.data[0]

# def get_rohan_profile():
#     # Keep Rohan's profile in Supabase as requested
#     res = supabase.table("user_profiles").select("*").eq("username", "Rohan").execute()
#     if not res.data:
#         return None
#     return res.data[0]

# def get_recent_history(username: str, limit: int = 10):
#     res = supabase.table("conversation_history_2").select("*") \
#         .or_(f"sender.eq.{username},receiver.eq.{username}") \
#         .order("timestamp", desc=True).limit(limit).execute()
#     recent_conversations = list(reversed(res.data))
#     return recent_conversations if res.data else []

# def add_conversation(sender, receiver, message, category):
#     supabase.table("conversation_history").insert({
#         "sender": sender,
#         "receiver": receiver,
#         "message": message,
#         "category": category
#     }).execute()

# # ------------------ PROFILE ACCESS/CRUD (Neo4j-backed) ------------------
# def get_profile(username: str):
#     """
#     Return user profile from Neo4j as a dict.
#     """
#     try:
#         with driver.session(database=NEO4J_DB) as session:
#             rec = session.execute_read(get_user_profile_tx, username)
#             if not rec:
#                 return None
#             # Convert record to dict with defaults to match previous shape
#             profile = {
#                 "username": rec["username"],
#                 "nickname": rec["nickname"] if rec["nickname"] is not None else rec["username"],
#                 "age": rec["age"] if rec["age"] is not None else 0,
#                 "designation": rec["designation"] if rec["designation"] is not None else "",
#                 "location": rec["location"] if rec["location"] is not None else "",
#                 "likes": rec["likes"] if rec["likes"] is not None else [],
#                 "dislikes": rec["dislikes"] if rec["dislikes"] is not None else [],
#                 "major_events": rec["major_events"] if rec["major_events"] is not None else [],
#                 "people": rec["people"] if rec["people"] is not None else []
#             }
#             # Some Neo4j drivers serialize nested maps as neo4j.types.Map which behaves like dict
#             # Ensure lists of maps are plain python lists/dicts
#             def normalize_list_of_maps(lst):
#                 if not lst:
#                     return []
#                 normalized = []
#                 for item in lst:
#                     if hasattr(item, "items"):
#                         normalized.append(dict(item))
#                     else:
#                         normalized.append(item)
#                 return normalized

#             profile["major_events"] = normalize_list_of_maps(profile["major_events"])
#             profile["people"] = normalize_list_of_maps(profile["people"])
#             return profile
#     except Exception as e:
#         print("Error fetching profile from Neo4j:", e)
#         return None

# def create_profile_in_neo4j(user_dict):
#     """
#     user_dict should minimally contain:
#     username, nickname, age, designation, location, likes (list)
#     """
#     try:
#         with driver.session(database=NEO4J_DB) as session:
#             session.execute_write(create_user_profile_tx, user_dict)
#     except Exception as e:
#         print("Error creating profile in Neo4j:", e)
#         raise

# def update_profile_in_neo4j(username, update_payload):
#     try:
#         with driver.session(database=NEO4J_DB) as session:
#             session.execute_write(update_user_profile_tx, username, update_payload)
#     except Exception as e:
#         print("Error updating profile in Neo4j:", e)

# async def enrich_profile(username: str):
#     try:
#         res = supabase.table("conversation_history").select("*") \
#             .eq("sender", username).order("timestamp", desc=True).limit(15).execute()
#         convos = list(reversed(res.data))
#         text = "\n".join([c["message"] for c in convos if "message" in c])
#         if not text:
#             return

#         extract_prompt = f"""
# You are an AI extracting personal information from user conversations.
# Return ONLY valid JSON:
# {{
#     "nickname": "string or null",
#     "age": int or null,
#     "designation": "string or null",
#     "location": "string or null",
#     "likes": [{{"name": "item", "type": "hobby/food/media/etc."}}],
#     "dislikes": [{{"name": "item", "type": "hobby/food/media/etc."}}],
#     "major_events": [{{"event_name": "string", "description": "string"}}],
#     "people": ["Name (relationship)"]
# }}
# Rules:
# - Include only if explicitly mentioned.
# - If not available, set value to null or [].
# Conversation:
# {text}
# """
#         raw = generate(extract_prompt, "gemini-2.5-pro")
#         cleaned = clean_json_response(raw)
#         extracted = json.loads(cleaned)

#         existing = get_profile(username)
#         if not existing:
#             print(f"No Neo4j profile found for {username}")
#             return

#         payload = {}

#         # Scalars
#         for key in ["nickname", "age", "designation", "location"]:
#             val = extracted.get(key)
#             if val not in [None, "", 0]:
#                 if val != existing.get(key):
#                     payload[key] = val

#         # Likes/dislikes as list of dicts
#         def merge_interests(old, new):
#             if not isinstance(old, list): old = []
#             if not isinstance(new, list): new = []
#             old_names = {i["name"] for i in old if isinstance(i, dict) and i.get("name")}
#             merged = old[:]
#             for i in new:
#                 if isinstance(i, dict) and i.get("name") not in old_names:
#                     merged.append({"name": i["name"], "type": i.get("type", "")})
#             return merged

#         if extracted.get("likes"):
#             payload["likes"] = merge_interests(existing.get("likes", []), extracted["likes"])
#         if extracted.get("dislikes"):
#             payload["dislikes"] = merge_interests(existing.get("dislikes", []), extracted["dislikes"])

#         # Major events
#         if extracted.get("major_events"):
#             old_events = existing.get("major_events", [])
#             old_names = {e["event_name"] for e in old_events if isinstance(e, dict) and e.get("event_name")}
#             new_unique = [e for e in extracted["major_events"] if e.get("event_name") not in old_names]
#             if new_unique:
#                 payload["major_events"] = old_events + new_unique

#         # People
#         if extracted.get("people"):
#             valid_people = []
#             for p in extracted["people"]:
#                 if isinstance(p, str):
#                     m = re.match(r"^(.*)\s*\((.*)\)\s*$", p)
#                     if m:
#                         valid_people.append({"name": m.group(1).strip(), "relationship": m.group(2).strip()})
#                     else:
#                         valid_people.append({"name": p.strip(), "relationship": ""})
#             if valid_people:
#                 old_people = existing.get("people", [])
#                 old_names = {p.get("name") for p in old_people}
#                 new_unique = [p for p in valid_people if p["name"] not in old_names]
#                 if new_unique:
#                     payload["people"] = old_people + new_unique

#         if payload:
#             update_profile_in_neo4j(username, payload)
#             print(f"✅ Updated Neo4j profile for {username} with:", payload)
#         else:
#             print(f"ℹ️ No updates for {username}")

#     except Exception as e:
#         print("❌ enrich_profile error:", e)


# # # ------------------ ENRICH PROFILE (extract + update Neo4j) ------------------
# # async def enrich_profile(username: str):
# #     """
# #     Extract likes/dislikes/major_events/people and also nickname, age, designation, location
# #     from recent conversations using Gemini; then update Neo4j profile.
# #     (This function is called asynchronously from /chat logic when needed.)
# #     """
# #     try:
# #         # Fetch recent user messages (only user's messages to avoid bot text noise)
# #         res = supabase.table("conversation_history").select("*") \
# #             .eq("sender", username) \
# #             .order("timestamp", desc=True).limit(15).execute()
# #         convos = list(reversed(res.data))
# #         text = "\n".join([c["message"] for c in convos if "message" in c])
# #         if not text:
# #             return

# #         extract_prompt = f"""You are an AI that analyzes conversations between a user and their AI companion to extract important personal information.

# #     Task: Analyze the conversation log and extract a structured summary of key personal details.

# #     What to Extract:

# #     1. likes: Things the user enjoys or loves
# #     - Hobbies and activities (e.g., "painting", "hiking")
# #     - Foods and restaurants (e.g., "biryani", "Truffles")
# #     - Media (e.g., "Inception", "The Beatles", "Stranger Things")
# #     - Brands, places, or anything they express positive sentiment about

# #     2. dislikes: Things the user does not enjoy or loves
# #     - Hobbies and activities (e.g., "painting", "hiking")
# #     - Foods and restaurants (e.g., "biryani", "Truffles")
# #     - Media (e.g., "Inception", "The Beatles", "Stranger Things")
# #     - Brands, places, or anything they express negative sentiment about

# #     3. major_ events: Significant life events or milestones mentioned
# #     - Career changes (e.g., "started new job at Google")
# #     - Life transitions (e.g., "moved to Bangalore", "graduated college")
# #     - Important occasions (e.g., "sister's wedding", "promotion")
# #     - Only include specific events, not general statements

# #     4. minor_events: Smaller but notable events or experiences
# #     - Daily life events (e.g., "went to a concert", "tried a new restaurant")
# #     - Travel experiences (e.g., "visited Goa", "road trip to Mysore")
# #     - Social activities (e.g., "hung out with friends", "family dinner")
# #     - Only include specific events, not general statements

# #     5. people: Names and relationships of people in their life
# #     - Format: "Name (relationship)" (e.g., "Priya (sister)", "Alex (colleague)")
# #     - Include family, friends, colleagues, partners
# #     - Only include if both name AND relationship are mentioned

# #     Extraction Rules:
# #     - Only extract information explicitly stated in the conversation
# #     - Don't infer or assume information not directly mentioned
# #     - Use exact quotes or paraphrasing from the conversation
# #     - If a category has no clear information, use an empty array []
# #     - Maintain consistent formatting across all entries
# #     - Remove duplicates

# #     Output Format:
# #     Return ONLY valid JSON with no additional text, explanations, or markdown:

# #     Conversation Log:
# #     {text}

# #     Output JSON Format:
# #     {{
# #     "nickname": "string or null",
# #     "age": int or null,
# #     "designation": "string or null",
# #     "location": "string or null",
# #     "likes": [
# #         {"name": "item1", "type": "hobby/food/media/brand/etc."}, {"name": "item2", "type": "hobby/food/media/brand/etc."}
# #     ],
# #     "dislikes": ": "string or null",
# #     "likes": [
# #         {"name": "item1", "type": "hobby/food/media/brand/etc."}, {"name": "item2", "type": "hobby/food/media/brand/etc."}
# #     ],
# #     "major_events": [{"event_name": "string", "description": "string - short description of the event"}, {"event_name": "string", "description": "string"}],
# #     "people": ["Name (relationship)", "Name (relationship)"]
# #     }}


# #     Extract and return JSON
# #     """

# #         raw = generate(extract_prompt, "gemini-2.5-pro")
# #         cleaned = clean_json_response(raw)
# #         try:
# #             extracted = json.loads(cleaned)
# #         except Exception as ex:
# #             print("Failed to parse enrichment JSON:", ex)
# #             return

# #         # Normalize/prepare payload for Neo4j
# #         payload = {}
# #         # Scalars
# #         for key in ["nickname", "age", "designation", "location"]:
# #             if key in extracted and extracted[key] not in [None, "", []]:
# #                 payload[key] = extracted[key]
# #         # Arrays
# #         if extracted.get("likes"):
# #             payload["likes"] = extracted["likes"]
# #         if extracted.get("dislikes"):
# #             payload["dislikes"] = extracted["dislikes"]
# #         # major_events expected as list of dicts with event_name and date
# #         if extracted.get("major_events"):
# #             # ensure consistent key names
# #             me = []
# #             for item in extracted["major_events"]:
# #                 if isinstance(item, dict):
# #                     name = item.get("event_name") or item.get("name") or item.get("event")
# #                     date = item.get("date") or ""
# #                     me.append({"event_name": name, "date": date})
# #             if me:
# #                 payload["major_events"] = me
# #         # people expected list of "Name (relationship)" or dicts
# #         if extracted.get("people"):
# #             ppl = []
# #             for p in extracted["people"]:
# #                 if isinstance(p, dict):
# #                     # store as {name, relationship}
# #                     name = p.get("name")
# #                     rel = p.get("relationship", "")
# #                     if name:
# #                         ppl.append({"name": name, "relationship": rel})
# #                 elif isinstance(p, str):
# #                     # try to parse "Name (relationship)"
# #                     m = re.match(r"^(.*)\s*\((.*)\)\s*$", p)
# #                     if m:
# #                         ppl.append({"name": m.group(1).strip(), "relationship": m.group(2).strip()})
# #                     else:
# #                         ppl.append({"name": p, "relationship": ""})
# #             if ppl:
# #                 payload["people"] = ppl

# #         if payload:
# #             update_profile_in_neo4j(username, payload)

# #     except Exception as e:
# #         print("Error in enrich_profile:", e)

# # async def enrich_profile(username: str):
# #     """
# #     Improved enrichment:
# #     - Extract nickname, age, designation, location, likes, dislikes, major_events, people
# #     - Update Neo4j only for fields that are newly extracted and valid
# #     - Avoid overwriting existing properties with null or empty values
# #     """
# #     try:
# #         # Fetch last 15 user messages
# #         res = supabase.table("conversation_history").select("*") \
# #             .eq("sender", username) \
# #             .order("timestamp", desc=True).limit(15).execute()
# #         convos = list(reversed(res.data))
# #         text = "\n".join([c["message"] for c in convos if "message" in c])
# #         if not text:
# #             return

# #         extract_prompt = f"""
# # You are an AI that analyzes user chat logs to extract personal info.

# # Return ONLY valid JSON:
# # {{
# # "nickname": "string or null",
# # "age": int or null,
# # "designation": "string or null",
# # "location": "string or null",
# # "likes": ["string"],
# # "dislikes": ["string"],
# # "major_events": [{{"event_name": "string", "date": "YYYY-MM-DD"}}],
# # "people": [{{"name": "string", "relationship": "string"}}]
# # }}

# # Rules:
# # - Only include fields explicitly stated.
# # - Omit or use null for unknown fields.
# # - Never guess or infer missing details.
# # - No text outside the JSON.

# # Conversation:
# # {text}
# # """
# #         raw = generate(extract_prompt, "gemini-2.5-pro")
# #         cleaned = clean_json_response(raw)
# #         try:
# #             extracted = json.loads(cleaned)
# #         except Exception as ex:
# #             print("⚠️ Failed to parse enrichment JSON:", ex, "\nRaw:", raw)
# #             return

# #         # Get existing Neo4j profile to compare current properties
# #         existing = get_profile(username)
# #         if not existing:
# #             print(f"⚠️ No existing Neo4j profile found for {username}")
# #             return

# #         # Prepare payload (only new or improved fields)
# #         payload = {}

# #         # USER NODE PROPERTIES
# #         scalar_fields = ["nickname", "age", "designation", "location"]
# #         for key in scalar_fields:
# #             new_val = extracted.get(key)
# #             old_val = existing.get(key)
# #             # Update only if new_val is meaningful and different
# #             if new_val not in [None, "", [], 0] and new_val != old_val:
# #                 payload[key] = new_val

# #         # LIKES / DISLIKES / RELATIONSHIPS / EVENTS
# #         def merge_unique_list(old_list, new_list):
# #             if not isinstance(old_list, list): old_list = []
# #             if not isinstance(new_list, list): new_list = []
# #             merged = list({x for x in (old_list + new_list) if x})
# #             return merged

# #         # Likes and dislikes
# #         if "likes" in extracted and isinstance(extracted["likes"], list):
# #             payload["likes"] = merge_unique_list(existing.get("likes", []), extracted["likes"])
# #         if "dislikes" in extracted and isinstance(extracted["dislikes"], list):
# #             payload["dislikes"] = merge_unique_list(existing.get("dislikes", []), extracted["dislikes"])

# #         # Major events
# #         if "major_events" in extracted and isinstance(extracted["major_events"], list):
# #             valid_events = []
# #             for ev in extracted["major_events"]:
# #                 if isinstance(ev, dict) and ev.get("event_name"):
# #                     valid_events.append({"event_name": ev["event_name"], "date": ev.get("date", "")})
# #             if valid_events:
# #                 old_events = existing.get("major_events", [])
# #                 old_names = {e.get("event_name") for e in old_events if isinstance(e, dict)}
# #                 new_unique = [e for e in valid_events if e["event_name"] not in old_names]
# #                 if new_unique:
# #                     payload["major_events"] = old_events + new_unique

# #         # People
# #         if "people" in extracted and isinstance(extracted["people"], list):
# #             valid_people = []
# #             for p in extracted["people"]:
# #                 if isinstance(p, dict) and p.get("name"):
# #                     valid_people.append({"name": p["name"], "relationship": p.get("relationship", "")})
# #                 elif isinstance(p, str):
# #                     m = re.match(r"^(.*)\s*\((.*)\)\s*$", p)
# #                     if m:
# #                         valid_people.append({"name": m.group(1).strip(), "relationship": m.group(2).strip()})
# #                     else:
# #                         valid_people.append({"name": p.strip(), "relationship": ""})
# #             if valid_people:
# #                 old_people = existing.get("people", [])
# #                 old_names = {p.get("name") for p in old_people if isinstance(p, dict)}
# #                 new_unique = [p for p in valid_people if p["name"] not in old_names]
# #                 if new_unique:
# #                     payload["people"] = old_people + new_unique

# #         # ✅ Update only if there's something new
# #         if payload:
# #             update_profile_in_neo4j(username, payload)
# #             print(f"✅ Profile of '{username}' updated with:", payload)
# #         else:
# #             print(f"ℹ️ No new profile data to update for {username}")

# #     except Exception as e:
# #         print("❌ Error in enrich_profile:", e)


# # ------------------ MESSAGE CLASSIFICATION & RESPONSE (unchanged logic, uses get_profile) ------------------
# def classify_message(user_input, profile, rohan_profile):
#     prompt = f"""You are analyzing user intent for a companion chatbot. Classify the message into exactly ONE category.

# Categories :
# - Suggestive
# - Discussive
# - Humorous

# User Profile :
# Nickname: {profile['nickname']}
# Designation: {profile['designation']}
# Likes: {', '.join(profile.get('likes', []))}
# Dislikes: {', '.join(profile.get('dislikes', []))}

# Bot Profile :
# Nickname: {rohan_profile['nickname']}
# Likes: {', '.join(rohan_profile.get('likes', []))}

# <Recent Conversation Context>
# {get_recent_history(profile['nickname'], 4)}
# </Recent Conversation Context>

# User message : "{user_input}"

# Return ONLY one word: Suggestive, Discussive, or Humorous
# """
#     raw = generate(prompt)
#     for label in ["suggestive", "discussive", "humorous"]:
#         if label in (raw or "").lower():
#             return label
#     return "discussive"

# def handle_suggestive(user_input, profile, rohan_profile):
#     prompt = f"""You are Rohan. Your Personality is ENTP type with casual GenZ indian slang.
# You are a supportive and knowledgeable friend. The user is seeking advice, recommendations, or suggestions.
# User Profile:
# Nickname: {profile['nickname']}
# Age: {profile['age']}
# Location: {profile['location']}
# Likes: {', '.join(profile.get('likes', []))}
# User's Request: {user_input}
# Provide helpful, personalized advice."""
#     return generate(prompt)

# def handle_discussive(user_input, profile, rohan_profile):
#     prompt = f"""You are Rohan. Thoughtful, empathetic friend.
# User Profile:
# Nickname: {profile['nickname']}
# Age: {profile['age']}
# Location: {profile['location']}
# Conversation History: {get_recent_history(profile['nickname'], 6)}
# User's Message: {user_input}
# Respond naturally and empathetically."""
#     return generate(prompt)

# def handle_humorous(user_input, profile, rohan_profile):
#     prompt = f"""You are Rohan. Witty, fun-loving Bangalore Gen Z vibe.
# User's Message: {user_input}
# Make a short light-hearted reply."""
#     return generate(prompt)

# def chatbot_reply(user_input, username):
#     profile = get_profile(username)
#     if not profile:
#         raise HTTPException(status_code=404, detail="User profile not found")
#     rohan_profile = get_rohan_profile()
#     if not rohan_profile:
#         # fallback minimal profile
#         rohan_profile = {"nickname": "Rohan", "likes": []}

#     category = classify_message(user_input, profile, rohan_profile)

#     if category == "suggestive":
#         reply = handle_suggestive(user_input, profile, rohan_profile)
#     elif category == "discussive":
#         reply = handle_discussive(user_input, profile, rohan_profile)
#     elif category == "humorous":
#         reply = handle_humorous(user_input, profile, rohan_profile)
#     else:
#         reply = "I'm here for you 😊 tell me more"

#     # Save both user message and bot reply to Supabase conversation_history
#     add_conversation(username, "Rohan", user_input, "user input")
#     add_conversation("Rohan", username, reply, category)

#     # Enrich profile every 15 messages (async trigger)
#     total = supabase.table("conversation_history").select("id", count="exact").eq("sender", username).execute()
#     if total.count and total.count % 3 == 0:
#         print(f"Enriching profile for user: {username}")
#         # schedule enrichment (non-blocking)
#         asyncio.create_task(enrich_profile(username))

#     return reply, category

# # ------------------ ROUTES ------------------
# @app.post("/signup")
# def signup(req: SignupRequest):
#     # Check if user exists in auth_users (Supabase)
#     existing = supabase.table("auth_users").select("*").eq("username", req.username).execute()
#     if existing.data:
#         raise HTTPException(status_code=400, detail="Username already exists")

#     # Insert into auth_users (Supabase)
#     supabase.table("auth_users").insert({
#         "username": req.username,
#         "email": req.email,
#         "password": req.password
#     }).execute()

#     # Create profile in Neo4j (user_profiles moved to Neo4j)
#     user_data = {
#         "username": req.username,
#         "nickname": req.nickname,
#         "age": req.age,
#         "designation": req.designation,
#         "location": req.location,
#         "likes": req.likes or []
#     }
#     try:
#         create_profile_in_neo4j(user_data)
#     except Exception as e:
#         # rollback supabase auth insertion if Neo4j fails would be ideal in prod,
#         # for now report error
#         raise HTTPException(status_code=500, detail=f"Failed to create user profile: {e}")

#     return {"message": f"User {req.username} registered successfully"}

# @app.post("/signin")
# def signin(req: SigninRequest):
#     user = get_user(req.username)
#     if not user:
#         raise HTTPException(status_code=404, detail="User not found")
#     if user["password"] != req.password:
#         raise HTTPException(status_code=401, detail="Invalid password")
#     return {"success": True, "message": f"Welcome back {req.username}!"}

# @app.post("/chat")
# async def chat(req: ChatRequest):
#     if not req.username:
#         raise HTTPException(status_code=400, detail="Username required")
#     reply, category = chatbot_reply(req.message, req.username)
#     return {"reply": reply, "category": category}

# @app.get("/history")
# def history(username: str, offset: int = Query(0, ge=0), limit: int = Query(20, gt=0)):
#     try:
#         res = supabase.table("conversation_history").select("*") \
#             .or_(f"sender.eq.{username},receiver.eq.{username}") \
#             .order("timestamp", desc=True) \
#             .range(offset, offset + limit - 1) \
#             .execute()
#         recent_conversations = list(reversed(res.data))
#         conversations = recent_conversations if res.data else []
#         return {
#             "total": len(conversations),
#             "offset": offset,
#             "limit": limit,
#             "conversations": conversations
#         }
#     except Exception as e:
#         print("!!! ERROR in /history endpoint:", e)
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/")
# def root():
#     return {"message": "Rohan - AI (Supabase + Neo4j)"}

# # ------------------ LIFECYCLE / CLEANUP ------------------
# @app.on_event("shutdown")
# def shutdown_event():
#     try:
#         driver.close()
#         print("Neo4j driver closed.")
#     except Exception:
#         pass

# # ------------------ RUN ------------------
# if __name__ == "__main__":
#     uvicorn.run("app_neo4j:app", host="127.0.0.1", port=8000, reload=True)

# def update_user_profile_tx(tx, username: str, update_payload: Dict[str, Any]):
#     set_parts = []
#     params = {"username": username}
#     for key in ["nickname", "age", "designation", "location"]:
#         if key in update_payload:
#             set_parts.append(f"u.{key} = ${key}")
#             params[key] = update_payload[key]
#     set_clause = "SET " + ", ".join(set_parts) if set_parts else ""
#     query = f"""
#     MATCH (u:User {{username: $username}})
#     {set_clause}
#     WITH u
#     """

#     if update_payload.get("likes"):
#         query += """
#         FOREACH (like IN $likes |
#             MERGE (i:Interest {name: like.name})
#             SET i.type = like.type
#             MERGE (u)-[:LIKES]->(i)
#         )
#         WITH u
#         """
#         params["likes"] = update_payload["likes"]
#     if update_payload.get("dislikes"):
#         query += """
#         FOREACH (dislike IN $dislikes |
#             MERGE (d:Interest {name: dislike.name})
#             SET d.type = dislike.type
#             MERGE (u)-[:DISLIKES]->(d)
#         )
#         WITH u
#         """
#         params["dislikes"] = update_payload["dislikes"]
#     if update_payload.get("major_events"):
#         query += """
#         FOREACH (ev IN $major_events |
#             MERGE (e:Event {name: ev.event_name})
#             SET e.description = ev.description
#             MERGE (u)-[:HAS_EVENT]->(e)
#         )
#         WITH u
#         """
#         params["major_events"] = update_payload["major_events"]
#     if update_payload.get("people"):
#         query += """
#         FOREACH (person IN $people |
#             MERGE (p:Person {name: person.name})
#             SET p.relationship = person.relationship
#             MERGE (u)-[:KNOWS]->(p)
#         )
#         """
#         params["people"] = update_payload["people"]
#     print("Update query in update_user_profile_tx:", query)
#     tx.run(query, **params)
#     print("Profile updated in Neo4j for user:", username)

# def update_user_profile_tx(tx, username: str, update_payload: Dict[str, Any]):
#     """
#     Safely updates Neo4j User node and connected nodes.
#     Compatible with Gemini enrichment JSON format.
#     """
#     set_parts = []
#     params = {"username": username}

#     # --- Scalar properties (User node) ---
#     for key in ["nickname", "age", "designation", "location"]:
#         if key in update_payload and update_payload[key] not in [None, ""]:
#             set_parts.append(f"u.{key} = ${key}")
#             params[key] = update_payload[key]

#     set_clause = "SET " + ", ".join(set_parts) if set_parts else ""

#     # --- Start query ---
#     query = f"""
#     MATCH (u:User {{username: $username}})
#     {set_clause}
#     """

#     # --- Likes ---
#     if update_payload.get("likes"):
#         query += """
#         WITH u
#         FOREACH (like IN $likes |
#             MERGE (i:Interest {name: like.name})
#             SET i.type = like.type
#             MERGE (u)-[:LIKES]->(i)
#         )
#         """
#         params["likes"] = update_payload["likes"]

#     # --- Dislikes ---
#     if update_payload.get("dislikes"):
#         query += """
#         WITH u
#         FOREACH (dislike IN $dislikes |
#             MERGE (d:Interest {name: dislike.name})
#             SET d.type = dislike.type
#             MERGE (u)-[:DISLIKES]->(d)
#         )
#         """
#         params["dislikes"] = update_payload["dislikes"]

#     # --- Major Events ---
#     if update_payload.get("major_events"):
#         query += """
#         WITH u
#         FOREACH (ev IN $major_events |
#             MERGE (e:Event {name: ev.event_name})
#             SET e.description = ev.description
#             MERGE (u)-[:HAS_EVENT]->(e)
#         )
#         """
#         params["major_events"] = update_payload["major_events"]

#     # --- People ---
#     if update_payload.get("people"):
#         query += """
#         WITH u
#         FOREACH (person IN $people |
#             MERGE (p:Person {name: person.name})
#             SET p.relationship = person.relationship
#             MERGE (u)-[:KNOWS]->(p)
#         )
#         """
#         params["people"] = update_payload["people"]

#     # Final return for debugging / safety
#     query += "\nRETURN count(u) AS updated;"

#     print("🚀 Running update_user_profile_tx Cypher:")
#     print(query)
#     print("Params:", json.dumps(params, indent=2, default=str))

#     tx.run(query, **params)
#     print(f"✅ Profile updated in Neo4j for {username}")

# async def enrich_profile(username: str):
#     try:
#         res = supabase.table("conversation_history_2").select("*") \
#             .eq("sender", username) \
#             .order("timestamp", desc=True).limit(15).execute()
#         convos = list(reversed(res.data)) if res.data else []
#         text = "\n".join([c["message"] for c in convos if "message" in c])
#         if not text:
#             return

#         extract_prompt = f"""You are an AI that analyzes conversations between a user and their AI companion to extract important personal information.

#         Task: Analyze the conversation log and extract a structured summary of key personal details.

#         What to Extract:

#         1. likes: Things the user enjoys or loves
#         - Hobbies and activities (e.g., "painting", "hiking")
#         - Foods and restaurants (e.g., "biryani", "Truffles")
#         - Media (e.g., "Inception", "The Beatles", "Stranger Things")
#         - Brands, places, or anything they express positive sentiment about

#         2. dislikes: Things the user does not enjoy or loves
#         - Hobbies and activities (e.g., "painting", "hiking")
#         - Foods and restaurants (e.g., "biryani", "Truffles")
#         - Media (e.g., "Inception", "The Beatles", "Stranger Things")
#         - Brands, places, or anything they express negative sentiment about

#         3. major_events: Significant life events or milestones mentioned
#         - Career changes (e.g., "started new job at Google")
#         - Life transitions (e.g., "moved to Bangalore", "graduated college")
#         - Important occasions (e.g., "sister's wedding", "promotion")
#         - Only include specific events, not general statements

#         4. minor_events: Smaller but notable events or experiences
#         - Daily life events (e.g., "went to a concert", "tried a new restaurant")
#         - Travel experiences (e.g., "visited Goa", "road trip to Mysore")
#         - Social activities (e.g., "hung out with friends", "family dinner")
#         - Only include specific events, not general statements

#         5. people: Names and relationships of people in their life
#         - Format: "Name (relationship)" (e.g., "Priya (sister)", "Alex (colleague)")
#         - Include family, friends, colleagues, partners
#         - Only include if both name AND relationship are mentioned

#         Extraction Rules:
#         - Only extract information explicitly stated in the conversation
#         - Don't infer or assume information not directly mentioned
#         - Use exact quotes or paraphrasing from the conversation
#         - If a category has no clear information, use an empty array []
#         - Maintain consistent formatting across all entries
#         - Remove duplicates

#         Output Format:
#         Return ONLY valid JSON with no additional text, explanations, or markdown:

#         Conversation Log:
#         {text}

#         Output JSON Schema:
#         {{
#         "nickname": "string or null",
#         "age": int or null,
#         "designation": "string or null",
#         "location": "string or null",
#         "likes": [{{"name": "item", "type": "hobby/food/media/brand/etc."}}],
#         "dislikes": [{{"name": "item", "type": "hobby/food/media/brand/etc."}}],
#         "major_events": [{{"event_name": "string", "description": "short description"}}],
#         "people": [{{"name": "string", "relationship": "string"}}]
#         }}

#         Extract and return JSON
#         """
#         raw = generate(extract_prompt, "gemini-2.5-pro")
#         cleaned = clean_json_response(raw)
#         print("Enrichment json response:", cleaned)
#         try:
#             extracted = json.loads(cleaned)
#         except Exception:
#             return

#         existing = get_profile(username)
#         if not existing:
#             return

#         payload: Dict[str, Any] = {}

#         # Scalars: update only if new non-empty and different
#         for key in ["nickname", "age", "designation", "location"]:
#             new_val = extracted.get(key, None)
#             if new_val not in [None, "", []]:
#                 old_val = existing.get(key)
#                 if key == "age":
#                     try:
#                         if isinstance(new_val, str) and new_val.isdigit():
#                             new_val = int(new_val)
#                     except Exception:
#                         pass
#                 if new_val != old_val:
#                     payload[key] = new_val

#         # Helper to merge interest lists (list of dicts {name,type})
#         def merge_interests(old_list, new_list):
#             if not isinstance(old_list, list):
#                 old_list = []
#             if not isinstance(new_list, list):
#                 new_list = []
#             old_names = {i.get("name") for i in old_list if isinstance(i, dict) and i.get("name")}
#             merged = old_list[:]
#             for i in new_list:
#                 if not isinstance(i, dict):
#                     continue
#                 name = i.get("name")
#                 if not name:
#                     continue
#                 typ = i.get("type", "") or ""
#                 if name not in old_names:
#                     merged.append({"name": name, "type": typ})
#             return merged

#         if isinstance(extracted.get("likes"), list):
#             merged_likes = merge_interests(existing.get("likes", []), extracted.get("likes"))
#             if merged_likes != existing.get("likes", []):
#                 payload["likes"] = merged_likes

#         if isinstance(extracted.get("dislikes"), list):
#             merged_dislikes = merge_interests(existing.get("dislikes", []), extracted.get("dislikes"))
#             if merged_dislikes != existing.get("dislikes", []):
#                 payload["dislikes"] = merged_dislikes

#         # Major events: list of dicts with event_name & description
#         if isinstance(extracted.get("major_events"), list):
#             valid_events = []
#             for ev in extracted.get("major_events", []):
#                 if isinstance(ev, dict) and ev.get("event_name"):
#                     valid_events.append({
#                         "event_name": ev.get("event_name"),
#                         "description": ev.get("description", "")
#                     })
#             if valid_events:
#                 old_events = existing.get("major_events", [])
#                 old_names = {e.get("event_name") for e in old_events if isinstance(e, dict) and e.get("event_name")}
#                 new_unique = [e for e in valid_events if e.get("event_name") not in old_names]
#                 if new_unique:
#                     payload["major_events"] = old_events + new_unique

#         # People: expect list of strings "Name (relationship)"
#         if isinstance(extracted.get("people"), list):
#             valid_people = []
#             for p in extracted.get("people", []):
#                 if isinstance(p, str):
#                     m = re.match(r"^(.*)\s*\((.*)\)\s*$", p.strip())
#                     if m:
#                         valid_people.append({"name": m.group(1).strip(), "relationship": m.group(2).strip()})
#                     else:
#                         valid_people.append({"name": p.strip(), "relationship": ""})
#                 elif isinstance(p, dict) and p.get("name"):
#                     valid_people.append({"name": p.get("name"), "relationship": p.get("relationship", "")})
#             if valid_people:
#                 old_people_maps = []
#                 for p in existing.get("people", []):
#                     # existing people are returned as "Name (relationship)" strings
#                     if isinstance(p, str):
#                         m = re.match(r"^(.*)\s*\((.*)\)\s*$", p.strip())
#                         if m:
#                             old_people_maps.append({"name": m.group(1).strip(), "relationship": m.group(2).strip()})
#                         else:
#                             old_people_maps.append({"name": p.strip(), "relationship": ""})
#                 old_names = {p.get("name") for p in old_people_maps}
#                 new_unique = [p for p in valid_people if p.get("name") not in old_names]
#                 if new_unique:
#                     payload["people"] = old_people_maps + new_unique

#         if payload:
#             print("Updating profile in Neo4j with payload:", payload)
#             update_profile_in_neo4j(username, payload)

#     except Exception:
#         return

# async def enrich_profile(username: str):
#     """
#     Extracts user traits, merges with existing Neo4j profile,
#     and updates graph with new data.
#     Compatible with Gemini JSON format:
#     {
#       "nickname": str | null,
#       "age": int | null,
#       "designation": str | null,
#       "location": str | null,
#       "likes": [{"name": "...", "type": "..."}],
#       "dislikes": [{"name": "...", "type": "..."}],
#       "major_events": [{"event_name": "...", "description": "..."}],
#       "people": [{"name": "...", "relationship": "..."}]
#     }
#     """
#     try:
#         # 1️⃣ Fetch last 15 user messages
#         res = supabase.table("conversation_history_2").select("*") \
#             .eq("sender", username) \
#             .order("timestamp", desc=True).limit(15).execute()
#         convos = list(reversed(res.data)) if res.data else []
#         text = "\n".join([c["message"] for c in convos if "message" in c])
#         if not text:
#             print(f"ℹ️ No conversation text for {username}")
#             return

#         # 2️⃣ Extract info via Gemini
        # extract_prompt = f"""You are an AI that analyzes conversations between a user and their AI companion to extract important personal information.

        # Task: Analyze the conversation log and extract a structured summary of key personal details.

        # What to Extract:

        # 1. likes: Things the user enjoys or loves
        # - Hobbies and activities (e.g., "painting", "hiking")
        # - Foods and restaurants (e.g., "biryani", "Truffles")
        # - Media (e.g., "Inception", "The Beatles", "Stranger Things")
        # - Brands, places, or anything they express positive sentiment about

        # 2. dislikes: Things the user does not enjoy or loves
        # - Hobbies and activities (e.g., "painting", "hiking")
        # - Foods and restaurants (e.g., "biryani", "Truffles")
        # - Media (e.g., "Inception", "The Beatles", "Stranger Things")
        # - Brands, places, or anything they express negative sentiment about

        # 3. major_events: Significant life events or milestones mentioned
        # - Career changes (e.g., "started new job at Google")
        # - Life transitions (e.g., "moved to Bangalore", "graduated college")
        # - Important occasions (e.g., "sister's wedding", "promotion")
        # - Only include specific events, not general statements

        # 4. minor_events: Smaller but notable events or experiences
        # - Daily life events (e.g., "went to a concert", "tried a new restaurant")
        # - Travel experiences (e.g., "visited Goa", "road trip to Mysore")
        # - Social activities (e.g., "hung out with friends", "family dinner")
        # - Only include specific events, not general statements

        # 5. people: Names and relationships of people in their life
        # - Format: "Name (relationship)" (e.g., "Priya (sister)", "Alex (colleague)")
        # - Include family, friends, colleagues, partners
        # - Only include if both name AND relationship are mentioned

        # Extraction Rules:
        # - Only extract information explicitly stated in the conversation
        # - Don't infer or assume information not directly mentioned
        # - Use exact quotes or paraphrasing from the conversation
        # - If a category has no clear information, use an empty array []
        # - Maintain consistent formatting across all entries
        # - Remove duplicates

        # Output Format:
        # Return ONLY valid JSON with no additional text, explanations, or markdown:

        # Conversation Log:
        # {text}

        # Output JSON Schema:
        # {{
        # "nickname": "string or null",
        # "age": int or null,
        # "designation": "string or null",
        # "location": "string or null",
        # "likes": [{{"name": "item", "type": "hobby/food/media/brand/etc."}}],
        # "dislikes": [{{"name": "item", "type": "hobby/food/media/brand/etc."}}],
        # "major_events": [{{"event_name": "string", "description": "short description"}}],
        # "people": [{{"name": "string", "relationship": "string"}}]
        # }}

        # Extract and return JSON
        # """
#         raw = generate(extract_prompt, "gemini-2.5-pro")
#         cleaned = clean_json_response(raw)
#         extracted = json.loads(cleaned)
#         print("🧠 Enrichment extracted JSON:", extracted)

#         # 3️⃣ Get existing profile
#         existing = get_profile(username)
#         if not existing:
#             print(f"⚠️ No existing profile found for {username}")
#             return

#         # 4️⃣ Build payload — only new/changed values
#         payload: Dict[str, Any] = {}

#         # --- Scalar fields ---
#         for key in ["nickname", "age", "designation", "location"]:
#             new_val = extracted.get(key)
#             old_val = existing.get(key)
#             if new_val not in [None, "", 0] and new_val != old_val:
#                 payload[key] = new_val

#         # --- Helper to merge lists of dicts ---
#         def merge_dict_lists(old_list, new_list, key_field):
#             if not isinstance(old_list, list):
#                 old_list = []
#             if not isinstance(new_list, list):
#                 new_list = []
#             existing_names = {item.get(key_field) for item in old_list if isinstance(item, dict)}
#             merged = old_list[:]
#             for item in new_list:
#                 if isinstance(item, dict) and item.get(key_field) and item.get(key_field) not in existing_names:
#                     merged.append(item)
#             return merged

#         # Likes / Dislikes / Events / People
#         if extracted.get("likes"):
#             merged_likes = merge_dict_lists(existing.get("likes", []), extracted["likes"], "name")
#             if merged_likes != existing.get("likes", []):
#                 payload["likes"] = merged_likes

#         if extracted.get("dislikes"):
#             merged_dislikes = merge_dict_lists(existing.get("dislikes", []), extracted["dislikes"], "name")
#             if merged_dislikes != existing.get("dislikes", []):
#                 payload["dislikes"] = merged_dislikes

#         if extracted.get("major_events"):
#             merged_events = merge_dict_lists(existing.get("major_events", []), extracted["major_events"], "event_name")
#             if merged_events != existing.get("major_events", []):
#                 payload["major_events"] = merged_events

#         if extracted.get("people"):
#             merged_people = merge_dict_lists(existing.get("people", []), extracted["people"], "name")
#             if merged_people != existing.get("people", []):
#                 payload["people"] = merged_people

#         # 5️⃣ Update only if something changed
#         if payload:
#             print("🔄 Updating Neo4j with payload:", json.dumps(payload, indent=2))
#             update_profile_in_neo4j(username, payload)
#         else:
#             print(f"ℹ️ No new data to update for {username}")

#     except Exception as e:
#         print(f"❌ enrich_profile error for {username}: {e}")