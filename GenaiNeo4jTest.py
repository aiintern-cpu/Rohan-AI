from typing import List, Optional, Any, Dict
from dotenv import load_dotenv
from supabase import create_client
from google import genai
from neo4j import GraphDatabase
import os
import json
import re

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if genai and GEMINI_API_KEY else None

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASS = os.getenv("NEO4J_PASS")
NEO4J_DB = "demo2"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

def clean_json_response(raw_text: str) -> str:
    cleaned = re.sub(r"^(`{3,}|'{3,})\s*json\s*", "", raw_text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"(`{3,}|'{3,})$", "", cleaned.strip())
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    return cleaned

def generate(prompt: str, model: str = "gemini-2.5-pro") -> str:
    if not client:
        raise ValueError("GenAI client not initialized. Check GEMINI_API_KEY.")
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text

def update_graph_from_payload(payload: Dict[str, Any]):
    """Entry point: handles Neo4j session and calls recursive function."""
    with driver.session(database=NEO4J_DB) as session:
        session.execute_write(process_payload, None, payload)
    print("✅ Graph update completed successfully!")

def process_payload(tx, parent_node: Dict[str, str] | None, payload: Dict[str, Any]):
    """
    Recursively process a nested payload and insert nodes + relationships into Neo4j.
    parent_node: dict like {"type": "USER", "name": "SUNIDHI"}
    payload: nested JSON describing graph relations
    """
    for key, value in payload.items():
        # Split key into node type and name
        try:
            node_type, node_name = key.split("_", 1)
        except ValueError:
            print(f"⚠️ Skipping malformed key: {key}")
            continue

        # Ensure node exists
        tx.run(f"MERGE (n:{node_type} {{name: $name}})", name=node_name)

        # Handle simple relationship: (parent) -[:RELATION]-> (current)
        if isinstance(value, str):
            # Primary relationship case
            if parent_node:
                rel = value.upper()
                query = f"""
                MATCH (p:{parent_node['type']} {{name: $pname}}),
                    (c:{node_type} {{name: $cname}})
                MERGE (p)-[:{rel}]->(c)
                """
                tx.run(query, pname=parent_node["name"], cname=node_name)

        # Handle nested relationships (secondary / deeper)
        elif isinstance(value, dict):
            # If this dict has a direct relationship field like "_REL"
            rel = value.get("_REL")
            if rel and parent_node:
                rel = rel.upper()
                query = f"""
                MATCH (p:{parent_node['type']} {{name: $pname}}),
                    (c:{node_type} {{name: $cname}})
                MERGE (p)-[:{rel}]->(c)
                """
                tx.run(query, pname=parent_node["name"], cname=node_name)

            # Continue recursion for nested relationships (excluding "_REL")
            nested_dict = {k: v for k, v in value.items() if k != "_REL"}
            process_payload(tx, {"type": node_type, "name": node_name}, nested_dict)

def enrich_profile(username: str = "misha"):
    """
    Extracts structured profile info using Gemini and updates Neo4j safely.
    Compatible with JSON schema:
    {
        "USERNODE_USERNAME": {
            "NODETYPE1_NODENAME1": {
                "_REL": "RELATIONSHIP1", // Primary relationship
                "NODETYPE2_NODENAME2": "RELATIONSHIP2" // Secondary relationship
            },
            "NODETYPE3_NODENAME3": "RELATIONSHIP3"
        }
    }
    """
    try:
        # 1️⃣ Fetch last 15 user messages
        res = supabase.table("conversation_history").select("*") \
            .eq("sender", username).order("timestamp", desc=True).limit(30).execute()
        convos = list(reversed(res.data)) if res.data else []
        text = "\n".join([c["message"] for c in convos if "message" in c])
        print(f"Conversation History:\n {text}")
        if not text:
            print(f"ℹ️ No messages found for {username}")
            return
        # 2️⃣ Extract info using Gemini
        
        extract_prompt = f"""
            You are an intelligent graph relation extractor.
            Given the conversation history between a user and the system, you must extract entities and their relationships
            according to the allowed node types and relationships below.

            Allowed Node Types:
            USER, PERSON, LOCATION, JOB, ORGANIZATION, COMPANY, COLLEGE, SCHOOL, INTEREST, PROJECT, SKILL, DOMAIN, EVENT, COURSE

            Allowed Relationship Types:
            STUDIES_AT, WORKS_AT, LIVES_IN, HAS_SKILL, LIKES, DISLIKES, FRIENDS_WITH, COLLEAGUES_WITH,
            WORKS_ON, PARTICIPATED_IN, ENROLLED_IN, RELATED_TO, BELONGS_TO_DOMAIN

            JSON Output Format:
            {{
                "USERNODE_USERNAME": {{
                    "NODETYPE1_NODENAME1": {{
                        "_REL": "RELATIONSHIP1", // Primary relationship
                        "NODETYPE2_NODENAME2": "RELATIONSHIP2" // Secondary relationship
                    }},
                    "NODETYPE3_NODENAME3": "RELATIONSHIP3"
                }}
            }}

            Rules:
            1. You must always respond **only in JSON**. No text outside JSON.
            2. Node keys must be in the format: NODETYPE_NODENAME (e.g., PERSON_SUJAL, COLLEGE_GMIT).
            3. If the same node has both a direct and nested relationship, include "_REL" for the direct relationship.
            4. Keep names concise and proper nouns capitalized (e.g., "INFOSIGHT", "BANGALORE").
            5. Avoid unrelated or speculative data; extract only what’s clearly implied in the conversation.
            6. Keep the entire response in uppercase letters. 

            Username: {username.upper()}
            Conversation History:
            {text}
            Your Response (JSON Only):
        """

        raw = generate(extract_prompt, "gemini-2.5-pro")
        cleaned = clean_json_response(raw)
        payload = json.loads(cleaned)

        print("🧠 Enrichment extracted:", json.dumps(payload, indent=2))

        update_graph_from_payload(payload)

    except Exception as e:
        print(f"❌ Error during enrichment for {username}: {e}")

enrich_profile()

def get_user_profile(username: str) -> dict:
    query = """
    MATCH (u:USER {name: $username})
    OPTIONAL MATCH (u)-[:LIVES_IN]->(loc:LOCATION)
    OPTIONAL MATCH (u)-[:WORKS_AT]->(comp:COMPANY)
    OPTIONAL MATCH (u)-[:STUDIES_AT]->(col)
    OPTIONAL MATCH (u)-[:LIKES]->(like:INTEREST)
    OPTIONAL MATCH (u)-[:DISLIKES]->(dis:INTEREST)

    RETURN 
        collect(DISTINCT loc.name) AS lives_in,
        collect(DISTINCT comp.name) AS works_at,
        collect(DISTINCT col.name) AS studies_at,
        collect(DISTINCT like.name) AS likes,
        collect(DISTINCT dis.name) AS dislikes
    """

    with driver.session(database=NEO4J_DB) as session:
        record = session.run(query, username=username.upper()).single()

        # Ensure record exists
        if not record:
            return {
                "lives_in": [],
                "job" : [],
                "works_at": [],
                "studies_at": [],
                "likes": [],
                "dislikes": []
            }

        # Safely extract without errors
        def safe_list(key):
            value = record.get(key)
            if not value:
                return []
            # Remove None values
            return [v for v in value if v is not None]

        return {
            "lives_in": safe_list("lives_in"),
            "works_at": safe_list("works_at"),
            "studies_at": safe_list("studies_at"),
            "likes": safe_list("likes"),
            "dislikes": safe_list("dislikes"),
        }

# def get_user_profile(username: str) -> dict:
#     query = """
#     MATCH (u:USER {name: $username})
#     OPTIONAL MATCH (u)-[:LIVES_IN]->(loc:LOCATION)
#     OPTIONAL MATCH (u)-[:WORKS_AT]->(comp:COMPANY)
#     OPTIONAL MATCH (u)-[:STUDIES_AT]->(col)
#     OPTIONAL MATCH (u)-[:LIKES]->(like:INTEREST)
#     OPTIONAL MATCH (u)-[:DISLIKES]->(dis:INTEREST)

#     RETURN 
#         collect(DISTINCT loc.name) AS lives_in,
#         collect(DISTINCT comp.name) AS works_at,
#         collect(DISTINCT col.name) AS studies_at,
#         collect(DISTINCT like.name) AS likes,
#         collect(DISTINCT dis.name) AS dislikes
#     """

#     with driver.session(database=NEO4J_DB) as session:
#         result = session.run(query, username=username.upper()).single()
# # "likes": [x for x in result["likes"] if x],
#         return {
#             "username" : f"{username}",
#             "lives_in": [x for x in result["lives_in"] if x],
#             "works_at": [x for x in result["works_at"] if x],
#             "studies_at": [x for x in result["studies_at"] if x],
#             "likes": ["yoga", "walking"],
#             "dislikes": [x for x in result["dislikes"] if x]
#        }

profile = get_user_profile("SUNIDHI")
prompt = f"""
    <User Profile>
    Username: {profile['username']}
    Works At: {profile['works_at']}
    Studies At: {profile['studies_at']}
    Location: {profile['lives_in']}
    Likes: {', '.join([f"{like}" for like in profile.get('likes', [])])}
    Dislikes: {', '.join([f"{dislike}" for dislike in profile.get('dislikes', [])])}
    <User Profile>
    """

print(prompt)