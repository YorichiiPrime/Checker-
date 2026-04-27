import os
import time
import sys
import asyncio
from quart import Quart, request, Response, jsonify
from quart_cors import cors
from groq import AsyncGroq
from groq import APIError # Groq's equivalent of an API Error
import uuid

# Note: This will trigger the loading of the model and DB into RAM
try:
    from rag_engine.vector_search import fast_search
except Exception as e:
    print(f"CRITICAL: Could not load RAG engine. Ensure DB is built. Error: {e}")
    sys.exit(1)
import apis


API_KEY_HARDCODED = apis.api 
MONGO_URI = apis.mongo_uri  
session_timeout = 600
sessions = {}
HISTORY_THRESHOLD = 6

app = Quart(__name__, static_folder='static', template_folder='static')
app = cors(app) 


try:
    if API_KEY_HARDCODED == "YOUR_VALID_GROQ_API_KEY_HERE":
        print("ERROR: Please update API_KEY_HARDCODED in app.py with your GROQ Key.")
        sys.exit(1)
        
    client = AsyncGroq(api_key=API_KEY_HARDCODED)
    SUMMARY_MODEL = "llama-3.1-8b-instant" 
    FINAL_MODEL = "llama-3.3-70b-versatile"
    
except Exception as e:
    print(f"FATAL ERROR initializing Groq client: {e}")
    sys.exit(1)


#  Helper: Validate and Update Session 
async def validate_session(session_id):
    if not session_id or session_id not in sessions:
        return False
        
    current_time = time.time()
    # Access the timestamp inside the session dictionary
    last_activity = sessions[session_id]['last_activity']
    
    if (current_time - last_activity) > session_timeout:
        # clean up RAM and return False
        del sessions[session_id]
        return False
        
    # Valid session: Update activity timestamp (Sliding Window)
    sessions[session_id]['last_activity'] = current_time
    return True

def build_ai_history(session_id):

    session = sessions[session_id]
    memory = session["memory_summary"]
    full_history = session["full_history"]
    total = len(full_history)
    remainder = total % HISTORY_THRESHOLD

    if remainder == 0:

        # if summary exists → start new chunk
        if memory:
            recent_messages = []

        # if summary not yet created → use full history
        else:
            recent_messages = full_history

    else:
        recent_messages = full_history[-remainder:]

    ai_messages = []
    if memory:
        ai_messages.append({
            "role": "system",
            "content": f"Conversation memory:\n{memory}"
        })

    ai_messages.extend(recent_messages)
    return ai_messages

async def update_memory_summary(session_id):

    session = sessions[session_id]
    memory = session["memory_summary"]
    full_history = session["full_history"]
    total = len(full_history)

    if total != 0 and total % HISTORY_THRESHOLD == 0:

        last_chunk = full_history[-HISTORY_THRESHOLD:]
        chunk_string = "\n".join(
            [f"{m['role']}: {m['content']}" for m in last_chunk]
        )

        if memory:
            chunk_string = f"""
Previous memory:
{memory}

New messages:
{chunk_string}
"""
        print("SUMMARY TRIGGERED")
        try:
            summary_res = await client.chat.completions.create(
                model=SUMMARY_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": '''Extract persistent facts from the conversation.

Only include:
- user name
- user goals
- important topics discussed

Do not invent information.

Return concise bullet points.'''
                    },
                    {"role": "user", "content": chunk_string}
                ]
            )
            session["memory_summary"] = summary_res.choices[0].message.content
            print("NEW MEMORY:")
            print(session["memory_summary"])
        except Exception as e:
            print("Summary failed:", e)

# --- ROUTING ---
@app.route('/get_userinfo', methods=['POST'])
async def get_userinfo():
    data = await request.get_json()
    sid = data.get('session_id')
    name = data.get('name')
    contact = data.get('contact')
    
    # Ensure we ALWAYS return a response
    if not sid or sid not in sessions:
        return jsonify({"status": "error", "message": "Session not found. Please start a chat first."}), 401

    try:
        sessions[sid]['user_name'] = name
        sessions[sid]['user_contact'] = contact
        print(f"DEBUG: Profile updated for {name}")
        return jsonify({"status": "success", "message": "Profile updated"}), 200
    except Exception as e:
        print(f"ERROR updating user info: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/get-session', methods=['GET'])
async def get_session():
    sid = str(uuid.uuid4())
    sessions[sid] = {
    'last_activity': time.time(),
    'memory_summary': "",
    'recent_history': [],
    'full_history': [],  # optional, only for frontend reload
    'user_name': None,
    'user_contact': None
    }
    return jsonify({"session_id": sid})

@app.route('/get-history', methods=['POST'])
async def get_history():
    try:
        data = await request.get_json()
        sid = data.get('session_id') if data else None
        
        # Return empty history for unknown/expired sessions instead of 401
        # This prevents the frontend from destroying its session state
        if not sid or sid not in sessions:
            return jsonify({"history": []})
        
        if await validate_session(sid):
            return jsonify({"history": sessions[sid]['full_history']})
        
        # Session expired — return empty history gracefully
        return jsonify({"history": []})
    except Exception as e:
        print(f"ERROR in get-history: {e}")
        return jsonify({"history": []})

# --- Route: Streaming Chat with History Persistence ---
@app.route('/stream-chat', methods=['POST'])
async def stream_chat():
    data = await request.get_json()
    user_prompt = data.get('prompt', '')
    session_id = data.get('session_id', '')
    
    print(f"\n[DEBUG] Received prompt: {user_prompt}")

    if not session_id or session_id not in sessions:
        # Re-initialize the session in RAM if it's missing (e.g. after restart)
        session_id = session_id or str(uuid.uuid4())
        sessions[session_id] = {
            'last_activity': time.time(),
            'memory_summary': "",
            'full_history': [],
            'user_name': None,
            'user_contact': None
        }
        print(f"DEBUG: Session {session_id} re-initialized on-the-fly.")
    else:
        # Refresh the activity timer for existing sessions
        sessions[session_id]['last_activity'] = time.time()

    # Session is guaranteed valid at this point (re-created or refreshed above)
    
    # 2. Append history and get history context
    sessions[session_id]['full_history'].append({"role": "user", "content": user_prompt})

    ai_history = build_ai_history(session_id)
    print(f"Current history for reply :{ai_history} and {len(ai_history)}")
    user_name = sessions[session_id].get('user_name') or "User"

    # 3. Define the Streaming Generator
    async def generate():
        try:
            # Move retrieval and shortening INSIDE the generator to prevent 500 timeouts
            results, _ = await asyncio.to_thread(fast_search, user_prompt, top_k=5)
            raw_context = "\n".join(results['documents'][0])

            # Initialize with a fallback in case the AI shortener fails
            concise_context = "No specific context found."
            
            try:
                shortener_response = await client.chat.completions.create(
                    model=SUMMARY_MODEL,
                    messages=[
                        {"role": "system", "content": "Extract only relevant info for the user question."},
                        {"role": "user", "content": f"User Question: {user_prompt}\n\nContext: {raw_context}"}
                    ]
                )
                concise_context = shortener_response.choices[0].message.content
                print(f"[DEBUG] Concise Context Prepared \n Selected topics get summurized")
            except Exception as e:
                print(f"Warning: AI Shortener failed: {e}")

            # 4. Final Answer Generation
            full_res = ""
            try:
                response_stream = await client.chat.completions.create(
                    model=FINAL_MODEL,
                    messages=ai_history + [
                        {"role": "system", "content": f"""
You are GraceBot, a friendly assistant giving short answers.

User name: {user_name}

Always read the conversation history and memory before replying. 
If information exists there (like the user's name or earlier topics), use it and do not say you forgot it.

Say you don't have information only if it is not in the history, memory, or context.

Context:
{concise_context}
"""},
                        {"role": "user", "content": user_prompt}
                    ],
                    stream=True
                )
                print("Chat get generated....................................................")
            except APIError as e:
                print("FINAL model error")
                yield  "⚠️ AI service unavailable."
                return

            async for chunk in response_stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_res += content 
                    yield content
            
            # 5. Finalize history update
            sessions[session_id]['full_history'].append({"role": "assistant", "content": full_res})

            await update_memory_summary(session_id)

        except Exception as e:
            print(f"CRITICAL STREAM ERROR: {e}")
            yield f"⚠️ GraceBot encountered an error: {str(e)}"

    return Response(generate(), mimetype='text/plain')

if __name__ == '__main__':

    print("🚀 Flask Groq server starting...")
    app.run(debug=True, host='127.0.0.1',use_reloader=False, port=5000)