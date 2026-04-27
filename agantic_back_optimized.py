'''
A semi agentic optimized backend for this project,
might  strugle with memory management and long conversations,
but should be more efficient and cost effective for short interactions.
'''

import os
import time
import sys
import json
import re
import asyncio
from quart import Quart, request, Response, jsonify
from quart_cors import cors
from groq import AsyncGroq
from groq import APIError
import uuid

try:
    from rag_engine.vector_search import fast_search
except Exception as e:
    print(f"CRITICAL: Could not load RAG engine. Ensure DB is built. Error: {e}")
    sys.exit(1)
import apis

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
API_KEY_HARDCODED = apis.api
MONGO_URI = apis.mongo_uri
session_timeout = 600
sessions = {}
HISTORY_THRESHOLD = 6
MAX_AGENT_ITERATIONS = 2

app = Quart(__name__, static_folder='static', template_folder='static')
app = cors(app)

try:
    if API_KEY_HARDCODED == "YOUR_VALID_GROQ_API_KEY_HERE":
        print("ERROR: Please update API_KEY_HARDCODED with your GROQ Key.")
        sys.exit(1)
    client = AsyncGroq(api_key=API_KEY_HARDCODED)
    SUMMARY_MODEL   = "llama-3.1-8b-instant"
    PLANNER_MODEL   = "llama-3.1-8b-instant"   # lightweight; only does JSON planning
    REASONING_MODEL = "llama-3.3-70b-versatile" # heavyweight; reasoning + final answer
except Exception as e:
    print(f"FATAL ERROR initializing Groq client: {e}")
    sys.exit(1)


# ─────────────────────────────────────────────
#  PROMPTS
# ─────────────────────────────────────────────

PLANNER_SYSTEM = """\
You are a conservative planning agent. Given a user question, conversation history, and memory,
decide which tools must be called BEFORE answering.

Available tools (planner stage only):
  search          – vector search over documents
  history_lookup  – search conversation history for relevant facts
  memory_lookup   – retrieve stored long-term memory facts (e.g. user name)

Rules:
- Only include a tool if it is clearly required.
- If the question requires document or factual knowledge → include "search"
- If the question references something said earlier in the conversation → include "history_lookup"
- If the question asks for a stored user fact (name, preference) AND no memory is shown → include "memory_lookup"
- If memory already contains the answer, do NOT include "memory_lookup"
- For greetings, simple math, or general knowledge → return an empty plan
- NEVER include "expand_search" — it is reserved for the verifier stage only
- NEVER include both "search" and "expand_search"
- Do not add tools you are not sure are needed

Respond ONLY with valid JSON. No prose. No explanation.
Example: {"plan": ["search"]}
"""

CONTEXT_BUILDER_SYSTEM = """\
You are a context compressor. Given a user question and raw tool results,
extract ONLY the facts needed to answer the question.
Keep output under 200 tokens. Use bullet points. Remove redundant text. Be ruthlessly concise.
"""

REASONING_SYSTEM = """\
You are GraceBot's reasoning engine. Synthesize the provided facts into a concise draft answer.

Rules:
- Never invent facts. Use only the provided context, history, and memory.
- If a fact is not present, say you don't have that information.
- Produce a draft answer, not a final polished response.
- Keep the draft under 200 tokens. Include only facts needed to answer the question.
- Do not restate the full context. Do not repeat the question.
"""

VERIFIER_SYSTEM = """\
You are a verification agent. Check if the draft answer relies ONLY on facts in the provided context.

If the draft is grounded → respond with exactly: {"verification": "pass"}
If the draft references facts NOT in the context → respond with exactly: {"verification": "search_more"}

Respond ONLY with valid JSON. No prose.
"""

QUERY_REWRITER_SYSTEM = """\
You are a retrieval query optimizer. Rewrite the user question into a short, keyword-rich search query.

Rules:
- Expand abbreviations and jargon into full terms
- Add domain-relevant keywords the documents likely use
- Remove filler words (why, is, what, the, etc.)
- Output ONLY the rewritten query string. No JSON, no explanation, no punctuation.

Example:
Input:  "why is bert bad for contracts?"
Output: BERT limitations long documents contract analysis context window
"""

FINAL_SYSTEM = """\
You are GraceBot, a friendly and helpful assistant.
Produce the final, user-facing response based on the verified draft.

Rules:
- Be clear and concise.
- Use the user's name if known.
- Do not expose internal reasoning, tool calls, or intermediate steps.
- If no answer was found, say: "I do not have enough information to answer that."
"""


# ─────────────────────────────────────────────
#  SESSION HELPERS
# ─────────────────────────────────────────────

async def validate_session(session_id):
    if not session_id or session_id not in sessions:
        return False
    current_time = time.time()
    last_activity = sessions[session_id]['last_activity']
    if (current_time - last_activity) > session_timeout:
        del sessions[session_id]
        return False
    sessions[session_id]['last_activity'] = current_time
    return True

# ── Rate-limit safe Groq call ───────────────────────────────────────────────
async def groq_call_with_retry(model: str, messages: list, max_tokens: int, stream: bool = False):
    """Wraps client.chat.completions.create with exponential backoff on 429."""
    delays = [1, 2, 4]
    last_exc = None
    for attempt, delay in enumerate(delays):
        try:
            return await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                stream=stream
            )
        except APIError as e:
            if hasattr(e, 'status_code') and e.status_code == 429:
                print(f"[RATE LIMIT] Attempt {attempt + 1} — retrying in {delay}s...")
                await asyncio.sleep(delay)
                last_exc = e
            else:
                raise
    raise last_exc  # exhausted retries

# ── Auto-extract user name from prompt ─────────────────────────────────────
_NAME_PATTERNS = re.compile(
    r"(?:my name is|i am|i'm|im)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    re.IGNORECASE
)

def extract_name_from_prompt(prompt: str, session_id: str):
    """If user states their name, store it immediately without a memory_lookup call."""
    if sessions[session_id].get('user_name'):
        return  # already known
    match = _NAME_PATTERNS.search(prompt)
    if match:
        name = match.group(1).strip().title()
        sessions[session_id]['user_name'] = name
        print(f"[AGENT] Auto-extracted user name: {name}")

def build_ai_history(session_id):
    session = sessions[session_id]
    memory = session["memory_summary"]
    full_history = session["full_history"]
    total = len(full_history)
    remainder = total % HISTORY_THRESHOLD

    if remainder == 0:
        recent_messages = [] if memory else full_history
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
        chunk_string = "\n".join([f"{m['role']}: {m['content']}" for m in last_chunk])
        if memory:
            chunk_string = f"Previous memory:\n{memory}\n\nNew messages:\n{chunk_string}"

        print("SUMMARY TRIGGERED")
        try:
            summary_res = await groq_call_with_retry(
                model=SUMMARY_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "Extract persistent facts: user name, user goals, important topics. Bullet points only. Do not invent."
                    },
                    {"role": "user", "content": chunk_string}
                ],
                max_tokens=150
            )
            session["memory_summary"] = summary_res.choices[0].message.content
            print("NEW MEMORY:\n", session["memory_summary"])
        except Exception as e:
            print("Summary failed:", e)


# ─────────────────────────────────────────────
#  TOOL REGISTRY
# ─────────────────────────────────────────────

async def tool_search(query: str, top_k: int = 5) -> str:
    results, _ = await asyncio.to_thread(fast_search, query, top_k=top_k)
    docs = results['documents'][0]
    return "\n".join(docs)

async def tool_expand_search(query: str) -> str:
    return await tool_search(query, top_k=10)

def tool_history_lookup(query: str, session_id: str) -> str:
    history = sessions[session_id].get('full_history', [])
    q = query.lower()
    relevant = [
        f"{m['role']}: {m['content']}"
        for m in history
        if q in m['content'].lower()
    ]
    return "\n".join(relevant[-6:]) if relevant else "No relevant history found."

def tool_memory_lookup(key: str, session_id: str) -> str:
    memory = sessions[session_id].get('memory_summary', '')
    if not memory:
        return "No memory available."
    return memory

async def execute_tool(tool_call: dict, session_id: str) -> str:
    """Dispatch a tool call dict to the appropriate function."""
    name = tool_call.get("tool", "")
    if name == "search":
        return await tool_search(tool_call.get("query", ""), tool_call.get("top_k", 5))
    elif name == "expand_search":
        return await tool_expand_search(tool_call.get("query", ""))
    elif name == "history_lookup":
        return tool_history_lookup(tool_call.get("query", ""), session_id)
    elif name == "memory_lookup":
        return tool_memory_lookup(tool_call.get("key", ""), session_id)
    else:
        return f"Unknown tool: {name}"


# ─────────────────────────────────────────────
#  JSON EXTRACTION HELPER
# ─────────────────────────────────────────────

def extract_json(text: str) -> dict | None:
    """Robustly extract first JSON object from a string."""
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


# ─────────────────────────────────────────────
#  AGENTIC LOOP
# ─────────────────────────────────────────────

async def rewrite_query(user_prompt: str) -> str:
    """Use small model to rewrite user question into a retrieval-optimized query."""
    try:
        res = await groq_call_with_retry(
            model=PLANNER_MODEL,
            messages=[
                {"role": "system", "content": QUERY_REWRITER_SYSTEM},
                {"role": "user",   "content": user_prompt}
            ],
            max_tokens=48
        )
        rewritten = res.choices[0].message.content.strip()
        print(f"[AGENT] Query rewritten: '{user_prompt}' → '{rewritten}'")
        return rewritten if rewritten else user_prompt
    except Exception as e:
        print(f"[AGENT] Query rewrite failed: {e}")
        return user_prompt


async def agentic_pipeline(user_prompt: str, session_id: str):
    """
    Multi-step agentic pipeline.
    Yields string chunks for streaming to the client.

    Stages:
      1. Name extraction (zero-cost, regex)
      2. Planner   → decide which tools to call (conservative, no expand_search)
      3. Query rewrite → optimized retrieval query
      4. Tool loop → execute tools (deduped, no expand_search at this stage)
      5. Context   → compress tool results (skipped if results < 1200 chars)
      6. Reasoning → draft answer (<200 tokens)
      7. Verifier  → check grounding; triggers expand_search only if needed
      8. Final     → stream polished answer to user
    """

    session = sessions[session_id]

    # ── PRE-STAGE: Auto name extraction ─────────────────────────────
    extract_name_from_prompt(user_prompt, session_id)

    ai_history = build_ai_history(session_id)
    user_name = session.get('user_name') or "User"
    memory = session.get('memory_summary', '')

    # ── STAGE 1: PLANNER ────────────────────────────────────────────
    planner_user_msg = (
        f"User question: {user_prompt}\n\n"
        f"Known memory:\n{memory if memory else 'None'}\n\n"
        f"Recent history (last 3 turns):\n"
        + "\n".join(
            f"{m['role']}: {m['content']}"
            for m in ai_history[-6:]
            if isinstance(m['content'], str)
        )
    )

    try:
        planner_res = await groq_call_with_retry(
            model=PLANNER_MODEL,
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user",   "content": planner_user_msg}
            ],
            max_tokens=64
        )
        planner_text = planner_res.choices[0].message.content
        plan_json = extract_json(planner_text) or {}
        plan = plan_json.get("plan", [])
    except Exception as e:
        print(f"Planner failed: {e}")
        plan = ["search"]   # safe fallback

    # Safety: strip expand_search if planner hallucinated it, deduplicate
    plan = list(dict.fromkeys(t for t in plan if t != "expand_search"))
    print(f"[AGENT] Plan: {plan}")

    # ── STAGE 2: QUERY REWRITE (only if search is planned) ──────────
    retrieval_query = user_prompt
    if "search" in plan:
        retrieval_query = await rewrite_query(user_prompt)

    # ── STAGE 3: TOOL EXECUTION ──────────────────────────────────────
    tool_results: list[dict] = []
    executed_tools: set[str] = set()

    for tool_name in plan:
        if tool_name in executed_tools:
            continue  # no duplicate execution
        try:
            if tool_name == "search":
                tool_call = {"tool": "search", "query": retrieval_query, "top_k": 5}
            elif tool_name == "history_lookup":
                tool_call = {"tool": "history_lookup", "query": user_prompt}
            elif tool_name == "memory_lookup":
                tool_call = {"tool": "memory_lookup", "key": "all"}
            else:
                continue

            result_text = await execute_tool(tool_call, session_id)
            tool_results.append({"tool": tool_name, "result": result_text})
            executed_tools.add(tool_name)
            print(f"[AGENT] Tool '{tool_name}' executed. Result length: {len(result_text)}")
        except Exception as e:
            print(f"[AGENT] Tool '{tool_name}' failed: {e}")
            tool_results.append({"tool": tool_name, "result": "Tool execution failed."})

    # ── STAGE 4: CONTEXT BUILDER (skipped if results are short) ──────
    if tool_results:
        raw_tool_dump = "\n\n".join(
            f"[{r['tool']}]\n{r['result']}" for r in tool_results
        )
        if len(raw_tool_dump) < 1200:
            # Results are small enough — skip LLM compression
            concise_context = raw_tool_dump
            print(f"[AGENT] Context builder skipped (results {len(raw_tool_dump)} chars < 1200)")
        else:
            try:
                ctx_res = await groq_call_with_retry(
                    model=SUMMARY_MODEL,
                    messages=[
                        {"role": "system", "content": CONTEXT_BUILDER_SYSTEM},
                        {"role": "user",   "content": f"User question: {user_prompt}\n\nRaw tool results:\n{raw_tool_dump}"}
                    ],
                    max_tokens=256
                )
                concise_context = ctx_res.choices[0].message.content
            except Exception as e:
                print(f"Context builder failed: {e}")
                concise_context = raw_tool_dump[:1200]  # fallback: truncate
    else:
        concise_context = "No external context retrieved."
        raw_tool_dump = ""

    print(f"[AGENT] Context length: {len(concise_context)}")

    # ── STAGE 5 + 6: REASONING + VERIFIER LOOP ───────────────────────
    draft_answer = ""
    for iteration in range(MAX_AGENT_ITERATIONS):
        print(f"[AGENT] Reasoning iteration {iteration + 1}")

        reasoning_user_content = (
            f"User question: {user_prompt}\n\n"
            f"Memory:\n{memory if memory else 'None'}\n\n"
            f"Context facts:\n{concise_context}\n\n"
            f"Conversation history:\n"
            + "\n".join(
                f"{m['role']}: {m['content']}"
                for m in ai_history[-4:]
                if isinstance(m['content'], str)
            )
        )

        try:
            reasoning_res = await groq_call_with_retry(
                model=REASONING_MODEL,
                messages=[
                    {"role": "system", "content": REASONING_SYSTEM},
                    {"role": "user",   "content": reasoning_user_content}
                ],
                max_tokens=256   # reduced from 512
            )
            draft_answer = reasoning_res.choices[0].message.content
        except Exception as e:
            print(f"Reasoning failed: {e}")
            draft_answer = "I was unable to generate a response at this time."
            break

        print(f"[AGENT] Draft produced. Length: {len(draft_answer)}")

        # Verifier
        verifier_content = (
            f"User question: {user_prompt}\n\n"
            f"Context facts:\n{concise_context}\n\n"
            f"Draft answer:\n{draft_answer}"
        )
        try:
            verifier_res = await groq_call_with_retry(
                model=PLANNER_MODEL,
                messages=[
                    {"role": "system", "content": VERIFIER_SYSTEM},
                    {"role": "user",   "content": verifier_content}
                ],
                max_tokens=16
            )
            verifier_text = verifier_res.choices[0].message.content
            verifier_json = extract_json(verifier_text) or {}
            verification = verifier_json.get("verification", "pass")
        except Exception as e:
            print(f"Verifier failed: {e}")
            verification = "pass"  # fail open

        print(f"[AGENT] Verification: {verification}")

        if verification == "pass":
            break

        # Only expand_search on explicit verifier request, and only once
        if verification == "search_more" and iteration < MAX_AGENT_ITERATIONS - 1:
            print("[AGENT] Verifier requested more search. Running expand_search...")
            try:
                expand_query = await rewrite_query(user_prompt)
                expanded = await tool_expand_search(expand_query)
                tool_results.append({"tool": "expand_search", "result": expanded})
                raw_tool_dump = "\n\n".join(
                    f"[{r['tool']}]\n{r['result']}" for r in tool_results
                )
                ctx_res2 = await groq_call_with_retry(
                    model=SUMMARY_MODEL,
                    messages=[
                        {"role": "system", "content": CONTEXT_BUILDER_SYSTEM},
                        {"role": "user",   "content": f"User question: {user_prompt}\n\nRaw tool results:\n{raw_tool_dump}"}
                    ],
                    max_tokens=256
                )
                concise_context = ctx_res2.choices[0].message.content
            except Exception as e:
                print(f"Expand search failed: {e}")
                break  # don't loop again on expand failure

    # ── STAGE 7: FINAL RESPONSE GENERATOR (STREAMING) ────────────────
    final_messages = ai_history + [
        {
            "role": "system",
            "content": (
                f"{FINAL_SYSTEM}\n\n"
                f"User name: {user_name}\n\n"
                f"Verified draft:\n{draft_answer}"
            )
        },
        {"role": "user", "content": user_prompt}
    ]

    full_res = ""
    try:
        stream = await groq_call_with_retry(
            model=REASONING_MODEL,
            messages=final_messages,
            max_tokens=512,
            stream=True
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_res += content
                yield content
    except APIError as e:
        print(f"Final model API error: {e}")
        yield "⚠️ AI service unavailable."
        return
    except Exception as e:
        print(f"Final model error: {e}")
        yield f"⚠️ GraceBot encountered an error: {str(e)}"
        return

    # Persist assistant response
    sessions[session_id]['full_history'].append({"role": "assistant", "content": full_res})
    await update_memory_summary(session_id)


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────

@app.route('/get_userinfo', methods=['POST'])
async def get_userinfo():
    data = await request.get_json()
    sid = data.get('session_id')
    name = data.get('name')
    contact = data.get('contact')

    if not sid or sid not in sessions:
        return jsonify({"status": "error", "message": "Session not found."}), 401

    try:
        sessions[sid]['user_name'] = name
        sessions[sid]['user_contact'] = contact
        print(f"DEBUG: Profile updated for {name}")
        return jsonify({"status": "success", "message": "Profile updated"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/get-session', methods=['GET'])
async def get_session():
    sid = str(uuid.uuid4())
    sessions[sid] = {
        'last_activity': time.time(),
        'memory_summary': "",
        'full_history': [],
        'user_name': None,
        'user_contact': None
    }
    return jsonify({"session_id": sid})


@app.route('/get-history', methods=['POST'])
async def get_history():
    try:
        data = await request.get_json()
        sid = data.get('session_id') if data else None
        if not sid or sid not in sessions:
            return jsonify({"history": []})
        if await validate_session(sid):
            return jsonify({"history": sessions[sid]['full_history']})
        return jsonify({"history": []})
    except Exception as e:
        print(f"ERROR in get-history: {e}")
        return jsonify({"history": []})


@app.route('/stream-chat', methods=['POST'])
async def stream_chat():
    data = await request.get_json()
    user_prompt = data.get('prompt', '')
    session_id  = data.get('session_id', '')

    print(f"\n[DEBUG] Received prompt: {user_prompt}")

    # Ensure session exists
    if not session_id or session_id not in sessions:
        session_id = session_id or str(uuid.uuid4())
        sessions[session_id] = {
            'last_activity': time.time(),
            'memory_summary': "",
            'full_history': [],
            'user_name': None,
            'user_contact': None
        }
        print(f"DEBUG: Session {session_id} re-initialized.")
    else:
        sessions[session_id]['last_activity'] = time.time()

    # Auto-extract name before pipeline (zero-cost regex)
    extract_name_from_prompt(user_prompt, session_id)

    # Append user message to history
    sessions[session_id]['full_history'].append({"role": "user", "content": user_prompt})

    async def generate():
        try:
            async for chunk in agentic_pipeline(user_prompt, session_id):
                yield chunk
        except Exception as e:
            print(f"CRITICAL STREAM ERROR: {e}")
            yield f"⚠️ GraceBot encountered an error: {str(e)}"

    return Response(generate(), mimetype='text/plain')


if __name__ == '__main__':
    print("🚀 GraceBot Agentic server starting...")
    app.run(debug=True, host='127.0.0.1', use_reloader=False, port=5000)