# ceaf_tester.py
import httpx
import asyncio
import litellm
import json
import os
from datetime import datetime
from dotenv import load_dotenv
import argparse
import uuid

# --- Configuration ---
# These can be overridden by command-line arguments
CONFIG = {
    "AURA_API_URL": "http://localhost:8000",
    "AURA_USERNAME": "admin6",  # Change to your test user
    "AURA_PASSWORD": "admin",  # Change to your test password

    # The ID of the CEAF agent you want to test
    "TARGET_AGENT_ID": "ebf05c20-269a-4f5b-9ba8-ec5b1097ecbf",  # IMPORTANT: Change this or provide via CLI

    "MAX_TURNS": 50,  # The number of conversational turns (1 turn = 1 bot message + 1 agent response)

    # The LLM model the TESTER BOT will use to generate its messages
    "TESTER_BOT_MODEL": "openrouter/openai/gpt-oss-20b",

    # The "personality" of the tester bot. This guides its conversational style.
    "TESTER_BOT_PERSONA": """You are an AI testing assistant. Your goal is to engage a target AI in a long, coherent, and interesting conversation.
- Be curious. Ask follow-up questions.
- Introduce new, related topics if the conversation lulls.
- Sometimes challenge the AI's statements or ask for clarification.
- Vary your sentence structure and length.
- Your responses should be natural and conversational, not robotic.
- Your goal is to keep the conversation going for many turns.
""",

    # The first message the tester bot will send to start the conversation.
    "INITIAL_PROMPT": "Eu gosto de cerejas no sorvete"

}

# --- Load Environment Variables ---
load_dotenv()
litellm.api_key = os.getenv("OPENROUTER_API_KEY")
litellm.api_base = "https://openrouter.ai/api/v1"


# --- API Client Functions ---

async def login(client, url, username, password):
    """Logs into the Aura API and returns the auth token."""
    print(f"Attempting to log in as '{username}'...")
    try:
        response = await client.post(f"{url}/auth/login", json={"username": username, "password": password})
        response.raise_for_status()
        print("✅ Login successful.")
        return response.json()["access_token"]
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        print(f"❌ Login failed: {e}")
        return None


async def chat_with_agent(client, url, token, agent_id, message, session_id):
    """Sends a message to a CEAF agent and gets a response."""
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"message": message}
    if session_id:
        payload["session_id"] = session_id

    try:
        response = await client.post(f"{url}/agents/{agent_id}/chat", headers=headers, json=payload, timeout=180.0)
        response.raise_for_status()
        data = response.json()
        return data.get("response"), data.get("session_id")
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        print(f"❌ Error during chat: {e}")
        return f"Error: Could not get a response from the agent. ({e})", session_id


# --- Tester Bot's "Brain" ---

async def generate_tester_reply(chat_history, persona, model):
    """Uses an LLM to generate the tester bot's next message."""
    messages = [
        {"role": "system", "content": persona},
        *chat_history
    ]

    print("🧠 Tester bot is thinking...")
    try:
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            temperature=0.8,
            max_tokens=200
        )
        reply = response.choices[0].message.content.strip()
        print("✅ Tester bot generated a reply.")
        return reply
    except Exception as e:
        print(f"❌ Tester bot's LLM failed: {e}")
        return "I'm not sure what to say next. Can you elaborate?"


# --- Main Test Runner ---

async def run_test_session(config):
    """Runs a full, autonomous test session against a CEAF agent."""

    async with httpx.AsyncClient() as client:
        # 1. Login to get token
        token = await login(client, config["AURA_API_URL"], config["AURA_USERNAME"], config["AURA_PASSWORD"])
        if not token:
            return

        # 2. Initialize test state
        agent_id = config["TARGET_AGENT_ID"]
        max_turns = config["MAX_TURNS"]

        full_conversation_log = []
        tester_chat_history = []  # This is the context for the tester's LLM

        current_turn = 0
        ceaf_session_id = str(uuid.uuid4())  # Start with a new session ID
        next_message_from_tester = config["INITIAL_PROMPT"]

        print("\n" + "=" * 50)
        print(f"🚀 Starting Autonomous Test Session")
        print(f"    Target Agent ID: {agent_id}")
        print(f"    Max Turns: {max_turns}")
        print(f"    Tester Model: {config['TESTER_BOT_MODEL']}")
        print("=" * 50 + "\n")

        # 3. Main conversation loop
        while current_turn < max_turns:
            print(f"\n--- Turn {current_turn + 1}/{max_turns} ---")

            # Tester bot sends its message
            print(f"\n🤖 Tester Bot says:\n{next_message_from_tester}")
            full_conversation_log.append({"role": "tester_bot", "content": next_message_from_tester})
            tester_chat_history.append({"role": "user", "content": next_message_from_tester})

            # Get CEAF agent's response
            ceaf_response, updated_session_id = await chat_with_agent(
                client, config["AURA_API_URL"], token, agent_id, next_message_from_tester, ceaf_session_id
            )
            ceaf_session_id = updated_session_id  # Keep the session ID consistent

            print(f"\n👤 CEAF Agent says:\n{ceaf_response}")
            full_conversation_log.append({"role": "ceaf_agent", "content": ceaf_response})
            tester_chat_history.append({"role": "assistant", "content": ceaf_response})

            current_turn += 1

            if current_turn >= max_turns:
                break

            # Generate the tester bot's next reply
            next_message_from_tester = await generate_tester_reply(
                tester_chat_history, config["TESTER_BOT_PERSONA"], config["TESTER_BOT_MODEL"]
            )

            # <<< NOVA LINHA ADICIONADA >>>
            print("⏳ Pausando por 2 segundos para dar tempo ao backend...")
            await asyncio.sleep(10)

    # 4. Save the conversation log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_run_{agent_id}_{timestamp}.json"

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(full_conversation_log, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("🏁 Test Session Complete")
    print(f"💾 Conversation log saved to: {filename}")
    print("=" * 50)


# --- Command-Line Interface ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an autonomous test session against a CEAF agent.")
    parser.add_argument("--agent-id", type=str,
                        help=f"The ID of the CEAF agent to test. Overrides default: {CONFIG['TARGET_AGENT_ID']}")
    parser.add_argument("--turns", type=int,
                        help=f"The number of conversational turns. Overrides default: {CONFIG['MAX_TURNS']}")
    parser.add_argument("--model", type=str,
                        help=f"The LLM model for the tester bot. Overrides default: {CONFIG['TESTER_BOT_MODEL']}")
    parser.add_argument("--prompt", type=str, help=f"The initial prompt to start the conversation. Overrides default.")

    args = parser.parse_args()

    # Update config with CLI arguments if provided
    if args.agent_id:
        CONFIG["TARGET_AGENT_ID"] = args.agent_id
    if args.turns:
        CONFIG["MAX_TURNS"] = args.turns
    if args.model:
        CONFIG["TESTER_BOT_MODEL"] = args.model
    if args.prompt:
        CONFIG["INITIAL_PROMPT"] = args.prompt

    if not CONFIG["TARGET_AGENT_ID"]:
        print("❌ Error: No target agent ID specified. Please set it in the script or use --agent-id.")
    elif not os.getenv("OPENROUTER_API_KEY"):
        print("❌ Error: OPENROUTER_API_KEY is not set in your .env file. The tester bot cannot function.")
    else:
        asyncio.run(run_test_session(CONFIG))