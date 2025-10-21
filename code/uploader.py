# uploader_v3.py
import requests
import json
import os
import argparse
import getpass
from pathlib import Path
from io import BytesIO

# --- Configuration ---
DEFAULT_API_URL = "http://localhost:8000"


def login(api_url: str, username: str, password: str) -> str | None:
    """
    Logs into the Aura API and returns an authentication token.
    Returns None if login fails. (This function remains unchanged)
    """
    login_endpoint = f"{api_url}/auth/login"
    login_payload = {
        "username": username,
        "password": password
    }
    headers = {"Content-Type": "application/json"}

    print(f"Attempting to log in as '{username}'...")
    try:
        response = requests.post(login_endpoint, json=login_payload, headers=headers)

        if response.status_code == 200:
            token = response.json().get("access_token")
            print("✅ Login successful!")
            return token
        else:
            print(f"❌ Login failed. Status code: {response.status_code}")
            print(f"   Reason: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"❌ A network error occurred while trying to log in: {e}")
        return None


def upload_agent_from_biography(api_url: str, auth_token: str, file_path: Path) -> bool:
    """
    Reads a biography JSON file and uploads it to create a new agent in Aura V3.
    Returns True on success, False on failure.
    """
    # V3 CHANGE: The endpoint URL has been simplified.
    upload_endpoint = f"{api_url}/agents/from-biography"
    headers = {"Authorization": f"Bearer {auth_token}"}

    print(f"\nProcessing file for creation: {file_path.name}")

    try:
        # 1. Read the raw file bytes
        with open(file_path, 'rb') as f:
            file_bytes = f.read()

        # V3 CHANGE: Removed the logic that modified 'is_public'.
        # In V3, agent creation and publishing are separate steps. An agent is created
        # as private by default. You would use the `/agents/{id}/publish` endpoint later.
        print("   - Note: Agents are created as private. Use the API's '/publish' endpoint to make them public templates.")

        # 2. Prepare the file for multipart/form-data upload
        files_payload = {
            'file': (file_path.name, BytesIO(file_bytes), 'application/json')
        }

        # 3. Make the request
        print(f"   - Uploading to create new agent...")
        response = requests.post(upload_endpoint, headers=headers, files=files_payload)

        # V3 CHANGE: The success status code is now 200 for this endpoint.
        # The response JSON also has a different structure.
        if response.status_code == 200:
            agent_id = response.json().get("agent_id")
            memories_injected = response.json().get("memories_injected", 0)
            print(f"✅ Success! Created agent (ID: {agent_id}) with {memories_injected} memories injected.")
            return True
        else:
            print(f"❌ Upload failed for {file_path.name}. Status code: {response.status_code}")
            print(f"   Reason: {response.text}")
            return False

    except FileNotFoundError:
        print(f"❌ Error: File not found at {file_path}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ A network error occurred during upload: {e}")
        return False
    except Exception as e:
        print(f"❌ An unexpected error occurred with {file_path.name}: {e}")
        return False


def add_memories_to_agent(api_url: str, auth_token: str, agent_id: str, file_path: Path) -> bool:
    """
    Reads a JSON file with a 'biography' or 'memories' list and adds those memories to an existing agent in Aura V3.
    Returns True on success, False on failure.
    """
    # V3 CHANGE: The endpoint URL is now '/memories/upload'.
    update_endpoint = f"{api_url}/agents/{agent_id}/memories/upload"
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }

    print(f"\nProcessing file to update agent: {agent_id}")
    print(f"   - Source file: {file_path.name}")

    try:
        # 1. Read the JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # V3 CHANGE: The payload expects a top-level key "memories".
        # We will gracefully handle if the old key "biography" is still used in the file.
        memories_list = data.get('memories')
        if memories_list is None:
            memories_list = data.get('biography') # Fallback for old format

        if memories_list is None or not isinstance(memories_list, list):
            print(f"❌ Error: The file '{file_path.name}' must contain a top-level key 'memories' or 'biography' with a list of memory objects.")
            return False

        # 3. Prepare the payload with the correct key.
        payload = {
            "memories": memories_list
        }

        # 4. Make the request
        print(f"   - Adding {len(payload['memories'])} new memories to agent...")
        response = requests.post(update_endpoint, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            successful = result.get('successful_uploads', 0)
            failed = result.get('failed_uploads', 0)
            print(f"✅ Success! Upload results for agent {agent_id}: {successful} successful, {failed} failed.")
            if result.get('errors'):
                print(f"   - Details: {result['errors']}")
            return True
        else:
            print(f"❌ Update failed for agent {agent_id}. Status code: {response.status_code}")
            print(f"   Reason: {response.text}")
            return False

    except FileNotFoundError:
        print(f"❌ Error: File not found at {file_path}")
        return False
    except json.JSONDecodeError:
        print(f"❌ Error: Could not parse JSON from {file_path.name}. Please check its format.")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ A network error occurred during update: {e}")
        return False
    except Exception as e:
        print(f"❌ An unexpected error occurred with {file_path.name}: {e}")
        return False


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description="Bulk upload or update agent memories on the Aura V3 platform."
    )
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--create",
        dest="directory",
        type=str,
        help="Path to the directory with .json files to CREATE new agents."
    )
    action_group.add_argument(
        "--add-memories-to",
        dest="agent_id",
        type=str,
        help="The ID of an EXISTING agent to add memories to."
    )

    parser.add_argument(
        "--file",
        type=str,
        help="The single .json file to use for the --add-memories-to action."
    )
    parser.add_argument(
        "--username",
        type=str,
        required=True,
        help="Your Aura AI username."
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_API_URL,
        help=f"The base URL of the Aura AI API (default: {DEFAULT_API_URL})."
    )
    args = parser.parse_args()

    if args.agent_id and not args.file:
        parser.error("--add-memories-to requires --file.")

    password = getpass.getpass(f"Enter password for user '{args.username}': ")

    token = login(args.url, args.username, password)
    if not token:
        return

    if args.directory:
        source_directory = Path(args.directory)
        if not source_directory.is_dir():
            print(f"❌ Error: The specified path '{args.directory}' is not a valid directory.")
            return

        json_files = list(source_directory.glob("*.json"))
        if not json_files:
            print(f"ℹ️ No .json files found in '{source_directory}'. Nothing to do.")
            return

        print(f"\nFound {len(json_files)} JSON file(s) to process for CREATION.")
        success_count = 0
        failure_count = 0
        for file_path in json_files:
            if upload_agent_from_biography(args.url, token, file_path):
                success_count += 1
            else:
                failure_count += 1

        print("\n" + "=" * 30)
        print("      UPLOAD SUMMARY (CREATE)")
        print("=" * 30)
        print(f"Successful uploads: {success_count}")
        print(f"Failed uploads:     {failure_count}")
        print("=" * 30)

    elif args.agent_id:
        file_path = Path(args.file)
        if not file_path.is_file():
            print(f"❌ Error: The specified file '{args.file}' does not exist.")
            return

        add_memories_to_agent(args.url, token, args.agent_id, file_path)


if __name__ == "__main__":
    main()