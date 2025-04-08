import sys
import time
import requests

def main(base_url: str):
    # Construct the endpoint URL for creating a character.
    create_endpoint = f"{base_url.rstrip('/')}/create-character"
    
    print(f"Posting to create character at {create_endpoint}...")
    try:
        response = requests.post(create_endpoint)
        response.raise_for_status()  # Raise an error for non-200 responses
    except requests.RequestException as e:
        print(f"Error calling create-character endpoint: {e}")
        return

    data = response.json()
    job_id = data.get("job_id")
    status = data.get("status")
    
    if not job_id:
        print("No job_id returned from create-character call.")
        return

    print(f"Job scheduled with ID: {job_id} (Status: {status})")

    # Poll the job status until the job is either complete or failed.
    status_endpoint = f"{base_url.rstrip('/')}/character-status/{job_id}"
    while True:
        try:
            status_response = requests.get(status_endpoint)
            status_response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error polling status: {e}")
            break

        status_data = status_response.json()
        current_status = status_data.get("status")
        print(f"Job {job_id} status: {current_status}")

        if current_status == "completed":
            result = status_data.get("result")
            print("Character creation completed successfully!")
            print("Result:")
            print(result)
            break
        elif current_status == "failed":
            error = status_data.get("error")
            print("Character creation failed!")
            print(f"Error: {error}")
            break

        time.sleep(1)  # Wait before polling again

if __name__ == "__main__":
    # Allow the user to pass a URL as a command-line argument.
    # Example: python testapi.py http://127.0.0.1:8000
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    main(base_url)
