import argparse
import requests
import subprocess
import time
import base64
import json



headers = {
    'Cache-Control': 'no-cache, no-store, must-revalidate',
    'Pragma': 'no-cache',
    'Expires': '0'
}

def import_and_run_github_script(username, repo, script_path):
    # Construct the URL to the raw script on GitHub
    raw_url = f"https://api.github.com/repos/{username}/{repo}/contents/{script_path}?ref=main"

    try:
        # Fetch the script content
        response = get_request(raw_url)

        # Execute the script
        exec(response, globals())
    except requests.exceptions.RequestException as e:
        print(f"Error fetching script from GitHub: {e}")
    except Exception as e:
        print(f"Error executing script: {e}")

def fetch_scripts(username, repo, path):
    contents_url = f"https://api.github.com/repos/{username}/{repo}/contents/{path}?ref=main"
    try:
        response = get_request(contents_url)
        scripts = []

        for item in response:
            if item['type'] == 'file' and item['path'].endswith('.py'):
                scripts.append(item['path'])
            elif item['type'] == 'dir':
                scripts.extend(fetch_scripts(username, repo, item['path']))

        return scripts

    except requests.exceptions.RequestException as e:
        print(f"Error fetching folder contents from GitHub: {e}")
        return []

def get_scripts_in_folder(username, repo, folder_path):
    return fetch_scripts(username, repo, folder_path)
    
def pull_and_install(username, repo):
    # Function to handle the --pull argument
    pipfile_url = f"https://api.github.com/repos/{username}/{repo}/contents/Pipfile"
    pipfile_lock_url = f"https://api.github.com/repos/{username}/{repo}/contents/Pipfile.lock"

    for file_url in [pipfile_url, pipfile_lock_url]:
        response = get_request(file_url)
        if response is not None:
            with open(file_url.split('/')[-1], 'w') as file:
                file.write(response)

    subprocess.run(["pipenv", "install"], check=True)

def get_request(url):
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad requests

        data = response.json()

        # Check if 'content' key exists in response
        if 'content' in data:
            decoded_content = base64.b64decode(data['content']).decode('utf-8')
            return decoded_content
        else:
            return data

    except requests.exceptions.HTTPError as errh:
        print(f"Http Error: {errh}")
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"Request Error: {err}")


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Run scripts from a GitHub repository.")
    parser.add_argument("--pull", action="store_true", help="Download Pipfile and Pipfile.lock and run pipenv install")
    args = parser.parse_args()

    # Replace these values with your GitHub username and repository name
    github_username = "Squeegy"
    github_repo = "spritenator"
    scripts_folder_path = "scripts"

    if args.pull:
        pull_and_install(github_username, github_repo)

    # Get a list of script paths in the "scripts" folder and subfolders
    script_paths = get_scripts_in_folder(github_username, github_repo, scripts_folder_path)

    # Import and run each script in the specified order
    for script_path in script_paths:
        import_and_run_github_script(github_username, github_repo, script_path)

    # Run the main script if it exists
    import_and_run_github_script(github_username, github_repo, "main.py")