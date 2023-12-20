import argparse
import requests
import subprocess
import time


headers = {
    'Cache-Control': 'no-cache, no-store, must-revalidate',
    'Pragma': 'no-cache',
    'Expires': '0'
}

def import_and_run_github_script(username, repo, script_path):
    # Construct the URL to the raw script on GitHub
    raw_url = f"https://api.github.com/repos/{username}/{repo}/main/{script_path}?ref=main"

    try:
        # Fetch the script content
        response = get_request(raw_url)
        response.raise_for_status()
        script_code = response.text

        # Execute the script
        exec(script_code, globals())

    except requests.exceptions.RequestException as e:
        print(f"Error fetching script from GitHub: {e}")
    except Exception as e:
        print(f"Error executing script: {e}")

def get_scripts_in_folder(username, repo, folder_path):
    # Construct the URL to fetch the contents of the folder
    contents_url = f"https://api.github.com/repos/{username}/{repo}/contents/{folder_path}?ref=main"

    try:
        # Fetch the contents of the folder from GitHub API
        response = get_request(contents_url)
        response.raise_for_status()
        contents = response.json()

        # Filter and extract script paths
        script_paths = [item['path'] for item in contents if item['type'] == 'file' and item['path'].endswith('.py')]

        return script_paths

    except requests.exceptions.RequestException as e:
        print(f"Error fetching folder contents from GitHub: {e}")
        return []
    
def pull_and_install(username, repo):
    # Function to handle the --pull argument
    pipfile_url = f"https://api.github.com/repos/{username}/{repo}/main/Pipfile?ref=main"
    pipfile_lock_url = f"https://api.github.com/repos/{username}/{repo}/main/Pipfile.lock?ref=main"

    for file_url in [pipfile_url, pipfile_lock_url]:
        response = get_request(file_url)
        if response.status_code == 200:
            with open(file_url.split('/')[-1], 'w') as file:
                file.write(response.text)

    subprocess.run(["pipenv", "install"], check=True)

def get_request(url):
    """
    Makes a GET request to the specified URL with predefined headers.
    Returns the response object.

    :param url: URL to which the GET request is to be sent.
    :return: requests.Response object
    """
    try:
        response = requests.get(url, headers=headers)
        if 'Cache-Control' in response.headers:
            print("Cache-Control header is present")
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
        return response
    except requests.exceptions.RequestException as e:
        print(f"Error making GET request to {url}: {e}")
        return None

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
