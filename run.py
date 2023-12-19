import requests

def import_and_run_github_script(username, repo, script_path):
    # Construct the URL to the raw script on GitHub
    raw_url = f"https://raw.githubusercontent.com/{username}/{repo}/main/{script_path}"

    try:
        # Fetch the script content
        response = requests.get(raw_url)
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
    contents_url = f"https://api.github.com/repos/{username}/{repo}/contents/{folder_path}"

    try:
        # Fetch the contents of the folder from GitHub API
        response = requests.get(contents_url)
        response.raise_for_status()
        contents = response.json()

        # Filter and extract script paths
        script_paths = [item['path'] for item in contents if item['type'] == 'file' and item['path'].endswith('.py')]

        return script_paths

    except requests.exceptions.RequestException as e:
        print(f"Error fetching folder contents from GitHub: {e}")
        return []

if __name__ == "__main__":
    # Replace these values with your GitHub username and repository name
    github_username = "Squeegy"
    github_repo = "spritenator"
    scripts_folder_path = "scripts"

    # Get a list of script paths in the "scripts" folder and subfolders
    script_paths = get_scripts_in_folder(github_username, github_repo, scripts_folder_path)

    # Import and run each script in the specified order
    for script_path in script_paths:
        import_and_run_github_script(github_username, github_repo, f"{scripts_folder_path}/{script_path}")

    # Run the main script if it exists
    import_and_run_github_script(github_username, github_repo, "scripts/main.py")
