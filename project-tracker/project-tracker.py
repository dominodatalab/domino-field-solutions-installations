import os
import requests
import json

DOMINO_API_BASE_URL = os.getenv(
    "DOMINO_API_BASE_URL", "https://<domino-url>/api/projects/beta/projects"
)
PROJECT_DATASET_PATH = "/path/to/previous_projects.json"
STATS_FILE_PATH = "dominostats.json"


def fetch_projects():
    headers = {"X-Domino-Api-Key": os.getenv("DOMINO_USER_API_KEY")}
    response = requests.get(DOMINO_API_BASE_URL, headers=headers)
    if response.status_code == 200:
        return response.json()["projects"]
    else:
        print(f"Failed to fetch projects. Status code: {response.status_code}")
        return None


# Load previous projects from dataset
def load_previous_projects():
    if os.path.exists(PROJECT_DATASET_PATH):
        with open(PROJECT_DATASET_PATH, "r") as file:
            return json.load(file)["projects"]
    else:
        return []


# Save current projects to dataset
def save_current_projects(projects):
    with open(PROJECT_DATASET_PATH, "w") as file:
        json.dump({"projects": projects}, file)


def log_diagnostics(new_projects_count, deleted_projects_count):
    diagnostics = {
        "New Projects Count": new_projects_count,
        "Deleted Projects Count": deleted_projects_count,
    }
    with open(STATS_FILE_PATH, "w") as f:
        json.dump(diagnostics, f)


def main():
    current_projects = fetch_projects()
    if current_projects is None:
        return

    previous_projects = load_previous_projects()

    current_project_ids = {project["id"] for project in current_projects}
    previous_project_ids = {project["id"] for project in previous_projects}

    new_project_ids = current_project_ids - previous_project_ids
    deleted_project_ids = previous_project_ids - current_project_ids

    new_projects = [
        project for project in current_projects if project["id"] in new_project_ids
    ]
    deleted_projects = [
        project for project in previous_projects if project["id"] in deleted_project_ids
    ]

    save_current_projects(current_projects)

    # Log diagnostic statistics
    log_diagnostics(len(new_projects), len(deleted_projects))

    if new_projects:
        print("New projects created since the last run:")
        for project in new_projects:
            print(
                f"- Name: {project['name']}, ID: {project['id']}, Owner: {project['ownerUsername']}"
            )
    else:
        print("No new projects.")

    if deleted_projects:
        print("Projects deleted since the last run:")
        for project in deleted_projects:
            print(
                f"- Name: {project['name']}, ID: {project['id']}, Owner: {project['ownerUsername']}"
            )
    else:
        print("No projects deleted since the last run.")


if __name__ == "__main__":
    main()
