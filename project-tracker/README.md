# Project Tracker Script

## Overview

This script is designed to track projects on a Domino instance. It compares the current list of projects with a previously stored list to identify new and deleted projects. The results are logged and displayed to the user.

## Requirements

- Python 3.x
- `requests` library

## Setup

1. Clone the repository containing this script.
2. Ensure the required environment variables are set:
    - `DOMINO_API_BASE_URL`: The base URL for the Domino API (default is `https://domino-url/api/projects/beta/projects`).
    - `DOMINO_USER_API_KEY`: The API key for authenticating with the Domino server.
3. Adjust the paths for the dataset and stats file if necessary:
    - `PROJECT_DATASET_PATH`: Path to store the JSON file containing the previous project list (default is `/domino/datasets/local/Notification_dev/previous_projects.json`).
    - `STATS_FILE_PATH`: Path to store the JSON file containing diagnostic statistics (default is `dominostats.json`).

## Usage

This script is intended to be run as a scheduled job in Domino. The frequency can be set to whatever the user desires.

1. Run the script using Python:

    ```bash
    python project_tracker.py
    ```

2. The script will:
    - Fetch the current list of projects from the Domino server.
    - Load the previous list of projects from the specified dataset path.
    - Identify new and deleted projects by comparing the current and previous lists.
    - Save the current list of projects for future comparison.
    - Log diagnostic statistics to the specified stats file.
    - Print details of new and deleted projects to the console.

## Functions

### `fetch_projects()`

Fetches the current list of projects from the Domino server.

- **Returns**: A list of current projects if the request is successful, otherwise `None`.

### `load_previous_projects()`

Loads the previous list of projects from the specified dataset path.

- **Returns**: A list of previous projects.

### `save_current_projects(projects)`

Saves the current list of projects to the specified dataset path.

- **Parameters**: 
    - `projects`: The current list of projects to be saved.

### `log_diagnostics(new_projects_count, deleted_projects_count)`

Logs diagnostic statistics to the specified stats file.

- **Parameters**: 
    - `new_projects_count`: The number of new projects.
    - `deleted_projects_count`: The number of deleted projects.

### `main()`

Main function to execute the script's logic:
- Fetches current projects.
- Loads previous projects.
- Identifies new and deleted projects.
- Saves the current projects.
- Logs diagnostics.
- Prints details of new and deleted projects to the console.

## Limitation

The script cannot accurately detect projects that were created and deleted between script runs, resulting in missed tracking of such events.

## Example Output

```plaintext
New projects created since the last run:
- Name: Project A, ID: 123, Owner: user1
- Name: Project B, ID: 456, Owner: user2

No projects deleted since the last run.
```