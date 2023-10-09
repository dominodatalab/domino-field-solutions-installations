import requests
import os
import json
import sys
import shutil


DOMINO_API_KEY= os.environ['DOMINO_USER_API_KEY']
DOMINO_URL = os.environ['DOMINO_API_HOST']
GET_USERS_ENDPOINT = 'v4/users'
GET_PROJECTS_ENDPOINT = 'v4/projects'

def get_all_projects():
    headers = {'X-Domino-Api-Key': DOMINO_API_KEY}
    url = os.path.join(DOMINO_URL,GET_PROJECTS_ENDPOINT)
    ret = requests.get(url, headers=headers)
    projects = ret.json()
    return projects

def get_all_project_names():
    headers = {'X-Domino-Api-Key': DOMINO_API_KEY}
    url = os.path.join(DOMINO_URL,GET_PROJECTS_ENDPOINT)
    ret = requests.get(url, headers=headers)
    projects = ret.json()
    print(projects)
    lst = []
    for p in projects:
        lst.append(p['ownerUsername']+ "/" + p['name'])
    return lst

def get_users_list():
    headers = {'X-Domino-Api-Key': DOMINO_API_KEY}
    url = os.path.join(DOMINO_URL,GET_USERS_ENDPOINT)
    ret = requests.get(url, headers=headers)
    users = ret.json()
    return users


def get_user_names_list():
    headers = {'X-Domino-Api-Key': DOMINO_API_KEY}
    url = os.path.join(DOMINO_URL,GET_USERS_ENDPOINT)
    ret = requests.get(url, headers=headers)
    users = ret.json()
    lst = []
    for u in users:
        lst.append(u['userName'])
    return lst