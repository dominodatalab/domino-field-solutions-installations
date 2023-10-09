from typing import Dict,List
import json
import requests
import os
import yaml
import logging

logger = logging.getLogger("domsed_client")
lvl = logging.getLevelName(os.environ.get("LOG_LEVEL", "WARNING"))
logging.basicConfig(
    level=lvl,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("werkzeug")
log.setLevel(logging.WARNING)
default_api_endpoint = 'http://extendedapi-svc.domino-field.svc.cluster.local'
api_endpoint = os.environ.get('DOMSED_WEBCLIENT_ENDPOINT',default_api_endpoint)


def list():
    auth_key = requests.get(os.environ.get('DOMINO_API_PROXY') + '/access-token').text
    listing_url = f'{api_endpoint}/mutation/list'
    print('Listing Mutations\n')
    resp = requests.get(listing_url, headers={"Authorization": f"Bearer {auth_key}"})

    if(resp.status_code==200):
        mutations_json:Dict = resp.json()
        mutations:List = mutations_json['items']
        for m in mutations:
            mutation_name = m["metadata"]["name"]
            print(f"\t{mutation_name}")
    else:
        logger.warning('Error Listing Mutations')
        logger.warning('Status Code :' + str(resp.status_code))
        logger.warning('Error :' + str(resp.text))

def get(mutation_name):
    auth_key = requests.get(os.environ.get('DOMINO_API_PROXY') + '/access-token').text
    get_url = f'{api_endpoint}/mutation/{mutation_name}'
    logger.warning(f'Get Mutation{mutation_name}')
    resp = requests.get(get_url,  headers={"Authorization": f"Bearer {auth_key}"})
    if(resp.status_code==200):
        print(f'Printing Mutation: {mutation_name}\n')
        print(resp.text)
    else:
        logger.warning('Error Listing Mutations')
        logger.warning('Status Code :' + str(resp.status_code))
        logger.warning('Error :' + str(resp.text))

def delete(mutation_name):
    auth_key = requests.get(os.environ.get('DOMINO_API_PROXY') + '/access-token').text
    delete_url = f'{api_endpoint}/mutation/{mutation_name}'
    logger.warning(f'Deleting Mutation{mutation_name}')
    resp = requests.delete(delete_url, headers={"Authorization": f"Bearer {auth_key}"})
    if(resp.status_code==200):
        print(f'Deleted Mutation {mutation_name}')
        logger.warning(resp.text)
    else:
        logger.warning('Error Publishing Mutation')
        logger.warning('Status Code :' + str(resp.status_code))
        logger.warning('Error :' + str(resp.text))

def apply_file(mutation_file):
    if(mutation_file.endswith(".yaml") or mutation_file.endswith(".yml") ):
        with open(mutation_file) as f:
            configuration = yaml.safe_load(f)
            mutation = json.loads(json.dumps(configuration))
    elif (mutation_file.endswith(".json")):
        with open(mutation_file) as f:
            mutation = json.loads(f.readlines())
    else:
        logger.warning('Invalid file format. Must be YAML or JSON')
        exit(1)
    apply(mutation)


def apply(mutation):
    auth_key = requests.get(os.environ.get('DOMINO_API_PROXY') + '/access-token').text
    publish_url = f'{api_endpoint}/mutation/apply'
    logger.warning('Publishing Mutation To Domsed')
    resp = requests.post(publish_url, json=mutation, headers={"Authorization": f"Bearer {auth_key}"})
    if(resp.status_code==200):
        print('Applied Mutation')
        print(resp.text)
    else:
        logger.warning('Error Publishing Mutation')
        logger.warning('Status Code :' + str(resp.status_code))
        logger.warning('Error :' + str(resp.text))