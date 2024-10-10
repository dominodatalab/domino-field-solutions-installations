import requests
def get_domino_user_api_key(domino_host,oauth_token):
    domino_user_api_key = ''
    #First fetch the users id
    url = f'https://{domino_host}/v4/users/self'
    # Set up the headers with the Authorization Bearer token
    headers = {
        "Authorization": f"Bearer {oauth_token}"
    }

    # Make the GET request with the headers
    response = requests.get(url, headers=headers)

    # Check the response status code and content
    if response.status_code == 200:

        print("Request was successful")
        user_identifier = response.json()['id']
        print(response.json())
        url = f'https://{domino_host}/account/{user_identifier}/regenerateApiKey'
        print(url)
        response = requests.post(url, headers=headers)
        if response.status_code == 200:
            domino_user_api_key = response.json()['raw']
        else:
            print(f"Request failed with status code: {response.status_code}")
            print(response.text)

    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)
    return domino_user_api_key

domino_host='secureds53799.cs.domino.tech'
#For any user you can get a token valid for 5 mins making the following call in a workspace
#curl http://localhost:8899/access-token

# Service accounts have token with validity of 3 months

oauth_token="eyJhbGciOiJ..."


api_key = get_domino_user_api_key(domino_host,oauth_token)
print(api_key)