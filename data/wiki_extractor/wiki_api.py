import requests
from dotenv import load_dotenv
import os
from pathlib import Path
from bs4 import BeautifulSoup

# Environment variables
load_dotenv()
USER_AGENT_EMAIL = os.getenv('USER_AGENT_EMAIL')
USER_AGENT_NAME = os.getenv('USER_AGENT_NAME')
ACCESS_TOKEN = os.getenv('ACCESS_TOKEN')
CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')

# API variables
base_url = 'https://api.wikimedia.org/core/v1'

endpoints = {
    'search': 'search/page',
    'resource_info': 'file',
}

HEADERS = {
  'Authorization': f'Bearer {ACCESS_TOKEN}',
'User-Agent': f'{USER_AGENT_NAME} ({USER_AGENT_EMAIL})'
}

DEFAULT_LANGUAGE = 'en'


def get_resource_info(filename, project='commons'):
    """Get information about a resource from the Wikimedia Commons API"""
    url = __make_request_data(endpoint=endpoints['resource_info'], project=project)
    url += f'/{filename}'

    info_resource = requests.get(url, headers=HEADERS)
    return info_resource.json()


def download_resource(filename, *, resource_info=None, name=None, project='commons', destination_path='.'):
    """Download a resource from the Wikimedia Commons API"""

    if resource_info is None:
        resource_info = get_resource_info(filename, project=project)

    # get preferred field from info_resource
    image_url = resource_info['preferred']['url']
    image = requests.get(image_url, headers=HEADERS)

    # set name if not provided
    if name is None:
        name = filename
    
    # set destination path
    destination = Path(destination_path)
    destination.mkdir(parents=True, exist_ok=True)

    # save image
    with open(destination / name, 'wb') as f:
        f.write(image.content)


def get_resource_description(url):
    """Get the description of a resource from the Wikimedia Commons using BeautifulSoup and the url of the description page"""

    # get page from url and parse it
    page = requests.get(url)
    soup: BeautifulSoup = BeautifulSoup(page.content, 'html.parser')

    # get description node
    description_node = soup.find("td", class_="description")

    # if there is no description node, return None
    if description_node is None:
        return None

    # get language nodes from description node
    language_nodes = description_node.find_all("div", class_="description")

    # if there are no language nodes, return the description node text as it is
    # TODO: check if this is correct, since the text could be in other language than the default one
    if len(language_nodes) == 0:
        return description_node.text.strip()

    # if there are language nodes, return the text of the node with the default language
    for language_node in language_nodes:
        if language_node['lang'] == DEFAULT_LANGUAGE:
            lang_span = language_node.find("span", class_="language")
            if lang_span is None:
                return None
            lang_span.string = ""
            return language_node.text.strip()

    # if there are language nodes but none of them is the default language, return None
    return None

    

def search(query, *,  number_of_results=1, project='wikipedia', language=DEFAULT_LANGUAGE):
    """Search for a query on the Wikimedia Commons API"""
    url = __make_request_data(endpoint=endpoints['search'], project=project, language=language)

    parameters = {
        'q': query,
        'limit': number_of_results,
        'type': 'image'
    }
    response = requests.get(url, headers=HEADERS, params=parameters)
    return response.json()


# Helper private functions
def __make_request_data(*, endpoint, project, language="") -> str:

    if language != "":
        language = f'/{language}'

    url = f'{base_url}/{project}{language}/{endpoint}'
    return url