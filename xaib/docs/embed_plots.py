import os
import sys
from bs4 import BeautifulSoup

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))
from xaib.evaluation.utils import visualize_results

REPO_PATH = ''
INDEX_PATH = ''

fig = visualize_results(REPO_PATH, write=False)
htmlfig = fig.to_html(include_plotlyjs='cdn')

htmlfig = BeautifulSoup(htmlfig, "html.parser")

with open(INDEX_PATH, 'r') as f:
    html = '\n'.join(f.readlines())

soup = BeautifulSoup(html, "html.parser")

soup.findAll('footer', {'class': 'md-footer'})[0].insert_before(htmlfig)

with open(INDEX_PATH, 'w', encoding='UTF-8') as f:
    f.write(soup.prettify())
