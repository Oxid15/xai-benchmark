import os
import sys
from bs4 import BeautifulSoup

SCRIPT_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(BASE_DIR)
from xaib.evaluation.utils import visualize_results

REPO_PATHS = [os.path.join(BASE_DIR, 'xaib', 'results', name)
              for name in ('feature_importance', 'example_selection')]
INDEX_PATH = os.path.join(BASE_DIR, 'xaib', 'docs', 'build', 'index.html')

for path in REPO_PATHS:
    fig = visualize_results(path, write=False)
    htmlfig = fig.to_html(include_plotlyjs='cdn')

    htmlfig = BeautifulSoup(htmlfig, "html.parser")

    with open(INDEX_PATH, 'r', encoding='UTF-8') as f:
        html = '\n'.join(f.readlines())

    soup = BeautifulSoup(html, "html.parser")

    soup.findAll('footer', {'class': 'md-footer'})[0].insert_before(htmlfig)

    with open(INDEX_PATH, 'w', encoding='UTF-8') as f:
        f.write(soup.prettify())
