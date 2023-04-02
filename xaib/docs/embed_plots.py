import os
import sys
from bs4 import BeautifulSoup
from plotly.io import to_html

SCRIPT_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(BASE_DIR)
from xaib.evaluation.utils import visualize_results

RESULTS = ("feature_importance", "example_selection")

REPO_PATHS = [os.path.join(BASE_DIR, "xaib", "results", name) for name in RESULTS]
INDEX_PATH = os.path.join(BASE_DIR, "xaib", "docs", "build", "results")

for name, path in zip(RESULTS, REPO_PATHS):
    figs = visualize_results(path)

    for fig in figs:
        htmlfig = to_html(fig["fig"], include_plotlyjs="cdn")

        htmlfig = BeautifulSoup(htmlfig, "html.parser")

        with open(os.path.join(INDEX_PATH, name + ".html"), "r", encoding="UTF-8") as f:
            html = "\n".join(f.readlines())

        soup = BeautifulSoup(html, "html.parser")

        soup.findAll("footer", {"class": "md-footer"})[0].insert_before(
            BeautifulSoup(
                f'<article class="md-content__inner md-typeset" role="main"><p>{fig["desc"]}</p></article>',
                "html.parser",
            )
        )
        soup.findAll("footer", {"class": "md-footer"})[0].insert_before(htmlfig)

        with open(os.path.join(INDEX_PATH, name + ".html"), "w", encoding="UTF-8") as f:
            f.write(soup.prettify())
