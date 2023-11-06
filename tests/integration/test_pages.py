from glob import glob

from streamlit.testing.v1 import AppTest


def test_all_pages():
    pages = glob("pages/*.py")

    # Run all pages and check that they don't raise any exceptions
    for page in pages:
        at = AppTest.from_file(page)
        at.run(timeout=5)
        assert not at.exception
