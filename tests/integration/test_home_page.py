from streamlit.testing.v1 import AppTest


def test_app_home_page():
    # Run the app and check that it doesn't raise any exceptions
    at = AppTest.from_file("abm.py")
    at.run(timeout=10)
    assert not at.exception
