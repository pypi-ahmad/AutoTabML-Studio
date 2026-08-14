from streamlit.testing.v1 import AppTest


def test_main_navigation_renders_without_streamlit_errors():
    app = AppTest.from_file("app/main.py")
    app.run(timeout=30)
    assert not app.exception
