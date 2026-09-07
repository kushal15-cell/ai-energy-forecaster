import unittest

from streamlit.testing.v1 import AppTest


class AppTests(unittest.TestCase):
    def test_home_and_model_card(self):
        for page in ["app.py", "pages/3_Model_Info.py"]:
            app = AppTest.from_file(page, default_timeout=60).run()
            self.assertEqual(len(app.exception), 0)

    def test_forecast_submission(self):
        app = AppTest.from_file("pages/1_Forecast.py", default_timeout=60).run()
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 3)
        self.assertIn("forecast_result", app.session_state)

    def test_budget_submission(self):
        app = AppTest.from_file("pages/2_Budget_Mode.py", default_timeout=60).run()
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertGreater(len(app.dataframe), 0)


if __name__ == "__main__":
    unittest.main()
