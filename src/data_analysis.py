"""This module creates a report on the data.

A HTML report is automatically generated using the `sweetviz` library. It
performs an exploratory data analysis (EDA) and compares the training data with
the test data.

"""

import sweetviz
import pandas as pd

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

my_report = sweetviz.compare([train, "Train"], [test, "Test"], "Survived")

my_report.show_html("reports/report_on_processed.html")

