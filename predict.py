import argparse
from pip._vendor.requests import Response

parser = argparse.ArgumentParser()

args_parsed = parser.parse_args()

print(args_parsed.body)
print(args_parsed.title)


class Test:
    def test(self):
        0


def predict():
    response = Response()
    response["Cache-Control"] = 'no-cache'

# Read models

# Return predictions
