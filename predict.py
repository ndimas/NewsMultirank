import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--title", required = True)
parser.add_argument("--body", required = True)
args_parsed = parser.parse_args()

args_parsed.body

args_parsed.title
print(args_parsed)