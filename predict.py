import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--title", required = True)
parser.add_argument("--body", required = True)
args_parsed = parser.parse_args()

args_parsed.body
args_parsed.title

with open(args_parsed.title, 'r') as f:
	title = f.readlines()

with open(args_parsed.body, 'r') as f:
	body = f.readlines()

#Read models

#Return predictions 

