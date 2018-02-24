import argparse

parser = argparse.ArgumentParser()

args_parsed = parser.parse_args()

print(args_parsed.body)
print(args_parsed.title)

#with open(args_parsed.title, 'r') as f:
	#title = f.readlines()

#with open(args_parsed.body, 'r') as f:
	#body = f.readlines()


#Read models

#Return predictions 

