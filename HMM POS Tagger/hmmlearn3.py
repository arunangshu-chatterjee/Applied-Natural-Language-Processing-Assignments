import sys
### Variable Declaration ###
wordTagPair = []
precedingTag = 'q0'
tagCount = {}
emissionProbability = {}
stateTransitionsCount = {}
stateCount = {}
transitionProbability = {}
currentTag = ''
line=''
tags = ''
############################



### Function to read Input Training Corpus File and store word/tag pairs per line ###
def read_training_corpus_file():
	inputTrgFileLoc = sys.argv[1]
	file = open(inputTrgFileLoc, 'r', encoding="utf8")
	for line in file:
		wordTagPair.append(line.split())
	file.close()
#####################################################################################



### Function to get the tags ###
def get_tags():
	for line in wordTagPair:
		count_tags(line)

################################



### Function to count the individual and total tags ###
def count_tags(line):
	precedingTag = 'q0'
	currentTag = ''
	# Tag count
	for words in line:
		# Logic to segregate the current tags
		i = 0
		for pos in range(len(words)-1,0,-1):
			if words[pos]=='/':
				i=pos
				break

		# Segregate the current tags
		currentTag = words[i+1:]

		if currentTag in tagCount:
			tagCount[currentTag] += 1
		else:
			tagCount[currentTag] = 1

		# Count state transitions
		if precedingTag == '' or currentTag == '':
			precedingTag = currentTag
			continue
		count_state_transitions(precedingTag, currentTag)

		# Count total word|tag
		count_emissions(words,line)

		# Count total tags
		count_tag_occurrence(precedingTag)

		precedingTag = currentTag
#######################################################



### Function to count transitions from previous to next state ###
def count_state_transitions(precedingTag,currentTag):
	transition = precedingTag + '==>' + currentTag
	if transition in stateTransitionsCount:
		stateTransitionsCount[transition] += 1
	else:
		stateTransitionsCount[transition] = 1
#################################################################



### Function to count total word|tag aka emissions ###
def count_emissions(words,line):
	if words in emissionProbability.keys():
		emissionProbability[words] += line.count(words)
	else:
		emissionProbability[words] = line.count(words)
######################################################



### Function to count total transitioning of label to label ###
def count_tag_occurrence(precedingTag):
	if precedingTag in stateCount.keys():
 		stateCount[precedingTag] += 1
	else:
		stateCount[precedingTag] = 1
################################################################



read_training_corpus_file()
get_tags()



# Store the tags in a variable to be written to a file
for item in tagCount.keys():
    tags = tags + '*,' + item

# Check if all states have been assigned count
for item in tagCount.keys():
	if item not in stateCount.keys():
		stateCount[item] = 0


### Function to write to a model file ###
def write_to_model_file():

	file = open("hmmmodel.txt", 'w+')
	file.write('No. of tags:' + str(len(tagCount)) + '\n')
	file.write('Tags=*>' + tags.strip(',') + '\n')
	file.write('State Count:\n')
	for item in stateCount.keys():
		file.write(item+'=*>' + str(stateCount[item])+ '\n')
	file.write('Transition Probability:\n')
	calculate_and_write_transition_probability(file)
	file.write('Emission Probability:\n')
	calculate_and_write_emission_probability(file)
	file.close()
#########################################



### Function to calculate transition probabilities with smoothing and write to the model file ###
def calculate_and_write_transition_probability(file):
	for item in stateTransitionsCount.keys():
		startTag = item.split('==>')[0]
		nextTag = item.split('==>')[1]
		#print(startTag + "->" + nextTag)
		if stateCount[startTag] > 0:
			transitionProbability[item] = (stateTransitionsCount[item] + 1) / (stateCount[startTag] + len(tagCount))
			if startTag == 'q0':
				var1 = str("{0:.5f}".format(transitionProbability[item]))
				output = 'START_STATE-' + nextTag + '=*>' + var1 + '\n'
			else:
				var2 = str("{0:.5f}".format(transitionProbability[item]))
				output = item + '=*>' + var2 + '\n'
			file.write(output)
			output = ''
#################################################################################################



### Function to calculate emission probabilitiesand write to the model file ###
def calculate_and_write_emission_probability(file):
	for item in emissionProbability.keys():
		# Logic to segregate the  tags
		i = 0
		for pos in range(len(item)-1,0,-1):
			if item[pos]=='/':
				i=pos
				break
		tag = item[i+1:]
		word = item[0:i]
		if tagCount[tag] > 0:
			emissionProbability[item] = emissionProbability[item] / tagCount[tag]
			if emissionProbability[item] > 1:
				emissionProbability[item] = 1
			var1 = str("{0:.5f}".format(emissionProbability[item]))
			output = 'P('+word+'|'+tag+') =*> ' + var1 + '\n'
			file.write(output)
###############################################################################



write_to_model_file()