import sys
import math

### Variable Declaration ###
transitionProbability = {}
emissionProbability = {}
tagStateList = []
tagStateDict = {}
stateCount = {}
wordsInModel = {}
line = ''
filecontents = ''
totalTags = 0
marker = -1
############################



### Function to read model file and process information ###
def read_and_process_model_file():
	global totalTags
	global marker
	global tagStateList
	global tagStateList
	global wordsInModel
	global stateCount
	global transitionProbability
	global emissionProbability
	modelFile = open('hmmmodel.txt','r', encoding = 'utf-8')
	lineCount = 0
	for line in modelFile:
		# Store number of tags
		if lineCount == 0:
			lineCount += 1
			totalTags = int(line.split(':')[1])
			continue
		# Store list of tags
		if lineCount == 1:
			lineCount += 1
			tagSet = line.split('=*>')[1]
			tagSet = tagSet.strip('\n')
			tagSet = tagSet.split('*,')
			tagSet.remove('')
			tagStateList = tagSet
			continue
		if line == "Emission Probability:\n":
			marker = 0
			continue
		if line == "Transition Probability:\n":
			marker = 1
			continue
		if line == "State Count:\n":
			marker = 2
			continue
		#assign_values_according_to_marker(marker)
		
		if marker == 0:
			data = line.split('=*>')
			var1 = data[0]
			var1 = var1[2:len(var1) - 2]
			corpusWord = var1.split('|')[0]
			wordsInModel[corpusWord] = 1
			emissionProbability[var1] = "{0:.5f}".format(float(data[1].strip('\n')))
		if marker == 1:
			data = line.split('=*>')
			var2 = data[0]
			var2 = var2.replace('START_STATE','q0')
			transitionProbability[var2] = "{0:.5f}".format(float(data[1].strip('\n')))
		if marker == 2:
			var3 = line.split('=*>')
			stateCount[var3[0]] = int(var3[1].strip('\n'))
###########################################################



### Viterbi Algorithm for Decoding ###
def viterbi_algorithm(line):
	global totalTags
	global marker
	global tagStateList
	global tagStateList
	global wordsInModel
	global stateCount
	global transitionProbability
	global emissionProbability
	global filecontents

	score = 0
	wordSequence = line.split(' ')
	T = len(wordSequence)
	viterbi = [[0 for x in range(T)] for y in range(totalTags+1)]
	backtrack = [[0 for x in range(T)] for y in range(totalTags+1)]

	# Initialization Step
	for item1 in tagStateDict.keys():
		emissionKey = wordSequence[0] + '|' + tagStateDict[item1]
		if wordSequence[0] not in wordsInModel.keys():
			probabilityOfEmission = 1.0
		elif emissionKey not in emissionProbability.keys():
			probabilityOfEmission = 0.0
		else:
			probabilityOfEmission = float(emissionProbability[emissionKey])
		transitionKey = 'q0-'+ tagStateDict[item1]
		if transitionKey not in transitionProbability.keys():
			probabilityOfTransition = float(1 / (int(stateCount['q0']) + totalTags))
		else:
			probabilityOfTransition = float(transitionProbability[transitionKey])
		viterbi[item1][0] = probabilityOfTransition * probabilityOfEmission
		backtrack[item1][0] = 0

	# For the next states do
	for item2 in range(1,T):
		for toState in tagStateDict.keys():
			for fromState in tagStateDict.keys():
				emissionKey = wordSequence[item2] + '|' + tagStateDict[toState]
				if wordSequence[item2] not in wordsInModel.keys():
					probabilityOfEmission = 1
				elif emissionKey not in emissionProbability.keys():
					probabilityOfEmission = 0
				else:
					probabilityOfEmission =  float(emissionProbability[emissionKey])
				transitionKey = tagStateDict[fromState] + '==>' + tagStateDict[toState]
				if transitionKey not in transitionProbability.keys():
					probabilityOfTransition =  float(1 / (int(stateCount[tagStateDict[fromState]]) + totalTags))
				else:
					probabilityOfTransition = float(transitionProbability[transitionKey])
				score = float(viterbi[fromState][item2-1]) * probabilityOfTransition * probabilityOfEmission
				if score > float(viterbi[toState][item2]):
					viterbi[toState][item2] = score
					backtrack[toState][item2] = fromState
				else:
					continue
	
	best = 0
	for item3 in tagStateDict.keys():
		if viterbi[item3][T-1] > viterbi[best][T-1]:
			best = item3
	ouput_line = wordSequence[T-1]+'/'+tagStateDict[best] + ' '
	for item4 in range(T-1, 0, -1):
		best = backtrack[best][item4]
		ouput_line = wordSequence[item4 - 1] + '/' + tagStateDict[best] + ' ' + ouput_line
	filecontents += ouput_line + '\n'
	return filecontents
######################################



### Function to read raw input file and generate tagged output file ###
def main():
	global tagStateList
	global tagStateDict
	global filecontents
	read_and_process_model_file()
	i = 0
	for tagName in tagStateList:
		tagStateDict[i] = tagName
		i += 1
	inputRawFileLoc = "catalan_corpus_dev_raw.txt" #sys.argv[1]
	inputFile = open(inputRawFileLoc, 'r', encoding='utf-8')
	#filecontents = ''
	outputFile = open('hmmoutput.txt','w')
	for line in inputFile:
		filecontents = viterbi_algorithm(line.strip())
	outputFile.write(filecontents)
	inputFile.close()
	outputFile.close()
#######################################################################

if __name__ == "__main__":
	main()