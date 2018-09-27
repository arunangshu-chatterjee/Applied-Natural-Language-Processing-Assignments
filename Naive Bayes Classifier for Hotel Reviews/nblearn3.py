import sys
import string


################### Variable Declaration ###################
label_dictionary = {}
stopWords = []
feature_list = []
feature_given_class_probabilities = {}
trueCount = 0
fakeCount = 0
posCount = 0
negCount = 0
prior_true = 0
prior_fake = 0
prior_pos = 0
prior_neg = 0
class_prior_probabilities = {}
unique_id = ''
training_file = ''
############################################################


################### Input File Parse ###################

def read_and_parse_input():
    global label_dictionary,feature_list, unique_id
    f = open(training_file, 'r', encoding="utf8")
    for line in f:
        token = line.split(" ")
        label_dictionary[token[0].strip()] = {'value1': token[1].strip(), 'value2': token[2].strip()}

        unique_id = line[:line.find(" ")]

        # Segregate the review text and store it into a variable
        review_text = " ".join(line.split()[3:])

        # Remove punctuations and convert to lower case
        review_text = "".join(l for l in review_text if l not in string.punctuation).lower()

        feature_list = create_feature_list(review_text)

        if unique_id in label_dictionary.keys():
            process_class_labels('value1', feature_list)
            process_class_labels('value2', feature_list)
    f.close()
########################################################


################### Create a list of features ###################
def create_feature_list(review_txt):
    global feature_list
    # Check if the feature is not a stop word and whether the feature is alphanumeric and store into a list
    feature_list = [word for word in review_txt.split() if ((word not in stopWords) and (word.isalnum()))]
    # Remove features which start or end with digits
    feature_list = " ".join(feature_list)
    feature_list = "".join([i for i in feature_list if not i.isdigit()])
    feature_list = feature_list.split()
    
    for item in feature_list:
        if len(item) < 3:
            feature_list.remove(item)
			
    return feature_list
#################################################################


################### Function to process classes ###################
def process_class_labels(label, f_list):
    global label_dictionary, prior_true, prior_fake, prior_pos, prior_neg
    if ((label_dictionary.get(unique_id)).get(label)) == 'True':
        add_to_feature_given_class_count(f_list, 0)
        prior_true += 1
    if ((label_dictionary.get(unique_id)).get(label)) == 'Fake':
        add_to_feature_given_class_count(f_list, 1)
        prior_fake += 1
    if ((label_dictionary.get(unique_id)).get(label)) == 'Pos':
        add_to_feature_given_class_count(f_list, 2)
        prior_pos += 1
    if ((label_dictionary.get(unique_id)).get(label)) == 'Neg':
        add_to_feature_given_class_count(f_list, 3)
        prior_neg += 1
####################################################################


################### Adding features to dictionary along with counts ###################
def add_to_feature_given_class_count(f_list, index):
    global feature_given_class_probabilities
    for feature in f_list:
        if feature in feature_given_class_probabilities.keys():
            feature_given_class_probabilities[feature][index] = int(feature_given_class_probabilities[feature][index]) + int('1')
        else:
            feature_given_class_probabilities[feature] = [0, 0, 0, 0]
            feature_given_class_probabilities[feature][index] = int(feature_given_class_probabilities[feature][index]) + int('1')
##################################################################


################### Function to remove low frequency words ###################
def remove_low_frequency_words():
    for key, value in feature_given_class_probabilities.items():
        if (sum(value)) <= 1:
            del feature_given_class_probabilities[key]
##############################################################################


################### Function to add smoothing ###################
def add_smoothing():
    for key, value in feature_given_class_probabilities.items():
        for i in range(0, 4):
            feature_given_class_probabilities[key][i] += 1
##############################################################


################### Function to get total count of each class ###################
def countLabels():
    global trueCount, fakeCount, posCount, negCount
    for key, value in feature_given_class_probabilities.items():
        trueCount += value[0]
        fakeCount += value[1]
        posCount += value[2]
        negCount += value[3]
#######################################################################


#################### Function to calculate calculating class prior probabilities ####################
def calculate_prior_probabilities():
    global prior_true, prior_fake, prior_pos, prior_neg, class_prior_probabilities
    total_probability1 = prior_true + prior_fake
    total_probability2 = prior_pos + prior_neg
    class_prior_probabilities['True'] = prior_true / float(total_probability1)
    class_prior_probabilities['Fake'] = prior_fake / float(total_probability1)
    class_prior_probabilities['Pos'] = prior_pos / float(total_probability2)
    class_prior_probabilities['Neg'] = prior_neg / float(total_probability2)
#####################################################################################################


################### Function to calculate probabilities ###################
def calculate_feature_given_class_probabilities():
    global posCount, negCount, trueCount, fakeCount, feature_given_class_probabilities
    for key, value in feature_given_class_probabilities.items():
        feature_given_class_probabilities[key][0] /= float(trueCount)
        feature_given_class_probabilities[key][1] /= float(fakeCount)
        feature_given_class_probabilities[key][2] /= float(posCount)
        feature_given_class_probabilities[key][3] /= float(negCount)
###########################################################################


################### Function to write to model file ###################
def write_to_model_file():
    global class_prior_probabilities, feature_given_class_probabilities
    with open("nbmodel.txt", 'w') as f:
        f.write("Class Prior Probabilities" + '\n')
        for key, value in class_prior_probabilities.items():
            f.write('%s:%s\n' % (key, value))
        f.write("Feature|Class Probabilities" + '\n')
        for key, value in feature_given_class_probabilities.items():
            f.write('%s:%s\n' % (key, value))
########################################################################


################### Function to read input file and generate model file ###################
def main():
    global training_file, stopWords
    training_file = "train-labeled.txt"
    stopWords = ['with', 'possible', 'j', 'thought', 'does', 'furthering', 'necessary', 'shouldnt', 'members',
                 'be', 'has', 'keeps', 'to', 'really', 'into', 'through', 'less', 'last', 'therefore', 'anywhere',
                 'right', 'this', 'seconds', 'anybody', 'just', 'seemed', 'among', 'many', 'until', 'newer', 'r',
                 'would', 'wouldnt', 'everybody', 'knows', 'interesting', 'ain', 'another', 'gives', 'those',
                 'yourself', 'very', 'sees', 'such', 'orders', 'full', 'arent', 'mustnt', 'isnt', 'thinks', 'going',
                 'whose', 'were', 'ever', 'turns', 'x', 'parted', 'finds', 'but', 'beings', 'h', 'whole', 'backing',
                 'rather', 'anyone', 'both', 'put', 'long', 'backed', 'presented', 'hers', 'dont', 'o', 's',
                 'smallest', 'been', 'am', 'give', 'some', 'areas', 'find', 'smaller', 'from', 'everything',
                 'problem', 'w', 'things', 'across', 'area', 'should', 'ends', 'when', 'doing', 'werent', 'he',
                 'like', 'every', 'quite', 'are', 'case', 'had', 'herself', 'any', 'hasnt', 'went', 'parts', 'faces',
                 'more', 'youngest', 'grouped', 'yet', 'great', 'if', 'away', 'could', 'took', 'enough', 'doesnt',
                 'order', 'against', 'we', 'good', 'several', 'ways', 'thatll', 'her', 'downs', 'upon', 'longest',
                 'kind', 'small', 'first', 'did', 'let', 'ending', 'wont', 'here', 'it', 'highest', 'and', 'have',
                 'best', 'made', 'says', 'nothing', 'youve', 'almost', 'keep', 'might', 'm', 'aint', 'your', 'seem',
                 'never', 'that', 'what', 'per', 'down', 'certainly', 'neednt', 'himself', 'needing', 'same',
                 'somebody', 'become', 'presenting', 'under', 'turn', 'state', 'know', 'after', 'different',
                 'either', 'which', 'cases', 'already', 'myself', 'came', 'saw', 'ours', 'places', 'q', 'while',
                 'out', 'nor', 'began', 'was', 'known', 'place', 'largely', 'early', 'toward', 'theirs', 'll',
                 'havent', 'within', 'ended', 'four', 'everywhere', 'younger', 'must', 'oldest', 'above',
                 'shes', 'again', 'z', 'group', 'opened', 'mightnt', 'puts', 'as', 'thus', 'side', 'knew', 'haven',
                 'ma', 'they', 'fact', 'fully', 'own', 'during', 'l', 'high', 'in', 'so', 'showing', 're', 'interest',
                 'v', 'certain', 'only', 'didnt', 'differ', 'old', 'ask', 'said', 'years', 'points', 'go', 'other',
                 'open', 'youd', 'next', 'thing', 'point', 'present', 'somewhere', 'asking', 'y', 'thoughts',
                 'whether', 'greatest', 'also', 'big', 'a', 'an', 'for', 'mostly', 'may', 'wants', 'over', 'once',
                 'having', 'than', 'get', 'cannot', 'youre', 'wasnt', 'together', 'his', 'see', 'at', 'p', 'all',
                 'important', 'us', 'mrs', 'evenly', 'now', 'why', 'sure', 'come', 'rooms', 'presents', 'states',
                 'everyone', 'wanted', 'shant', 'numbers', 'gets', 'shows', 'will', 'nobody', 'asked', 'backs',
                 'couldnt', 'can', 'there', 'e', 'man', 'goods', 'clearly', 'about', 'one', 'need', 'facts', 'off',
                 'needed', 'ourselves', 'non', 'c', 'uses', 'youll', 'lets', 'ordering', 'is', 'mr', 'interests',
                 'furthers', 'below', 'interested', 'someone', 'general', 'today', 'asks', 'turning', 'around',
                 'do', 'on', 'gave', 'something', 'way', 'who', 'furthered', 'opens', 'behind', 'pointed', 'downed',
                 'end', 'although', 'use', 'though', 'always', 'second', 'these', 'groups', 'by', 'works',
                 'differently', 'two', 'new', 'wanting', 'however', 'became', 'where', 'far', 'nowhere', 'hadnt',
                 'face', 'f', 'noone', 'men', 'before', 'd', 'work', 'generally', 'latest', 'making', 'seems',
                 'member', 'being', 'because', 'number', 've', 'becomes', 'newest', 'sides', 'part', 'yourselves',
                 'them', 'want', 'or', 'felt', 'most', 'showed', 'worked', 'my', 'room', 'seeming', 'often',
                 'perhaps', 'still', 'of', 'too', 'themselves', 'further', 'our', 'least', 'whom', 'no', 'anything',
                 'think', 'later', 'each', 'longer', 'between', 'downing', 'since', 'wells', 'make', 'itself',
                 'year', 'me', 'clear', 'yours', 'others', 'done', 'taken', 'you', 'along', 'three', 'young',
                 'higher', 'n', 'well', 'its', 'given', 'older', 'k', 't', 'working', 'without', 'u', 'she', 'much',
                 'got', 'opening', 'shouldve', 'their', 'pointing', 'g', 'shall', 'alone', 'how', 'the', 'say',
                 'parting', 'then', 'better', 'needs', 'greater', 'b', 'take', 'him', 'i', 'large', 'show', 'turned',
                 'used', 'back', 'ordered', 'not', 'likely', 'grouping', 'up', 'even', 'problems', 'few']
    read_and_parse_input()
    remove_low_frequency_words()
    add_smoothing()
    countLabels()
    calculate_prior_probabilities()
    calculate_feature_given_class_probabilities()
    write_to_model_file()
############################################################################################


if __name__ == "__main__":
    main()