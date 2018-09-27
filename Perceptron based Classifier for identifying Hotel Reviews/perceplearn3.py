import sys
import string
import copy


################### Variable Declaration ###################
tag_details = []
word_list = []
feature_list = []
stop_words = []
vanilla_pos_neg_weight_vector = {}
vanilla_true_fake_weight_vector = {}
averaged_pos_neg_weight_vector = {}
averaged_true_fake_weight_vector = {}
cached_pos_neg_weight_vector = {}
cached_true_fake_weight_vector = {}
training_file = ''
line_count = 0
############################################################


################### Function to parse input file ###################
def read_and_parse_input():
    global word_list, line_count, stop_words, training_file
    f = open(training_file, 'r', encoding="utf8")
    for line in f:
        token = line.split(" ")
        tag_details.append(token[:3])
        # Segregate the review text and store it into a variable
        review_text = " ".join(line.split()[3:])
        # Ensure proper spacing
        review_text = insert_space(review_text)
        # Remove punctuations and convert to lower case
        review_text = "".join(l for l in review_text if l not in string.punctuation).lower()
        # Check if the word is not in stop words and alpha numeric
        word_list = [word for word in review_text.split() if ((word not in stop_words) and (word.isalnum()))]
        # Remove features which start or end with digits
        word_list = " ".join(word_list)
        word_list = "".join([i for i in word_list if not i.isdigit()])
        word_list = word_list.split()
        # Remove one or two letter words
        for item in word_list:
            if len(item) <= 3:
                    word_list.remove(item)
        feature_list.append(word_list)
    f.close()
####################################################################


################### Function to add spaces after punctuation if none exists ###################
def insert_space(strng):
    for i in range(0, len(strng)-1):
        if strng[i] in [',', '.', '!'] and strng[i+1] not in [',', '.', '!', ' ']:
            strng = strng[0:i] + ' ' + strng[i:]
    return strng
###############################################################################################


################### Function to create a weight vector based on words ###################
def create_weight_vectors():
    global vanilla_pos_neg_weight_vector, vanilla_true_fake_weight_vector, feature_list
    global averaged_pos_neg_weight_vector, averaged_true_fake_weight_vector
    global cached_pos_neg_weight_vector, cached_true_fake_weight_vector
    for sentence in feature_list:
        for word in sentence:
            if word not in vanilla_pos_neg_weight_vector.keys():
                vanilla_pos_neg_weight_vector[word] = 0
            if word not in vanilla_true_fake_weight_vector.keys():
                vanilla_true_fake_weight_vector[word] = 0
    averaged_pos_neg_weight_vector = copy.deepcopy(vanilla_pos_neg_weight_vector)
    averaged_true_fake_weight_vector = copy.deepcopy(vanilla_true_fake_weight_vector)
    cached_pos_neg_weight_vector = copy.deepcopy(vanilla_pos_neg_weight_vector)
    cached_true_fake_weight_vector = copy.deepcopy(vanilla_true_fake_weight_vector)
#############################################################################


################### Function to count the frequency of word in a sentence ###################
def count_word_frequency(sentence):
    dic = {}
    for x in sentence:
        if x not in dic:
            dic[x] = sentence.count(x)
    return dic
#############################################################################################


################### Function to evaluate weights and bias for vanilla perceptron ###################
def train_vanilla_perceptron(max_iteration=50):
    global vanilla_pos_neg_weight_vector, vanilla_true_fake_weight_vector, feature_list
    pos_neg_b = 0
    true_fake_b = 0
    for i in range(0, max_iteration):
        for j in range(0, len(feature_list)):
            frequency_count = count_word_frequency(feature_list[j])
            true_fake_a = 0
            pos_neg_a = 0
            if tag_details[j][2] == "Pos":
                pos_neg_y = 1
            else:
                pos_neg_y = -1
            if tag_details[j][1] == "True":
                true_fake_y = 1
            else:
                true_fake_y = -1
            for key in frequency_count:
                pos_neg_a += frequency_count[key] * vanilla_pos_neg_weight_vector[key]
                true_fake_a += frequency_count[key] * vanilla_true_fake_weight_vector[key]
            pos_neg_a += pos_neg_b
            true_fake_a += true_fake_b
            if pos_neg_y * pos_neg_a <= 0:
                for key in frequency_count:
                    vanilla_pos_neg_weight_vector[key] += pos_neg_y * frequency_count[key]
                pos_neg_b += pos_neg_y
            if true_fake_y * true_fake_a <= 0:
                for key in frequency_count:
                    vanilla_true_fake_weight_vector[key] += true_fake_y * frequency_count[key]
                true_fake_b += true_fake_y
    return vanilla_pos_neg_weight_vector, vanilla_true_fake_weight_vector, pos_neg_b, true_fake_b
####################################################################################################


################### Function to evaluate weights and bias for average perceptron ###################
def train_averaged_perceptron(max_iteration=60):
    global averaged_pos_neg_weight_vector, averaged_true_fake_weight_vector, feature_list
    global cached_pos_neg_weight_vector, cached_true_fake_weight_vector
    pos_neg_b = 0
    true_fake_b = 0
    pos_neg_beta = 0
    true_fake_beta = 0
    counter = 1
    for i in range(0, max_iteration):
        for j in range(0, len(feature_list)):
            frequency_count = count_word_frequency(feature_list[j])
            true_fake_a = 0
            pos_neg_a = 0
            if tag_details[j][2] == "Pos":
                pos_neg_y = 1
            else:
                pos_neg_y = -1
            if tag_details[j][1] == "True":
                true_fake_y = 1
            else:
                true_fake_y = -1
            for key in frequency_count:
                pos_neg_a += frequency_count[key] * averaged_pos_neg_weight_vector[key]
                true_fake_a += frequency_count[key] * averaged_true_fake_weight_vector[key]
            pos_neg_a += pos_neg_b
            true_fake_a += true_fake_b
            if pos_neg_y * pos_neg_a <= 0:
                for key in frequency_count:
                    averaged_pos_neg_weight_vector[key] += pos_neg_y * frequency_count[key]
                    cached_pos_neg_weight_vector[key] += pos_neg_y * counter * frequency_count[key]
                pos_neg_b += pos_neg_y
                pos_neg_beta += pos_neg_y * counter
                true_fake_beta += pos_neg_y * counter
            if true_fake_y * true_fake_a <= 0:
                for key in frequency_count:
                    averaged_true_fake_weight_vector[key] += true_fake_y * frequency_count[key]
                    cached_true_fake_weight_vector[key] += true_fake_y * counter * frequency_count[key]
                true_fake_b += true_fake_y
                true_fake_beta += pos_neg_y * counter
        counter += 1
    for item in averaged_pos_neg_weight_vector:
        averaged_pos_neg_weight_vector[item] -= cached_true_fake_weight_vector[item]/counter
    for item in averaged_true_fake_weight_vector:
        averaged_true_fake_weight_vector[item] -= averaged_true_fake_weight_vector[item]/counter
    pos_neg_b -= pos_neg_beta/counter
    true_fake_b -= true_fake_beta/counter
    return averaged_pos_neg_weight_vector, averaged_true_fake_weight_vector, pos_neg_b, true_fake_b
####################################################################################################


################### Function to write to vanilla model file ###################
def write_to_vanilla_model_file(v_result):
    with open("vanillamodel.txt", 'w') as f:
        f.write("True/Fake Bias:%s\n" % v_result[3])
        f.write("Pos/Neg Bias:%s\n" % v_result[2])
        f.write("True/Fake Weight Vector:\n")
        f.write(str(v_result[1]))
        f.write("\nPos/Neg Weight Vector:\n")
        f.write(str(v_result[0]))
###############################################################################


################### Function to write to vanilla model file ###################
def write_to_averaged_model_file(avg_result):
    with open("averagedmodel.txt", 'w') as f:
        f.write("True/Fake Bias:%s\n" % avg_result[3])
        f.write("Pos/Neg Bias:%s\n" % avg_result[2])
        f.write("True/Fake Weight Vector:\n")
        f.write(str(avg_result[1]))
        f.write("\nPos/Neg Weight Vector:\n")
        f.write(str(avg_result[0]))
###############################################################################


################### Function to read input file and generate model file ###################
def main():
    global training_file, stop_words
    training_file = sys.argv[1]
    #training_file = 'train-labeled.txt'
    stop_words = ['with', 'possible', 'j', 'thought', 'does', 'furthering', 'necessary', 'shouldnt', 'members',
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
    create_weight_vectors()
    vanilla_result = train_vanilla_perceptron()
    averaged_result = train_averaged_perceptron()
    write_to_vanilla_model_file(vanilla_result)
    write_to_averaged_model_file(averaged_result)
############################################################################################


if __name__ == "__main__":
    main()


