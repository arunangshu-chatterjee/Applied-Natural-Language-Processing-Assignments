import sys
import string

################### Variable Declaration ###################
tag_details = []
word_list = []
feature_list = []
stop_words = []
pos_neg_weight_vector = {}
true_fake_weight_vector = {}
dev_file = ''
model_file = ''
output_file = ''
pos_neg_bias = 0
true_fake_bias = 0
line_count = 0
############################################################


################### Function to process Model File ###################
def process_model_file():
    global pos_neg_weight_vector, true_fake_weight_vector, training_file, stop_words, pos_neg_bias, true_fake_bias
    counter = 1
    f = open(model_file, 'r', encoding="utf8")
    for line in f:
        if counter == 1:
            var = line.split(':')
            true_fake_bias = float(var[1])
            counter += 1
            continue
        elif counter == 2:
            var = line.split(':')
            pos_neg_bias = float(var[1])
            counter += 1
            continue
        elif counter == 4:
            true_fake_weight_vector = eval(line)
            counter += 1
            continue
        elif counter == 6:
            pos_neg_weight_vector = eval(line)
        else:
            counter += 1
    f.close()
######################################################################


################### Function to count the frequency of word in a sentence ###################
def count_word_frequency(sentence):
    dic = {}
    for x in sentence:
        if x not in dic:
            dic[x] = sentence.count(x)
    return dic
#############################################################################################


################### Function to process and classify the dev data file ###################
def classify(out_file):
    global dev_file, pos_neg_weight_vector, true_fake_weight_vector, pos_neg_bias, true_fake_bias, word_list
    # unique_id = ''
    with open(dev_file, 'r', encoding='utf8') as f:
        for line in f:
            true_fake_classification_value = 0
            pos_neg_classification_value = 0
            unique_id = line[:line.find(" ")]
            # Segregate the review text and store it into a variable
            review_text = " ".join(line.split()[1:])
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
            frequency_count = count_word_frequency(word_list)
            for item in frequency_count:
                if item in true_fake_weight_vector:
                    true_fake_classification_value += frequency_count[item] * true_fake_weight_vector[item]
                if item in pos_neg_weight_vector:
                    pos_neg_classification_value += frequency_count[item] * pos_neg_weight_vector[item]
            true_fake_classification_value += true_fake_bias
            pos_neg_classification_value += pos_neg_bias

            if true_fake_classification_value > 0:
                label1 = "True"
            else:
                label1 = "Fake"
            if pos_neg_classification_value > 0:
                label2 = "Pos"
            else:
                label2 = "Neg"

            out_file.write(unique_id.strip() + " " + label1.strip() + " " + label2.strip())
            out_file.write('\n')
##########################################################################################


################### Function to add spaces after punctuation if none exists ###################
def insert_space(strng):
    for i in range(0, len(strng)-1):
        if strng[i] in [',', '.', '!'] and strng[i+1] not in [',', '.', '!', ' ']:
            strng = strng[0:i] + ' ' + strng[i:]
    return strng
###############################################################################################



################### Function to read model file and classify ###################
def main():
    global output_file, stop_words, dev_file, model_file
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
    dev_file = sys.argv[2]
    model_file = sys.argv[1]
    #dev_file = 'dev-text.txt'
    #model_file = 'vanillamodel.txt'
    process_model_file()
    output_file = open("percepoutput.txt", 'w')
    classify(output_file)
    output_file.close()
#################################################################################


if __name__ == "__main__":
    main()