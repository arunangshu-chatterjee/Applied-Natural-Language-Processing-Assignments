import sys
import string
import math

################### Variable Declaration ###################
file_path = ''
stopWords = []
feature_list = []
true_score = 0
fake_score = 0
pos_score = 0
neg_score = 0
feature_given_class_probabilities = {}
class_prior_probabilities = {}
unique_id = ''
############################################################


################### Function to read and process nbmodel.txt ###################
def read_and_process_model_file():
    global feature_given_class_probabilities, class_prior_probabilities
    count = 0
    f = open("nbmodel.txt", "r", encoding="utf8")
    for item in f:
        if ':' in item:
            key, value = item.split(':', 1)
            if count <= 4:
                class_prior_probabilities[key] = value.strip()
            elif count > 5:
                value = value.strip('[').replace(']', '')
                feature_given_class_probabilities[key] =[(x.strip()) for x in value.split(',')]
        count += 1
    f.close()
################################################################################


################### Function to process the dev data file ###################
def classify(f_path, out_file):
    global pos_score,neg_score,true_score,fake_score, feature_list, unique_id
    with open(f_path, 'r', encoding='utf8') as f:
        for line in f:
            unique_id = line[:line.find(" ")]

            # Segregate the review text and store it into a variable
            review_text = " ".join(line.split()[1:])

            # Remove punctuations and convert to lower case
            review_text = "".join(l for l in review_text if l not in string.punctuation).lower()

            feature_list = create_feature_list(review_text)

            true_score = 0
            fake_score = 0
            pos_score = 0
            neg_score = 0

            for feature in feature_list:
                if feature in feature_given_class_probabilities.keys():
                    true_score += math.log(float(feature_given_class_probabilities[feature][0]))
                    fake_score += math.log(float(feature_given_class_probabilities[feature][1]))
                    pos_score += math.log(float(feature_given_class_probabilities[feature][2]))
                    neg_score += math.log(float(feature_given_class_probabilities[feature][3]))
            true_score += math.log(float(class_prior_probabilities['True']))
            fake_score += math.log(float(class_prior_probabilities['Fake']))
            pos_score += math.log(float(class_prior_probabilities['Pos']))
            neg_score += math.log(float(class_prior_probabilities['Neg']))

            if true_score > fake_score:
                label1 = "True"
            else:
                label1 = "Fake"
            if pos_score > neg_score:
                label2 = "Pos"
            else:
                label2 = "Neg"
            out_file.write(unique_id.strip()+" "+label1.strip()+" "+label2.strip())
            out_file.write('\n')
#############################################################################


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


################### Function to read input file and generate model file ###################
def main():
    global file_path, stopWords
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
    read_and_process_model_file()
    output_file = open("nboutput.txt", 'w')
    file_path = "dev-text.txt"
    classify(file_path, output_file)
    output_file.close()
###########################################################################################


if __name__ == "__main__":
    main()