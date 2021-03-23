import json
import pdb

squad_ver = 1.1
data_path = "data/train-v{}.json".format(squad_ver)

data = json.load(open(data_path))

# counts = []
# for article in data['data']:
#     count = 0
#     for paragraph in  article['paragraphs']:
#         count += len(paragraph['qas'])
#     counts.append(count)

# print(sum(counts)/len(counts), min(counts), max(counts))

num_articles = len(data['data'])
num_dev_articles = 10

train_data = {}
train_data['version'] = str(squad_ver)
train_data['data'] = data['data'][0:num_articles - num_dev_articles].copy()

val_data = {}
val_data['version'] = str(squad_ver)
val_data['data'] = data['data'][num_articles - num_dev_articles:num_articles].copy()

pdb.set_trace()

with open("data/trainsplit-v{}.json".format(squad_ver), 'w') as outfile:
    json.dump(train_data, outfile)

with open("data/valsplit-v{}.json".format(squad_ver), 'w') as outfile:
    json.dump(val_data, outfile)
