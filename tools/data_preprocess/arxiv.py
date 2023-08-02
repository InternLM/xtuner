import json
from datetime import datetime


def has_intersection(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    return len(set1.intersection(set2)) > 0


def read_json_file(file_path):
    data = []
    with open(file_path) as file:
        for line in file:
            try:
                json_data = json.loads(line)
                data.append(json_data)
            except json.JSONDecodeError:
                print(f'Failed to parse line: {line}')
    return data


file_path = './data/arxiv-metadata-oai-snapshot.json'
json_data = read_json_file(file_path)

target_categories = ['cs.AI', 'cs.CL', 'cs.CV']
from_time = datetime(2020, 1, 1)
filtered_data = [
    item for item in json_data
    if has_intersection(target_categories, item['categories'].split())
    and datetime.strptime(item['update_date'], '%Y-%m-%d') >= from_time
]

file_path = './data/arxiv_postprocess_csAIcsCLcsCV_20200101.json'

with open(file_path, 'w') as file:
    json.dump(filtered_data, file)

print(f'Save to {file_path}\n{len(filtered_data)} items')
