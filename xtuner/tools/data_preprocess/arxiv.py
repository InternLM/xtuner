# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('src_file', help='source file path')
    parser.add_argument('dst_file', help='destination file path')
    parser.add_argument(
        '--categories',
        nargs='+',
        default=['cs.AI', 'cs.CL', 'cs.CV'],
        help='target categories')
    parser.add_argument(
        '--start-date',
        default='2020-01-01',
        help='start date (format: YYYY-MM-DD)')

    args = parser.parse_args()
    return args


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


def main():
    args = parse_args()
    json_data = read_json_file(args.src_file)
    from_time = datetime.strptime(args.start_date, '%Y-%m-%d')
    filtered_data = [
        item for item in json_data
        if has_intersection(args.categories, item['categories'].split())
        and datetime.strptime(item['update_date'], '%Y-%m-%d') >= from_time
    ]

    with open(args.dst_file, 'w') as file:
        json.dump(filtered_data, file)

    print(f'Save to {args.dst_file}\n{len(filtered_data)} items')


if __name__ == '__main__':
    main()
