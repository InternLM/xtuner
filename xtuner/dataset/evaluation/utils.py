import numpy as np
from collections import defaultdict
import json


def process_punctuation(inText):
    import re
    outText = inText
    punct = [
        ';', r'/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-',
        '>', '<', '@', '`', ',', '?', '!'
    ]
    commaStrip = re.compile('(\d)(,)(\d)')  # noqa: W605
    periodStrip = re.compile('(?!<=\d)(\.)(?!\d)')  # noqa: W605
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(
                commaStrip, inText) is not None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub('', outText, re.UNICODE)
    return outText


def YOrN_Extraction(output):
    s = output.lower()
    words = process_punctuation(s).split()
    if 'yes' in words and 'no' not in words:
        return 'Yes'
    if 'yes' not in words and 'no' in words:
        return 'No'
    return 'Unknown'


def MME_rating(data):
    stats = defaultdict(dict)
    lt = len(data)
    for i in range(lt):
        item = data.iloc[i]
        category = item['category']
        image_path = item['image_path']
        score = item['score']
        if image_path not in stats[category]:
            stats[category][image_path] = []
        stats[category][image_path].append(score)

    def acc(key, mode='normal'):
        res = stats[key]
        values = []
        for val in res.values():
            if mode == 'normal':
                values.extend(val)
            elif mode == 'plus':
                values.append(val[0] * val[1])
        return np.mean(values) * 100

    scores = {}
    for k in stats:
        scores[k] = acc(k) + acc(k, 'plus')

    super_cates = dict(
        perception=[
            'OCR', 'artwork', 'celebrity', 'color', 'count', 'existence',
            'landmark', 'position', 'posters', 'scene'
        ],
        reasoning=['code_reasoning', 'commonsense_reasoning', 'numerical_calculation', 'text_translation']
    )

    ret = {}
    for sc, cate_list in super_cates.items():
        base = 0
        for c in cate_list:
            base += scores[c]
        ret[sc] = base
    ret.update(scores)
    return ret


def Hallusion_rating(data):
    def calc_fAcc(data):
        res = defaultdict(list)
        lt = len(data)
        for i in range(lt):
            line = data.iloc[i]
            res[f"{line['l2-category']}_{line['set_id']}_{line['figure_id']}"].append(line['score'])
        return np.mean([np.all(x) for x in res.values()]) * 100

    def calc_qAcc(data):
        res = defaultdict(list)
        lt = len(data)
        for i in range(lt):
            line = data.iloc[i]
            res[f"{line['l2-category']}_{line['set_id']}_{line['question_id']}"].append(line['score'])
        return np.mean([np.all(x) for x in res.values()]) * 100

    def calc_aAcc(data):
        return np.mean(data['score']) * 100

    data['set_id'] = [x.split('_')[3] for x in data['index']]
    data['figure_id'] = [x.split('_')[4] for x in data['index']]
    data['question_id'] = [x.split('_')[5] for x in data['index']]

    res = dict(split=[], aAcc=[], fAcc=[], qAcc=[])
    res['split'].append('Overall')
    res['aAcc'].append(calc_aAcc(data))
    res['fAcc'].append(calc_fAcc(data))
    res['qAcc'].append(calc_qAcc(data))

    if 'category' in data:
        cates = list(set(data['category']))
        for c in cates:
            sub = data[data['category'] == c]
            res['split'].append(c)
            res['aAcc'].append(calc_aAcc(sub))
            res['fAcc'].append(calc_fAcc(sub))
            res['qAcc'].append(calc_qAcc(sub))

    if 'l2-category' in data:
        cates = list(set(data['l2-category']))
        for c in cates:
            sub = data[data['l2-category'] == c]
            res['split'].append(c)
            res['aAcc'].append(calc_aAcc(sub))
            res['fAcc'].append(calc_fAcc(sub))
            res['qAcc'].append(calc_qAcc(sub))
    return res


def load_jsonl(json_file):
    with open(json_file) as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(json.loads(line))
    return data
