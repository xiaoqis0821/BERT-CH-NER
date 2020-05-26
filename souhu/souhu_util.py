import codecs //处理编码的
import json
from tqdm import tqdm //进度条，可以在 Python 长循环中添加一个进度提示信息
import re //正则表达式
import tokenization 

#一般标题中会含有实体的信息，就把标题加在正文中，作为真正的正文
# tokenizer = FullTokenizer(vocab_file='/opt/hanyaopeng/souhu/data/chinese_L-12_H-768_A-12/vocab.txt', do_lower_case=True)
tokenizer = tokenization.BasicTokenizer(do_lower_case=True)
input_file = '/opt/hanyaopeng/souhu/data/data_v2/coreEntityEmotion_test_stage1.txt'

with open(input_file, encoding='utf-8') as f:
    test_data = []
    # 从数据集中读出数据
    for l in tqdm(f):
        #.strip()去头尾空格的
        data = json.loads(l.strip())
        news_id = data['newsId']
        title = data['title']
        title = tokenizer.tokenize(title)
        #.join 字符串拼接的
        title = ''.join([l for l in title])
        # 文章主体
        content = data['content']
        sentences = []
        # 每个字前面都有''
        ans = '' + title
        # re.split 字符串切割存到列表中
        for seq in re.split(r'[\n。]', content):
            # 用bert自带的洗数据
            seq = tokenizer.tokenize(seq)
            # 对文本分 字 
            seq = ''.join([l for l in seq])
            if len(seq) > 0:
                # 标题+文本分字后的长度不超过350的，就是ans + '。' + seq，否则就把标题append在后面
                if len(seq) + len(ans) <= 254:
                    if len(ans) == 0:
                        ans = ans + seq
                    else:
                        ans = ans + '。' + seq
                elif len(seq) + len(ans) > 254 and len(seq) + len(ans) < 350 and len(ans) < 150:
                    if len(ans) == 0:
                        ans = ans + seq + '。'
                    else :
                        ans = ans + '。' + seq + '。'
                    sentences.append(ans)
                    ans = ''
                else:
                    ans = ans + '。'
                    sentences.append(ans)
                    ans = ''
        if len(ans) != 0:
            sentences.append(ans)
        # 将文本和label 序列化
        for seq in sentences:
            # label列表初始化 全'0'
            label = ['O'] * len(seq)
            l = ' '.join([la for la in label])
            w = ' '.join([word for word in seq])
            test_data.append((news_id, w, l))


#将处理好的数据存到json中
import codecs
with codecs.open("/opt/hanyaopeng/souhu/data/data_v2/test_samplev2.json", 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False)
    
#文本实体
fr = open('/opt/hanyaopeng/souhu/data/chinese_L-12_H-768_A-12/outputv2/test_predictionv2.txt', 'r', encoding='utf-8')
result = fr.readlines()

test_sample = json.load(open('/opt/hanyaopeng/souhu/data/data_v2/test_samplev2.json', encoding='utf-8'))
entity = {}
for i in range(len(test_sample)):
    a = result[i].split(' ')[1:-1]  # label
    t = test_sample[i][1].split(' ')  # seq
    newsid = test_sample[i][0]
    if newsid not in entity:
        entity[newsid] = []
    # 如果长度超过254就截取前254个字
    if len(t) > 254: # max_seq_length
        t = t[:254]
    ent = {}
    assert len(a) == len(t)
    j = 0
    while j < len(a):
        if a[j] == 'S':
            entity[newsid].append(t[j])
            j += 1
        elif a[j] == 'B':
            flag = j
            k = j + 1
            while k < len(a):
                if a[k] == 'E':
                    ti = ''.join([la for la in t[flag:k+1]])
                    entity[newsid].append(ti)
                    j = k + 1
                    break
                elif a[k] == 'O':
                    j = k+1
                    break
                else:
                    k += 1
            j = k
        else:
            j += 1

    if i % 100000 == 0:
        print(i)

res = {}
for i in entity.keys():
    res[i] = []
    items = {}
    for j in entity[i]:
        if j in items:
            items[j] += 1
        else:
            items[j] = 1
    ans = sorted(items.items(),key=lambda x:x[1],reverse=True)
    if len(ans) >= 3: # 取 存放的实体数量
        res[i] = [i[0] for i in ans[:3]]
    else :
        res[i] = [i[0] for i in ans]

res_file = open("/opt/hanyaopeng/souhu/data/sub/subbmission4.txt", 'w', encoding='utf-8')

for i in res:
    ent = res[i]
    for en in range(len(ent)):
        ent[en] = ent[en].replace(",",'')
        ent[en] = ent[en].replace("，",'')
    emos = []
    for j in range(len(ent)):
        emos.append('POS')
    row = i +'\t'+','.join(ent)+'\t'+','.join(emos) + '\n'
    res_file.write(row)
    res_file.flush()
