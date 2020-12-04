import pickle

def output_pred(y_pred, testdata_article_id_list, testdata_list,output_path = 'output.tsv'):
    output="article_id\tstart_position\tend_position\tentity_text\tentity_type\n"
    for test_id in range(len(y_pred)):
        pos=0
        start_pos=None
        end_pos=None
        entity_text=None
        entity_type=None
        for pred_id in range(len(y_pred[test_id]) - 1):
            if y_pred[test_id][pred_id][0]=='B':
                start_pos=pos
                entity_type=y_pred[test_id][pred_id][2:]
            elif start_pos is not None and y_pred[test_id][pred_id][0]=='I' and y_pred[test_id][pred_id+1][0]=='O':
                end_pos=pos
                entity_text=''.join([testdata_list[test_id][position][0] for position in range(start_pos,end_pos+1)])
                line=str(testdata_article_id_list[test_id])+'\t'+str(start_pos)+'\t'+str(end_pos+1)+'\t'+entity_text+'\t'+entity_type
                output+=line+'\n'
            pos+=1
    with open(output_path,'w',encoding='utf-8') as f:
        f.write(output)


def loadDevFile(path):
    training_raw = list()
    article_id_set = list()
    training_set = list()
    with open(path, 'r', encoding='utf8') as f:
        file_text=f.read().encode('utf-8').decode('utf-8-sig')
    datas=file_text.split('\n\n--------------------\n\n')[:-1]
    for data in datas:
        data=data.split('\n')
        article_id=int(data[0].split(":")[1][1:])
        content = data[1]
        training_raw.append(content)
        article_id_set.append(article_id)
    
    for text in training_raw:
        t = list()
        for char_ in text:
            t.append(char_)
        training_set.append(t)
    return training_set,training_raw,article_id_set

def Dev_CRFFormatData(text_set, path):
    if (os.path.isfile(path)):
        os.remove(path)
    outputfile = open(path, 'a', encoding= 'utf-8')
    for text in text_set:
        for char_ in text:
            outputfile.write(char_ + "\n")
        outputfile.write("\n")
    outputfile.write("\n")
    # output file lines
    
    # close output file
    outputfile.close()

def merge_maps(dict1, dict2):
    """用于合并两个word2id或者两个tag2id"""
    for key in dict2.keys():
        if key not in dict1:
            dict1[key] = len(dict1)
    return dict1


def save_model(model, file_name):
    """用于保存模型"""
    with open(file_name, "wb") as f:
        pickle.dump(model, f)


def load_model(file_name):
    """用于加载模型"""
    with open(file_name, "rb") as f:
        model = pickle.load(f)
    return model


# LSTM模型训练的时候需要在word2id和tag2id加入PAD和UNK
# 如果是加了CRF的lstm还要加入<start>和<end> (解码的时候需要用到)
def extend_maps(word2id, tag2id, for_crf=True):
    word2id['<unk>'] = len(word2id)
    word2id['<pad>'] = len(word2id)
    tag2id['<unk>'] = len(tag2id)
    tag2id['<pad>'] = len(tag2id)
    # 如果是加了CRF的bilstm  那么还要加入<start> 和 <end>token
    if for_crf:
        word2id['<start>'] = len(word2id)
        word2id['<end>'] = len(word2id)
        tag2id['<start>'] = len(tag2id)
        tag2id['<end>'] = len(tag2id)

    return word2id, tag2id


def prepocess_data_for_lstmcrf(word_lists, tag_lists, test=False):
    assert len(word_lists) == len(tag_lists)
    for i in range(len(word_lists)):
        word_lists[i].append("<end>")
        if not test:  # 如果是测试数据，就不需要加end token了
            tag_lists[i].append("<end>")
    
    return word_lists, tag_lists


def flatten_lists(lists):
    flatten_list = []
    for l in lists:
            flatten_list.extend(l)
    return flatten_list
