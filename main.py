
def read_file(path):
    with open(path, 'r') as f:
        file = f.read()
    return file

def str_to_dict(string):
    col_1="a"
    col_2="b"
    col_3="c"
    col_4="d"
    col_5="result"
    string = string.replace(",", ".")
    l = string.split()
    l = [x.strip() for x in l]
    data = {col_1:[], col_2:[], col_3:[], col_4:[], col_5:[]}
    for i in range(0, len(l), 5):
        data[col_1].append(float(l[i]))
        data[col_2].append(float(l[i+1]))
        data[col_3].append(float(l[i+2]))
        data[col_4].append(float(l[i+3]))
        data[col_5].append(l[i+4])
    return data

def file_to_dict(path, check=False):
    f = read_file(path)
    dic = str_to_dict(f)
    if check:
        print(f'{path}:')
        for k, v in dic.items(): print(f'{k}: {len(v)}')
        print()
    return dic

def get_range(list):
    return max(list)-min(list)

if __name__ == '__main__':
    data = file_to_dict("iris_training.txt", True)
    test = file_to_dict("iris_test.txt", True)

    obs = [test[k][0] for k in test]

    val = (obs[0]-data[list(data.keys())[0]][0])/get_range(data[list(data.keys())[0]])

    print(val)

