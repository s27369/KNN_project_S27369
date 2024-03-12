import math
import matplotlib.pyplot as plt
def read_file(path):
    with open(path, 'r') as f:
        file = f.read()
    return file

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def str_to_dict(string):
    string = string.replace(",", ".")
    l = string.split()
    l = [x.strip() for x in l]

    num_attributes = 0
    temp = l[num_attributes]
    att_names =[]
    while is_float(temp):
        num_attributes+=1
        att_names.append(f'att{num_attributes}')
        temp = l[num_attributes]
    att_names.append("result")

    data = {x:[] for x in att_names}
    # print(num_attributes)
    for i in range(0, len(l), num_attributes+1):
        for j in range(0, num_attributes):
            data[att_names[j]].append(float(l[i+j]))
        data[att_names[-1]].append(l[i+num_attributes])



    return data

def file_to_dict(path, check=False):
    f = read_file(path)
    dic = str_to_dict(f)
    if check:
        print(f'{path}:')
        dataset_info(dic)
        print()
    return dic

def dataset_info(dic):
    for k, v in list(dic.items())[:-1]: print(f'{k}: {len(v)} values, ranging from {min(v)} to {max(v)}')
    s = set(dic["result"])
    print(f'result: {len(s)} unique values {s}')

def get_range(list):
    return max(list)-min(list)

def get_difference(observation, data): #returns a single float value representing the difference of one observation to another
    differences = [math.pow(observation[x]-data[x], 2) for x in range(len(observation)-1)]
    return math.sqrt(sum(differences))

def get_observation(dataset, index):
    return [dataset[x][index] for x in dataset]

def get_k_smallest(l, k):
    return sorted(l)[:k]

def get_indices(l, target_list):
    indices=[]
    for i in range(len(l)):
        for j in range(len(target_list)):
            if l[i] == target_list[j]:
                if not indices.__contains__(j):
                    indices.append(j)

    return indices

def get_dataset_size(dataset):
    return len(dataset["result"])

def knn(observaion, k, train_set, return_differences=False):
    differences = []
    for i in range(get_dataset_size(train_set)):
        x = get_observation(train_set, i)
        differences.append(get_difference(observaion, x))
    smallest = get_k_smallest(differences, k)#smallest differences
    indicies = get_indices(smallest, differences)#indicies of smallest differences
    if return_differences: return indicies, smallest
    return indicies

def classify(indicies,train_set, prnt=False):
    categories = {x:0 for x in set(train_set["result"])}
    for i in indicies:
        categories[train_set["result"][i]]+=1
    if prnt: print(categories)
    return max(categories, key=categories.get)

def classify_dataset(train_set, test_set, k, prnt=False, diff=False):
    test_size = get_dataset_size(test_set)
    correct=0
    for i in range(test_size):
        observation=get_observation(test_set, i)
        indicies, differences =knn(observation, k, train_set, True)
        prediction = classify(indicies, train_set)
        if prnt: print(f'Classified {observation} as {prediction}')
        if prediction==observation[-1]: correct+=1
        elif prnt: print(f"{'^'*10}INCORRECT{'^'*10}")
    accuracy= correct / test_size
    if prnt: print(f'Algorithm was correct in {accuracy}% of cases')
    if diff: return accuracy, differences
    return accuracy

def interface(train, test):
    quit=False
    k = 1
    while not quit:
        print("Choose number:\n1 - choose k parameter value (default: 1)\n2 - input sample data to classify\n3 - quit\n>>>", end="")
        i = int(input())
        if i == 1:
            try:
                k = int(input())
                classify_dataset(train, test, k, True)
            except:
                print("incorrect input")
        elif i == 2:
            obs = input("input values separated by commas\n>>>")
            try:
                obs = obs.split(",")
                obs = [float(i.strip()) for i in obs]
                obs.append("unknown")
                print(classify(knn(obs, k, train), train, prnt=True))
            except:
                print("incorrect input.")
        elif i==3:
            return
        else:
            print("Incorrect input.")

def get_plot(accuracy):
    max_acc, min_acc = max(accuracy.values()), min(accuracy.values())
    plt.plot(accuracy.keys(), accuracy.values())
    plt.xlabel("K value")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs K value")
    plt.xticks([x for x in range(0, 121, 5)])
    plt.yticks([x * 0.01 for x in range(30, 100, 5)])
    plt.axhline(y=max_acc, color='green')
    plt.text(x=0, y=max_acc, s=f'Max accuracy: {max_acc}', color='green', fontsize=8, verticalalignment='bottom')
    plt.axhline(y=min_acc, color='red')
    plt.text(x=0, y=min_acc, s=f'Min accuracy: {min_acc}', color='red', fontsize=8, verticalalignment='bottom')
    plt.show()

if __name__ == '__main__':
    train = file_to_dict("iris_training.txt", True)
    test = file_to_dict("iris_test.txt", True)

    # #demonstration
    # for k in range(0, 101):
    #     print(f'{"-"*20}K: {k}{"-"*20}')
    #     classify_dataset(train, test, k, True)
    #     print()

    # print(classify_dataset(train, test, 3, diff=True))
    accuracy ={}
    for k in range(0, 121):
        accuracy[k]=classify_dataset(train, test, k)

    get_plot(accuracy)

    interface(train, test)
