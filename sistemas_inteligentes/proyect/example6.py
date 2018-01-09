import pickle

with open('tagged_text_list_test.pkl', 'rb') as f:
    data = pickle.load(f)


print (data)