from datasets import load_dataset

ds = load_dataset("rahular/simple-wikipedia")

# Print the first 10 examples from the dataset
for i in range(10):
    print(ds['train'][i]['text'])  # Assuming you are using the 'train' split