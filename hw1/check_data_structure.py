import pickle

# Check the structure of one expert data file
data_file = 'rob831/expert_data/expert_data_Ant-v2.pkl'

with open(data_file, 'rb') as f:
    data = pickle.load(f)

print("Type of data:", type(data))
print("Keys/structure:", data.keys() if hasattr(data, 'keys') else "Not a dict")
print("Length:", len(data) if hasattr(data, '__len__') else "No length")

if isinstance(data, dict):
    for key, value in data.items():
        print(f"Key: {key}, Type: {type(value)}, Shape: {getattr(value, 'shape', 'No shape')}")
elif isinstance(data, list):
    print(f"List with {len(data)} elements")
    if len(data) > 0:
        print(f"First element type: {type(data[0])}")
        if hasattr(data[0], 'keys'):
            print(f"First element keys: {data[0].keys()}")