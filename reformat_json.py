import json

def reformat_json_keys(inpu_path, output_path):
    with open(input_path, 'r') as file:
        data = json.load(file)
    formatted_data = {}
    for key, value in data.items():
        b, g, r = map(int, key.strip('()').split(', '))
        new_key = f'({b}, {g}, {r})'
        new_value = {
            "Name": value,
            "BGR": [b, g, r]
        }
        formatted_data[new_key] = new_value
    with open(output_path, 'w') as file:
        json.dump(formatted_data, file, indent=2)

# # Convert a text file to a json file
# def convert_txt_json(input_path, output_path):
#     with open(input_path, 'r') as file:
#         lines = file.readlines()
#     color_dict = {}
#     for line in lines:
#         # Extract BGR values and color name
#         parts = line.strip().split(': "')
#         bgr_values = parts[0].strip("()").split(', ')
#         color_name = parts[1].strip('"')
#         # Convert BGR values to integers
#         bgr_values = [int(value) for value in bgr_values]
#         key = color_name.lower().replace(' ', '-')
#         color_dict[key] = {
#             "Name": color_name,
#             "BGR": bgr_values
#         }
#     with open(output_path, 'w') as json_file:
#         json.dump(color_dict, json_file, indent=4)

# # Change key of json dictionary
# def reformat_json(input_path, output_path):
#     with open(input_path, 'r') as f:
#         color_dict = json.load(f)
#     bgr_keyed_dict = {str(entry['BGR']): entry for entry in color_dict.values()}
#     # Save the modified dictionary back to a file (optional)
#     with open(output_path, 'w') as f:
#         json.dump(bgr_keyed_dict, f, indent=4)
#     print({k: bgr_keyed_dict[k] for k in list(bgr_keyed_dict)[:5]})
#     print(f'The dictionary has been saved to {output_path}')

# def reformat_json_key(input_path, output_path):
#     with open(input_path, 'r') as file:
#         data = json.load(file)
#     updated_data = {}
#     for key in data.keys():
#         # Convert the string to a tuple-like format
#         new_key = key.replace('[', '(').replace(']', ')')
#         updated_data[new_key] = data[key]
#     with open(output_path, 'w') as file:
#         json.dump(updated_data, file, indent=4)

if __name__ == '__main__':
    input_path = 'Dictionaries/bgr_json.json'
    output_path = 'Dictionaries/converted_color_dictionary.json'
    reformat_json_keys(input_path, output_path)
