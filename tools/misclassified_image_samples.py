from PIL import Image, ImageDraw, ImageFont
import os
import ast
import json

category_info = {'Drink.Frombottle': 0, 'Drink.Fromcup': 1, 'Eat.Useutensil': 2, 'Eat.Snack': 3, 'Use.Tablet': 4,
                 'Use.Phone': 5, 'Call.Onphone': 6, 'Use.Computer': 7, 'Type.Onkeyboard': 8, 'Use.Switch': 9,
                 'Read': 10, 'Write': 11, 'Play.Cards': 12, 'Play.Chess': 13, 'Play.Lego': 14,
                 'Play.Boardgame': 15, 'Cook.Cut': 16, 'Cook.Usestove': 17, 'Cook.Useoven': 18, 'Cook.Usemicrowave': 19,
                 'Use.Coffeemachine': 20, 'Use.Kettle': 21, 'Use.Refrig': 22, 'Wash.Hands': 23, 'Wash.Dishes': 24,
                 'Use.Sink': 25, 'Use.Shelf': 26, 'Use.Drawer': 27, 'Use.Dishwasher': 28, 'Use.Mop': 29,
                 'Use.Vaccum': 30, 'Nap': 31, 'Use.Gamecontroller': 32, 'Watch.TV': 33, 'Exercise': 34,
                 'Lay.Onbed': 35, 'Getup': 36, 'Draw.Curtain': 37, 'Move.Object': 38, 'Use.Tap': 39,
                 'Use.Switches': 40, 'Cook.Stir': 41, 'Use.Mouse': 42, 'Enter': 43, 'Leave': 44,
                 'Stand': 45, 'Sit': 46, 'lie flat': 47, 'lie on the side': 48, 'go prone': 49,
                 'squat': 50, '101': 51, '102': 52, '103': 53, '104': 54,
                 '105': 55, '106': 56, '107': 57, '108': 58, '109': 59,
                 '110': 60, '111': 61, '112': 62, '113': 63, '114': 64,
                 '115': 65, '116': 66, '117': 67, '118': 68, '119': 69,
                 '199': 70, 'male': 71, 'female': 72}
#get a reverse dictionary of the category_info
category_info_reverse = {v: k for k, v in category_info.items()}

def create_image_grid_with_labels(image_files_info, basedir, grid_size, image_size=(300, 300), label_space=20, save_path='image_grid_with_labels.jpg'):
    # Calculate the canvas size, accounting for label space
    canvas_width = image_size[0] * grid_size[0]
    canvas_height = (image_size[1] + label_space) * grid_size[1]

    incorrect_label_files = 0

    print(canvas_width, canvas_height)
    print(type(canvas_height))
    print(type(canvas_width))
    
    # Create a new blank canvas
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    draw = ImageDraw.Draw(canvas)
    
    # Optionally, load a font. Default font is used if not specified.
    # font = ImageFont.truetype("arial.ttf", 10)
    font = ImageFont.load_default()
    
    # sort the image files by the first element in the tuple
    image_files_info.sort(key=lambda x: x[0])


    # Load, resize, and paste each image onto the canvas, then add labels
    for index, image_info in enumerate(image_files_info):
        # Load image
        predicted, filename = image_info
        predicted_activity = category_info_reverse[predicted]
        img = Image.open(os.path.join(basedir, 'image', filename))
        # Resize image
        img = img.resize(image_size)
        # Calculate position
        x_pos = (index % grid_size[0]) * image_size[0]
        y_pos = (index // grid_size[0]) * (image_size[1] + label_space)
        # Paste image onto canvas
        canvas.paste(img, (x_pos, y_pos))
        # Add label below the image
        img_name = filename.split("/")[-1][:-4]
        label_file = os.path.join(basedir, 'label', img_name+ '.json')
        label = None
        with open(label_file) as f:
                data = json.load(f)
                if len(data['activities']) != 1:
                     print(f"{label_file} with {data['activities']}")
                     incorrect_label_files += 1
                label = data['activities'][0]
        draw.text((x_pos, y_pos + image_size[1]), f"{filename} {label}  vs. {predicted_activity} ({predicted})", fill="black", font=font)
    
    # Save or display the canvas
    canvas.save(save_path)
    canvas.show()
    return incorrect_label_files

basedir = '/Users/yinghong_imac/Sabella Research Project//cropped_/data65k/data/test/'
# Example usage
image_files = ['path/to/image1.jpg', 'path/to/image2.jpg', ...]  # List of image file paths


train_data_summary_file = '/Users/yinghong_imac/Sabella Research Project/cropped_/data65k/data/train/data_summary.json'

def parse_materialized_json(file_path):
    result_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Splitting each line by the first occurrence of ":"
            key_value = line.split(':', 1)
            if len(key_value) == 2:
                # Stripping whitespace from key and value
                key = key_value[0].strip()
                value = key_value[1].strip()
                if (key == 'Loss'):
                     value = float(value)
                if (key == 'Accuracy'):
                    value = float(value)
                if (key == 'Mismatched Label Stats'):
                     value = ast.literal_eval(value)
                if (key == 'Mismatched Label Images'):
                     value = ast.literal_eval(value)
                # Adding the key-value pair to the dictionary
                result_dict[key] = value
    return result_dict

# Example usage
#file_path = '/Users/yinghong_imac/Sabella Research Project/codes/model_testing_result_20240629_192212.txt'#'/Users/yinghong_imac/Downloads/model_testing_result_20240629_063801.txt'
file_path = '/Users/yinghong_imac/Sabella Research Project/codes/model_testing_result_20240629_183631.txt'
parsed_data = parse_materialized_json(file_path)

grid_columns = 20
label_value = 38
num_images = len(parsed_data['Mismatched Label Images'][label_value])
grid_size = (int((num_images+4)/grid_columns), grid_columns)
incorrect_label_files = create_image_grid_with_labels(parsed_data['Mismatched Label Images'][label_value], basedir, grid_size)
print(f"Number of incorrect label files: {incorrect_label_files}")

# with open(train_data_summary_file) as f:
#     data = json.load(f)
#     images_info = data['Mismatched Label Images']
#     misclassified_images = image_info[38]
#     print(misclassified_images)
#create_image_grid_with_labels(image_files, label, basedir, grid_size)