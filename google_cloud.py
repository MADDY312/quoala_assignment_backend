from google.cloud import vision
import os
import re
from flask_cors import CORS
from io import BytesIO
import base64
from PIL import Image
from flask import Flask,request, jsonify
# from flask import Flask,request, jsonify
from pymongo import MongoClient


app = Flask(__name__)
CORS(app)
print("tudutudu")

@app.route('/get')
def hello_world():

    return 'Hello from Flasktu naacj bhutni ke!'


def remove_non_encodable_chars(input_str):
    # Filter out characters that cannot be encoded using 'utf-8'
    cleaned_str = ''.join(char for char in str(input_str) if isinstance(char, str) and ord(char) < 128)
    return cleaned_str

def detect_text(path):
    """Detects text in the file."""

    # print("badhai ho")
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    # cv2.imwrite("dmdd.jpg",content)
    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    # print(response)

    texts = response.text_annotations
    # print("Texts:")
    texts=remove_non_encodable_chars(texts)
    print(texts)
    # for text in texts:
    #     # print(f'\n"{text.description}"')
    #     # print('\n{}'.format(text.description.encode('utf-8')))
    #     # textss=text.description.encode('utf-8')
    #     vertices = [
    #         f"({vertex.x},{vertex.y})" for vertex in text.bounding_poly.vertices
    #     ]

    #     # print("bounds: {}".format(",".join(vertices)))
    #     # print("bounds: {}".format(",".join(vertices)))
    #     break
    # # print("tudum",textss)
    # if response.error.message:
    #     raise Exception(
    #         "{}\nFor more info on error messages, check: "
    #         "https://cloud.google.com/apis/design/errors".format(response.error.message)
    #     )
    return texts

def preprocess_image(image_data):
    # Decode base64 and open image using PIL
    image_data_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_data_bytes))
    return image

def find_index_of_first_number_before_index(s, target_index):
    for i in range(target_index - 1, -1, -1):
        if s[i].isdigit() and 0 <= int(s[i]) <= 9:
            return i
    return None

@app.route('/get_image', methods=['POST'])
def main_module():

    # Set the path to your service account key file
    # key_path = 'cloudvisionkey.json'
    # Connect to MongoDB Atlas
    client = MongoClient("mongodb+srv://20ucc114:Vasu123@cluster0.dhfpn7l.mongodb.net/")
    db = client["people"]
    collection = db["person"]

    # app = Flask(__name__)
    CORS(app, origins="https://qoala-assignment-dce2b.web.app")

    key_path = '/home/litescanpy/mysite/cloudvisionkey.json'
    os.environ['GOOGLE_APPLICATION_CREDENTIALS']=key_path
# Specify the path to the image you want to analyze
    json_data = request.get_json()
    image_data = json_data['image']
    image_data=image_data.replace("data:image/jpeg;base64,","")
    padding = '=' * (4 - len(image_data) % 4)
    image_data += padding
    image = preprocess_image(image_data)
    image_path = 'new_im.jpg'
    image.save(image_path)
    text=detect_text(image_path)
    id_no_end_index=text.lower().index("identification number")
    name_start_ind=text.lower().index("name")
    last_name_start_ind=text.lower().index("last name")
    # dob_start_ind=text.index("Date of Birth")
    # doe_end_ind=text.index("Date of Expiry")
    # doi_end_ind=text.index("Date of Issue")



    # Specify the starting index
    start_index =find_index_of_first_number_before_index(text, id_no_end_index)  # Adjust this according to your needs

    # Find the index of the previous occurrence of '\n' before the starting index
    previous_newline_index = text.rfind('\\n', 0, start_index)
    # print(text.find('\n', name_start_ind))
    identification_number=text[previous_newline_index + 2:start_index]
    print(text[previous_newline_index + 2:start_index])
    first_name=text[name_start_ind+5:text.find('\\n', name_start_ind)]
    print(first_name)
    last_name=text[last_name_start_ind+10:text.find('\\n', last_name_start_ind)]
    print(last_name)
    # dob=text[dob_start_ind+14:text.find('\\n', dob_start_ind)]
    # print(dob)
    text=text.replace('\\n',' ')

    # date_pattern = r'\b\d{1,2} [a-zA-Z]+\. \d{4}\b'
    date_pattern = r'\b\d{1,2} [a-zA-Z]+[^\d\s]* \d{4}\b'

    dates = re.findall(date_pattern, text)
    def get_year(date):
        return int(date.split()[-1])

    # Sort the dates based on their years
    sorted_dates = sorted(dates, key=get_year)
    print(sorted_dates)
    dob=sorted_dates[0]
    doi=sorted_dates[1]
    doe=sorted_dates[2]
    # Print the sorted dates
    print(dob)
    print(doi)
    print(doe)
    # print("this this ",text)
    # private String id;
	#  private String first_name;
	#  private String last_name;
	#  public String date_of_birth;
	#  public String date_of_issue;
	#  public String date_of_expiry;
    # identification_number="nil"
    user_data = {
            "id":identification_number.strip() ,
            "first_name": first_name.strip(),
            "last_name": last_name.strip(),
            "date_of_birth": dob.strip(),
            "date_of_expiry": doe.strip(),
            "date_of_issue": doi.strip()
        }

    result = collection.insert_one(user_data)
    print("User data inserted with IDsss:", result.inserted_id)
    # user_data["_id"] = str(result.inserted_id)
    user_data.pop("_id", None)
    #sending data to the database restcontroller to be stored
    # api_url = 'http://localhost:8080/add/user'
    # # Making a POST API call using the requests library
    # response = requests.post(api_url, json=user_data)
    # this is sent back to the frontend for display purposes
    print("thakthak")
    response=jsonify(user_data)
    print("thakthak")
    print(response)
    return response


@app.route('/search', methods=['POST'])
def find_user():
    CORS(app)
    client = MongoClient("mongodb+srv://20ucc114:Vasu123@cluster0.dhfpn7l.mongodb.net/")
    db = client["people"]
    collection = db["person"]
    json_data = request.get_json()
    search_query = json_data.get('query')

    # Check if the search query is provided
    if not search_query:
        return jsonify({"error": "Search query is required"}), 400

    
    # Search for the user by name or identification number
    user = collection.find_one({
        "$or": [
            {"first_name": {"$regex": search_query, "$options": "i"}},
            {"last_name": {"$regex": search_query, "$options": "i"}},
            {"id": search_query}
        ]
    })

    # If user not found, return an error response
    if not user:
        return jsonify({"error": "User not found"}), 404

    user["_id"] = str(user["_id"])  # Convert ObjectId to string
    user.pop("_id", None)  # Remove the _id field
    # Return the found user data
    return jsonify(user)


@app.route('/alluser', methods=['GET'])
def find_all_docs():
    CORS(app)
    client = MongoClient("mongodb+srv://20ucc114:Vasu123@cluster0.dhfpn7l.mongodb.net/")
    db = client["people"]
    collection = db["person"]
    all_documents = collection.find()
    documents_list = []
    for document in all_documents:
        document["_id"] = str(document["_id"])  # Convert ObjectId to string
        document.pop("_id", None)  # Remove the _id field
        documents_list.append(document)
    return jsonify(list(documents_list))








if __name__ == '__main__':
    app.run(port=5000)    
