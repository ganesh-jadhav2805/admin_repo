import pickle
import pandas as pd
import json
from flask import Flask, jsonify,render_template,request
import config
import numpy as np


with open(r'C:\PYTHON\Lecture files\12_09_Linear_Regression_Model\project_app\Linear Model.pkl', 'rb') as f:
    model = pickle.load(f)
        
with open(r'C:\PYTHON\Lecture files\12_09_Linear_Regression_Model\project_app\Project Data.json', 'r') as f:
    json_data = json.load(f)

app = Flask(__name__)

##########################################################
####################### BASE API #########################
##########################################################

@app.route('/')
def admin_model():
    print('Welcome to admission model')
    return render_template('index.html')




@app.route('/admission_chance',methods=['POST'])
def abc():
    data=request.form
    GRE_Score = data['GRE_Score']
    TOEFL_Score = data['TOEFL_Score']
    University_Rating = data['University_Rating']
    SOP = data['SOP']
    LOR = data['LOR']
    CGPA = data['CGPA']
    test_array = np.zeros(len(json_data['Columns']))
    test_array[0] = GRE_Score
    test_array[1] = TOEFL_Score
    test_array[2] = University_Rating
    test_array[3] = SOP
    test_array[4] = LOR
    test_array[5] = CGPA

    print('Test Array :', test_array)
    result = model.predict([test_array])[0]
    return jsonify({'result': f"Chances of getting admission : {result*100 }"},'%')

if __name__ =='__main__':
    app.run(host='0.0.0.0',port =9000)