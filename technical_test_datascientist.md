# Test - Data Scientist (profile statistician)

## Challenge

The `test_data_CANDIDATE.csv` file contains some patient's data. The test consists in:

1) Analyzing the data
2) Building a SEX PREDICTOR alongside a script that allows the model to be used.

## Data descriptions

- age: in years
- sex: (M = male; F = female)
- cp: chest pain type
- trestbps: resting blood pressure (in mm Hg on admission to the hospital)
- chol: serum cholesterol in mg/dl
- fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
- restecg: resting electrocardiographic results
- thalach: maximum heart rate achieved
- nar: number of arms
- exang: exercise induced angina (1 = yes; 0 = no)
- oldpeak: ST depression induced by exercise relative to rest
- slope: the slope of the peak exercise ST segment
- hc: patient's hair colour
- sk: patient's skin colour
- trf: time spent in traffic daily (in seconds)
- ca: number of major vessels (0-3) colored by flourosopy
- thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

## Deliverables

1) The analysis of the data

    - can be presented in any form, but a python notebook is preferable;
    - an analysis and/or brief explanation of the model delivered in the deliverable 2 must also be present.

2) A model that predicts if a batch of patients are male or female, given an entirely new CSV file containing the same columns, except the 'sex' column.

    - the script must be a python file named 'sex_predictor.py'
    - a 'requirements.txt' file is expected, so that the necessary dependencies can be installed in a virtual environment using the command 'pip install requirements.txt'. Don't forget to add the dependencies necessary for the first deliverable too.
    - the script will be used this way:
        (env) python sex_predictor.py --input_file newsample.csv
    - the newsample.csv file is a file with the same structure of the 'test_data_CANDIDATE.csv', but with an unknown number of lines (n)
    - the expected output is a table saved in the same directory but with the name 'newsample_PREDICTIONS_{candidate_fullname}.csv' containing only the column 'sex' which has the predicted values.
    - Python version 3.6.x is preferred.

## Observations

- It is very important that you carefully describe your line of thinking, the steps made, the decisions you made and why, so that we can understand your way of working.
- Don't forget to add a README file detailing how to setup and run your test;
- If you have any tests that failed and you decided to abort, mention it, the parameters used and the results obtained in your report.
- Show us all the models you have tested and tell us why you chose the one you presented to us.

## Submission

- The challenge must be delivered using [GitHub](http://github.com/), [Bitbucket](http://bitbucket.org/) or [GitLab](http://gitlab.com/). Share with us and send the URL of your code by e-mail.
- Any questions should be asked in response to this e-mail.

Good luck!
