from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import scipy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

fileName1 = "RecipeData.csv"
RecipeData = pd.read_csv(fileName1)

app = Flask(__name__)

#/making the routes and assigning the templates to my pages -> names are self-explanatory.
@app.route('/')
@app.route('/aboutme')
def home_page():
    return render_template('home.html')

@app.route('/resume')
def resume_page():
    return render_template('resume.html')

@app.route('/projects')
def project_page():
    return render_template('project.html')

@app.route('/uniquecuisine')
def uniquecuisine_page():
    return render_template('uniquecuisine.html')

#This is the function being used to predict the cuisine from the list that will be passed
#by the user. My model, which was converted to a pkl file in model.py, is loaded and ready to fed the list.
#Returns the first result.

def ValuePredictor(to_predict_list):
    test_list = ['']
    test_list_final = []
    test_list_string = ""

    vectorizer = CountVectorizer()

    for x, y in enumerate(test_list):
        test_list_string = test_list_string + y + " "
        test_list_final.append(test_list_string)

    loaded_model = pickle.load(open("model.pkl", "rb"))

    result = loaded_model.predict(test_list_final)

    return result[0]

#When UniqueCuisine is submitted, /result is returned. The form results are passed to the ValuePredictor function
#and returned. I then passed it as an object to my recipe function which is designed to output the top 5 recipes with
#1. The cuisine predicted by my machine learning model.
#2. The most ingredients that were submitted by the user so it is tailored to their submission.
#3. The highest rating of those recipes.
#I pass prediction in so as to get a dataframe with only recipes of that cuisine. Then I assign each submitted
#ingredient to a variable. Then I create a new df that has just the recipes which contain those ingredients.
#I add a column of the counts of a recipe occurence, meaning that at this point, that count represents the number of
#ingredients which were in that recipe. I wanted the results to be more interactive so rather than just print the dataframe
#I returned each cell indivividually as part of the result function to be used on the results page.

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(str, to_predict_list))
        result = ValuePredictor(to_predict_list)

        prediction = str(result)

        def recipe(prediction):
            to_predict_list = request.form.to_dict()
            to_predict_list = list(to_predict_list.values())

            recipe = RecipeData[(RecipeData['tags'] == prediction)].sort_values(by=['rating'], ascending=False)
            ingredient1 = to_predict_list[0]
            ingredient2 = to_predict_list[1]
            ingredient3 = to_predict_list[2]
            ingredient4 = to_predict_list[3]
            ingredient5 = to_predict_list[4]
            recipeswithingredients = recipe.loc[
            recipe['ingredients'].isin([ingredient1, ingredient2, ingredient3, ingredient4, ingredient5])]
            recipeswithingredients = recipeswithingredients.drop_duplicates(
                subset=['id', 'ingredients'],
                keep='last').reset_index(drop=True)
            recipeswithingredients['counts'] = recipeswithingredients.groupby('name')['name'].transform('count')
            recipeswithingredients = recipeswithingredients.drop_duplicates('name', keep='last')
            recipeswithingredients = recipeswithingredients.sort_values(['counts', 'rating'], ascending=[False, False])
            recipeswithingredients = recipeswithingredients[['name', 'rating', 'counts']].copy()
            recipeswithingredients = recipeswithingredients.reset_index()
            recipeswithingredients.drop('index', axis=1, inplace=True)
            top5recipes = recipeswithingredients.head()

            return top5recipes

        top5recipes=recipe(prediction)

        firstname = top5recipes.loc[0, 'name']
        secondname = top5recipes.loc[1, 'name']
        thirdname = top5recipes.loc[2, 'name']
        fourthname = top5recipes.loc[3, 'name']
        fifthname = top5recipes.loc[4, 'name']
        firstrating = top5recipes.loc[0, 'rating']
        secondrating = top5recipes.loc[1, 'rating']
        thirdrating = top5recipes.loc[2, 'rating']
        fourthrating = top5recipes.loc[3, 'rating']
        fifthrating = top5recipes.loc[4, 'rating']
        firstcount = top5recipes.loc[0, 'counts']
        secondcount = top5recipes.loc[1, 'counts']
        thirdcount = top5recipes.loc[2, 'counts']
        fourthcount = top5recipes.loc[3, 'counts']
        fifthcount = top5recipes.loc[4, 'counts']

        if fifthname.len() > 1:
            return render_template("result.html", prediction=prediction, firstname=firstname, firstcount=firstcount,
                                   firstrating=firstrating, secondname=secondname, secondcount=secondcount,
                                   secondrating=secondrating, thirdname=thirdname, thirdcount=thirdcount,
                                   thirdrating=thirdrating, fourthcount=fourthcount, fourthname=fourthname,
                                   fourthrating=fourthrating, fifthname=fifthname, fifthrating=fifthrating,
                                   fifthcount=fifthcount)
        elif fourthname.len() > 1:
            return render_template("result4.html", prediction=prediction, firstname=firstname, firstcount=firstcount,
                                    firstrating=firstrating, secondname=secondname, secondcount=secondcount,
                                    secondrating=secondrating, thirdname=thirdname, thirdcount=thirdcount,
                                    thirdrating=thirdrating, fourthcount=fourthcount, fourthname=fourthname,
                                    fourthrating=fourthrating)
        elif thirdname.len() > 1:
                    return render_template("result3.html", prediction=prediction, firstname=firstname,
                                    firstcount=firstcount, firstrating=firstrating, secondname=secondname,
                                    secondcount=secondcount, secondrating=secondrating, thirdname=thirdname,
                                    thirdcount=thirdcount, thirdrating=thirdrating)
        elif secondname.len() > 1:
                    return render_template("result2.html", prediction=prediction, firstname=firstname,
                                    firstcount=firstcount, firstrating=firstrating, secondname=secondname,
                                    secondcount=secondcount, secondrating=secondrating)
        elif firstname.len() > 1:
                     return render_template("result1.html", prediction=prediction, firstname=firstname,
                                    firstcount=firstcount)
        else:
            return render_template("resultnone.html")



if __name__ == "__main__":
    app.run(debug=True)


