import pickle
import logging
import gensim
import praw
from praw.models import MoreComments
import os
import flask
from flask import Flask, flash, request,jsonify, json
import json
import joblib
from gensim import utils
import gensim.parsing.preprocessing as gsp

filters = [
		   gsp.strip_tags,
		   gsp.strip_punctuation,
		   gsp.strip_multiple_whitespaces,
		   gsp.strip_numeric,
		   gsp.remove_stopwords,
		   gsp.strip_short,
		   gsp.stem_text
		  ]

def clean(s):
	s = s.lower()
	s = utils.to_unicode(s)
	for f in filters:
		s = f(s)
	return s

app = Flask(__name__,template_folder='templates')


# Use joblib to load in the pre-trained model
model = joblib.load(open('model/xgb.bin', 'rb'))

reddit = praw.Reddit(client_id = "######",
					client_secret = "#######",
					user_agent = "#######",
					username = "########",
					password = "########")


def prediction(url):
	submission = reddit.submission(url = url)
	data = {}
	data["title"] = str(submission.title)
	data["url"] = str(submission.url)
	data["body"] = str(submission.selftext)

	submission.comments.replace_more(limit=None)
	comment = ''
	count = 0
	for top_level_comment in submission.comments:
		comment = comment + ' ' + top_level_comment.body
		count+=1
		if(count > 10):
			 break

	data["comment"] = str(comment)

	data['title'] = clean(str(data['title']))
	data['body'] = clean(str(data['body']))
	data['comment'] = clean(str(data['comment']))

	combined_features = data["title"] + data["comment"] + data["body"] + data["url"]

	return model.predict([combined_features])

# Initialise the Flask app

# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
	if flask.request.method == 'GET':
		# Just render the initial form, to get input
		return(flask.render_template('main.html'))

	if flask.request.method == 'POST':
		# Extract the input
		text = flask.request.form['url']

		# Get the model's prediction
		flair = str(prediction(str(text)))

		# Render the form again, but add in the prediction and remind user
		# of the values they input before
		return flask.render_template('main.html', original_input={'url':str(text)}, result=flair[2:-2])



@app.route("/automated_testing",methods=['POST'])
def test():
	if request.files:
			file = request.files["upload_file"]
			texts = file.read()
			texts = str(texts.decode('utf-8'))
			links = texts.split('\n')
			pred = {}
			for link in links:
				pred[link] =  str(prediction(str(link)))[2:-2]
			return jsonify(pred)
	else:
			return 400

if __name__ == '__main__':
	app.run()