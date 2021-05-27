
#pacotes para funcionamento web
from flask import Flask,url_for,request,render_template,jsonify,send_file
from flask_bootstrap import Bootstrap
import json

# pacotes nlp
import spacy
from textblob import TextBlob 

#inglês
nlp = spacy.load('en_core_web_sm')

#português
#nlp = spacy.load("pt_core_news_sm")


#pacotes para WordCloud e matplotlib 
from wordcloud import WordCloud
import matplotlib.pyplot as plt 
from io import BytesIO
import random
import time


app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def index():
	return render_template('index.html')


@app.route('/analyze',methods=['GET','POST'])
def analyze():
	start = time.time()
	
	if request.method == 'POST':
		rawtext = request.form['rawtext']
		#Análise
		docx = nlp(rawtext)
		# Tokens
		custom_tokens = [token.text for token in docx ]
		# informação da palavra
		custom_wordinfo = [(token.text,token.lemma_,token.shape_,token.is_alpha,token.is_stop) for token in docx ]
		custom_postagging = [(word.text,word.tag_,word.pos_,word.dep_) for word in docx]
		#reconhecimento da named entity (entidade nomeada)
		custom_namedentities = [(entity.text,entity.label_)for entity in docx.ents]
		
		#Textblob
		blob = TextBlob(rawtext)
		blob_sentiment,blob_subjectivity = blob.sentiment.polarity ,blob.sentiment.subjectivity
		
		allData = [('"Token":"{}","Tag":"{}","POS":"{}","Dependency":"{}","Lemma":"{}","Shape":"{}","Alpha":"{}","IsStopword":"{}"'.format(token.text,token.tag_,token.pos_,token.dep_,token.lemma_,token.shape_,token.is_alpha,token.is_stop)) for token in docx ]

		result_json = json.dumps(allData, sort_keys = False, indent = 2)

		end = time.time()
		final_time = end-start
	return render_template('index.html',ctext=rawtext,custom_tokens=custom_tokens,custom_postagging=custom_postagging,custom_namedentities=custom_namedentities,custom_wordinfo=custom_wordinfo,blob_sentiment=blob_sentiment,blob_subjectivity=blob_subjectivity,final_time=final_time,result_json=result_json)

#API simples para testes diretos
@app.route('/api')
def basic_api():
	return render_template('restfulapidocs.html')

#TOKENS
@app.route('/api/tokens/<string:mytext>',methods=['GET'])
def api_tokens(mytext):
	#Análise
	docx = nlp(mytext)
	# Tokens
	mytokens = [token.text for token in docx ]
	return jsonify(mytext,mytokens)

#Lematização
@app.route('/api/lemma/<string:mytext>',methods=['GET'])
def api_lemma(mytext):
	#Análise
	docx = nlp(mytext.strip())
	# Tokens & Lemma
	mylemma = [('Token:{},Lemma:{}'.format(token.text,token.lemma_))for token in docx ]
	return jsonify(mytext,mylemma)

#named entity
@app.route('/api/ner/<string:mytext>',methods=['GET'])
def api_ner(mytext):
	#Análise
	docx = nlp(mytext)
	# Tokens
	mynamedentities = [(entity.text,entity.label_)for entity in docx.ents]
	return jsonify(mytext,mynamedentities)

#named entity
@app.route('/api/entities/<string:mytext>',methods=['GET'])
def api_entities(mytext):
	#Análise
	docx = nlp(mytext)
	# Tokens
	mynamedentities = [(entity.text,entity.label_)for entity in docx.ents]
	return jsonify(mytext,mynamedentities)


#sentimento
@app.route('/api/sentiment/<string:mytext>',methods=['GET'])
def api_sentiment(mytext):
	#Análise
	blob = TextBlob(mytext)
	mysentiment = [ mytext,blob.words,blob.sentiment ]
	return jsonify(mysentiment)


@app.route('/api/nlpiffy/<string:mytext>',methods=['GET'])
def nlpifyapi(mytext):

	docx = nlp(mytext.strip())
	allData = ['Token:{},Tag:{},POS:{},Dependency:{},Lemma:{},Shape:{},Alpha:{},IsStopword:{}'.format(token.text,token.tag_,token.pos_,token.dep_,token.lemma_,token.shape_,token.is_alpha,token.is_stop) for token in docx ]
	
	return jsonify(mytext,allData)
	


@app.route('/images')
def imagescloud():
    return "Digite seu texto na URL. Ex:https://nlp-projeto-2.herokuapp.com/fig/SeutextoEntreAspas"


@app.route('/images/<mytext>')
def images(mytext):
    return render_template("index.html", title=mytext)

@app.route('/fig/<string:mytext>')
def fig(mytext):
    plt.figure(figsize=(20,10))
    wordcloud = WordCloud(background_color='white', mode = "RGB", width = 2000, height = 1000).generate(mytext)
    plt.imshow(wordcloud)
    plt.axis("off")
    img = BytesIO()
    plt.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')


@app.route('/about')
def about():
	return render_template('about.html')

if __name__ == '__main__':
	app.run(debug=True)