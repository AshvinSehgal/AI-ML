import requests
import spacy
import pycountry
import json
import polars as pl
import numpy as np
import re
import os
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import webbrowser
from io import BytesIO
import html

os.environ["QT_LOGGING_RULES"] = "qt.gui.icc=false"

nlp = spacy.load("en_core_web_sm")

GEO_KEYWORDS = ['war', 'conflict', 'military', 'trade', 'diplomacy', 'sanctions', 'border', 'embargo', 'treaty', 'geopolitics']

COUNTRIES = ['afghanistan', 'angola', 'anguilla', 'albania', 'andorra', 'united arab emirates', 'argentina', 'armenia',
             'antarctica', 'antigua and barbuda', 'australia', 'austria', 'azerbaijan', 'belgium', 'burkina faso', 'bangladesh', 'bulgaria',
             'bahrain', 'bahamas', 'bosnia and herzegovina', 'belarus', 'bermuda', 'bolivia', 'brazil', 'bhutan',
             'botswana', 'central african republic', 'canada', 'switzerland', 'chile', 'china', 'cote d\'ivoire', 'cameroon', 'congo',
             'democratic republic of congo', 'colombia', 'costa rica', 'cuba', 'cyprus', 'czech republic', 'germany', 'denmark', 'algeria',
             'ecuador', 'egypt', 'eritrea', 'western sahara', 'spain', 'estonia', 'ethiopia', 'finland', 'fiji', 'france', 'united kingdom',
             'georgia', 'ghana', 'guinea', 'gambia', 'guinea-bissau', 'equatorial guinea', 'greece', 'greenland', 'guam', 'guyana', 'hong kong',
             'honduras', 'croatia', 'haiti', 'hungary', 'indonesia', 'india', 'ireland', 'iran', 'iraq', 'iceland', 'israel', 'italy', 'jamaica',
             'jordan', 'japan', 'kazakhstan', 'kenya', 'kyrgyzstan', 'cambodia', 'south korea', 'north korea', 'kuwait', 'laos', 'lebanon', 'liberia',
             'libya', 'sri lanka', 'lithuania', 'luxembourg', 'latvia', 'macao', 'morocco', 'monaco', 'moldova', 'maldives', 'mexico', 'macedonia',
             'mali', 'myanmar', 'mongolia', 'mozambique', 'mauritius', 'malaysia', 'namibia', 'niger', 'nigeria', 'netherlands', 'norway', 'nepal',
             'new zealand', 'oman', 'pakistan', 'panama', 'peru', 'philippines', 'papua new guinea', 'poland', 'puerto rico', 'portugal', 'paraguay',
             'palestine', 'qatar', 'romania', 'russia', 'rwanda', 'saudi arabia', 'sudan', 'senegal', 'singapore', 'somalia', 'serbia', 'south sudan',
             'slovakia', 'slovenia', 'sweden', 'syria', 'chad', 'thailand', 'tajikistan', 'tunisia', 'turkey', 'taiwan', 'tanzania', 'uganda', 'ukraine',
             'united states', 'uzbekistan', 'vatican city', 'venezuela', 'vietnam', 'south africa', 'zambia', 'zimbabwe']

ABBREVIATIONS = {
    'aland': 'aland islands',
    'uae': 'united arab emirates',
    'u.a.e.': 'united arab emirates',
    'drc': 'democratic republic of congo',
    'd.r.c.': 'democratic republic of congo',
    'uk': 'united kingdom',
    'u.k.': 'united kingdom',
    'rok': 'south korea',
    'dprk': 'north korea',
    'macau': 'macao',
    'north macedonia': 'macedonia',
    'burma': 'myanmar',
    'nz': 'new zealand',
    'n.z.': 'new zealand',
    'us': 'united states',
    'u.s.': 'united states'
}

NEWS_API_KEY = '20a4bc1d3bc14ab1b5520927ac099d52'
NEWS_ENDPOINT = 'https://newsapi.org/v2/everything'

PATH_TO_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'news.csv')

news_dict = {
            'id': [],
            'source': [],
            'title': [],
            'description': [],
            'content': [],
            'url': [],
            'image': [],
            'country': []
}

ids = []
news_slides = []
current_idx = []

if os.path.exists(PATH_TO_CSV):
    news_df = pl.read_csv(PATH_TO_CSV)
    for col in news_df.columns:
        news_dict[col] = news_df[col].to_list()

def news_line(article):
    news_slide = QVBoxLayout()
    news_widget = QWidget()
    
    news_image = QLabel()
    # news_image.setFixedSize(800, 800)
    response = requests.get(article['image'])
    image_data = BytesIO(response.content)
    news_pixmap = QPixmap()
    news_pixmap.loadFromData(image_data.read())
    news_pixmap = news_pixmap.scaled(800, 800, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    news_image.setPixmap(news_pixmap)
    news_image.adjustSize()
    news_image.setAlignment(Qt.AlignCenter)
    
    news_source = QLabel(article['source'])
    news_source.setAlignment(Qt.AlignCenter)
    news_source.setFont(QFont('Arial', 30))
    news_source.setStyleSheet("font-weight: bold;")
    news_source.setWordWrap(True)
    news_source.adjustSize()
    
    news_headlines = QLabel(article['title'])
    news_headlines.setFont(QFont('Arial', 24))
    news_headlines.setAlignment(Qt.AlignCenter)
    news_headlines.setWordWrap(True)
    news_headlines.adjustSize()
    
    news_content = QLabel(article['content'])
    news_content.setFont(QFont('Arial', 16))
    news_content.setAlignment(Qt.AlignCenter)
    news_content.setWordWrap(True)
    news_content.adjustSize()
    
    news_link = QPushButton('Read More')
    news_link.clicked.connect(lambda: redirect_to_url(article['url']))
    
    news_slide.addWidget(news_image)
    news_slide.addWidget(news_source)
    news_slide.addWidget(news_headlines)
    news_slide.addWidget(news_content)
    news_slide.addWidget(news_link)
    
    news_widget.setLayout(news_slide)
    return news_widget
    
def redirect_to_url(url):
    webbrowser.open(url) 
    
def prev_article(news_list):
    current_idx[0] -= 1
    if current_idx[0] < 0:
        current_idx[0] = 0
        return
    news_list.setCurrentIndex(current_idx[0])
    
def next_article(news_list):
    current_idx[0] += 1
    if current_idx[0] >= len(ids):
        current_idx[0] = len(ids) - 1
    news_list.setCurrentIndex(current_idx[0])
    
def initialize_ui():
    fetch_latest_news()
    app = QApplication([])
    window = QWidget()
    window.setGeometry(50, 50, 1600, 1200)
    window.setWindowTitle("What's going on with the world?")
    app_layout = QHBoxLayout()
    
    countries_outer_layout = QVBoxLayout()
    countries_search_box = QLineEdit()
    countries_search_box.setPlaceholderText("Search for a country...")
    countries_outer_layout.addWidget(countries_search_box)
    
    countries_list = QScrollArea()
    countries_outer_layout.addWidget(countries_list)
    app_layout.addLayout(countries_outer_layout)
    
    countries_widget = QWidget()
    countries_inner_layout = QVBoxLayout()
    
    all_countries_label = QPushButton('ALL')
    countries_inner_layout.addWidget(all_countries_label)
    for country in set(news_dict['country']):
        country_label = QPushButton(country)
        countries_inner_layout.addWidget(country_label)
        
    countries_widget.setLayout(countries_inner_layout)
    countries_list.setWidget(countries_widget)
    
    news_layout = QHBoxLayout()
    
    prev_news_button = QPushButton('<')
    prev_news_button.setFont(QFont('Arial', 50))
    prev_news_button.clicked.connect(lambda: prev_article(news_list))
    news_layout.addWidget(prev_news_button)
    
    news_widget = QWidget()
    news_list = QStackedLayout()
    news_widget.setLayout(news_list)
    current_idx.append(0) 
    for i in range(len(news_dict['id'])):
        ids.append(news_dict['id'][i])
        article = {
            'source': news_dict['source'][i],
            'title': news_dict['title'][i],
            'url': news_dict['url'][i],
            'image': news_dict['image'][i],
            'content': news_dict['content'][i]
        }
        news_list.addWidget(news_line(article))
    #     news_layout.addWidget(news_line(article))
    news_layout.addWidget(news_widget)

    next_news_button = QPushButton('>')
    next_news_button.setFont(QFont('Arial', 50))
    next_news_button.clicked.connect(lambda: next_article(news_list))
    news_layout.addWidget(next_news_button)
    
    app_layout.addLayout(news_layout)
    
    app_layout.setStretchFactor(countries_outer_layout, 1)
    app_layout.setStretchFactor(news_layout, 4)

    window.setLayout(app_layout)
    window.show()
    app.exec_()

def fetch_latest_news():
    print("Fetching and analyzing news...")
    params = {
        'q': ' OR '.join(GEO_KEYWORDS),
        'language': 'en',
        'pageSize': 100,
        'apiKey': NEWS_API_KEY,
    }
    response = requests.get(NEWS_ENDPOINT, params=params)
    for article in response.json()['articles']:
        article['content'] = clean_content(article['content']) + ' ....'
        searchable_text = f'{article['title']} {article['description']} {article['content']}'.lower()
        is_geopolitical = any((keyword in searchable_text) for keyword in GEO_KEYWORDS)
        if is_geopolitical:
            article_countries = extract_countries(searchable_text)
            for country in article_countries:
                insert_article(article, country)         
    
    news_df = pl.DataFrame(news_dict)
    news_df.write_csv(PATH_TO_CSV)
    print('Saved news to news.csv...')

def clean_content(text):
    text = re.sub(r'\[\+\d+\s+chars\]$', '', text.strip())
    text = re.sub(r'\s*\b\w*…$', '', text.strip())
    text = re.sub(r'\s*\b\w*\.{3,}$', '', text.strip())
    text = html.unescape(text)
    return text

def insert_article(article, country):
    if(len(news_dict['id']) == 0):
        news_dict['id'].append(0)
    elif(article['title'] in news_dict['title']):
        return
    else:
        news_dict['id'].append(news_dict['id'][-1] + 1)
        
    news_dict['source'].append(article['source']['name'])
    news_dict['title'].append(article['title'])
    news_dict['description'].append(article['description'])
    news_dict['content'].append(article['content'])
    news_dict['url'].append(article['url'])
    news_dict['image'].append(article['urlToImage'])
    news_dict['country'].append(country.upper())
    
    if len(news_dict['id']) >= 1000:
        news_dict['id'].pop[0]
        news_dict['source'].pop[0]
        news_dict['title'].pop[0]
        news_dict['description'].pop[0]
        news_dict['content'].pop[0]
        news_dict['url'].pop[0]
        news_dict['image'].pop[0]
        news_dict['country'].pop[0]

def extract_countries(text):
    doc = nlp(text)
    countries_found = set()
    for ent in doc.ents:
        if ent.label_ == "GPE":
            countries = match_countries(ent.text)
            if len(countries) > 0:
                for country in countries:
                    countries_found.add(country)
    return list(countries_found)

def match_countries(entity_name):
    countries = []
    entity_name = entity_name.lower()
    entity_name = ABBREVIATIONS.get(entity_name, entity_name)
    for country in COUNTRIES:
            if entity_name == country:
                countries.append(country)
    return countries

def display_results():
    for article_id in news_dict['id']:
        print(f"\n{article_id}. {news_dict['country'][article_id]}")
        print(f"  - {news_dict['title'][article_id]}")
        print(f"  - {news_dict['description'][article_id]}")
        print(f"  - {news_dict['url'][article_id]}")

if __name__ == '__main__':
    initialize_ui()
    # display_results()