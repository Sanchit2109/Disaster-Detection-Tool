from flask import Flask, request, jsonify
import requests
import tweepy
import joblib
import spacy
from transformers import pipeline
from flask_cors import CORS

# Load trained model & vectorizer
model = joblib.load("disaster_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Load NLP model for location detection
nlp = spacy.load("en_core_web_sm")

# Load BERT-based sentiment analysis model
bert_classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# API Keys
TWITTER_BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAMFiywEAAAAAPtMzyfPOug6HDWwFtR%2BLrBaM2qM%3DKfHEyvNvgKiOmA7VCyIQnFldvhTHZHSVVd7TF9fYPUPDmeLNld"
NEWS_API_KEY = "6242744179024a4b9fbb77444c43977e"
WEATHER_API_KEY = "ce45ef9fe546adf905b3d3bd1274e1ed"

# Initialize Twitter client
client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)

# Flask app
app = Flask(__name__)
CORS(app)

# Function to classify tweet/news as Disaster or Not
def classify_text(text):
    transformed_text = vectorizer.transform([text])
    prediction = model.predict(transformed_text)[0]
    return "Disaster" if prediction == 1 else "Not a Disaster"

# Extract location from text
def extract_location(text):
    doc = nlp(text)
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    return locations if locations else ["Unknown Location"]

@app.route("/get_news", methods=["GET"])
def get_news():
    url = f"https://newsapi.org/v2/everything?q=disaster&apiKey={NEWS_API_KEY}"
    response = requests.get(url).json()

    results = []
    for article in response["articles"][:5]:
        classification = classify_text(article["title"])
        location = extract_location(article["title"])

        results.append({
            "title": article["title"],
            "url": article["url"],
            "classification": classification,
            "location": location
        })

    return jsonify(results)

# Fetch real-time disaster tweets
@app.route("/get_tweets", methods=["GET"])
def get_tweets():
    query = "(earthquake OR flood OR tsunami OR cyclone OR wildfire OR landslide) \
              (alert OR damage OR rescue OR help) -is:retweet lang:en"

    try:
        tweets = client.search_recent_tweets(query=query, tweet_fields=["text"], max_results=5)
        
        if not tweets.data:
            return jsonify({"message": "No recent disaster tweets found."})

        results = []
        for tweet in tweets.data:
            classification = classify_text(tweet.text)
            location = extract_location(tweet.text)
            sentiment = bert_classifier(tweet.text)[0]["label"]

            results.append({
                "text": tweet.text,
                "classification": classification,
                "location": location,
                "sentiment": sentiment
            })

        return jsonify(results)

    except tweepy.errors.BadRequest as e:
        return jsonify({"error": "Invalid API request. Check query parameters.", "details": str(e)}), 400
    except tweepy.errors.TooManyRequests as e:
        return jsonify({"error": "Rate limit exceeded. Try again later.", "details": str(e)}), 429
    except Exception as e:
        return jsonify({"error": "An unexpected error occurred.", "details": str(e)}), 500

# Fetch real-time weather alerts
@app.route("/get_weather", methods=["GET"])
def get_weather():
    city = request.args.get("city", default="New York", type=str)
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    response = requests.get(url).json()

    weather_data = {
        "city": response["name"],
        "country": response["sys"]["country"],
        "weather": response["weather"][0]["description"],
        "temperature_C": response["main"]["temp"],
        "temperature_F": round((response["main"]["temp"] * 9/5) + 32, 2),
        "humidity": response["main"]["humidity"],
        "visibility": response.get("visibility", "N/A"),
        "wind_speed_kmh": round(response["wind"]["speed"] * 3.6, 2),
        "wind_direction": response["wind"]["deg"],
    }
    
    return jsonify(weather_data)

# Fetch real-time earthquake data from USGS
@app.route("/get_earthquakes", methods=["GET"])
def get_earthquakes():
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        earthquakes = [
            {
                "magnitude": quake["properties"]["mag"],
                "location": quake["properties"]["place"],
                "time": quake["properties"]["time"],
                "url": quake["properties"]["url"]
            }
            for quake in data["features"][:5]  # Get first 5 results
        ]
        return jsonify(earthquakes)
    else:
        return jsonify({"error": "Failed to fetch earthquake data"}), 500

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Disaster Detection API is Running"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
