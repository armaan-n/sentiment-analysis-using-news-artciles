# About

This project is a web app that utilizes a keras model to analyze the sentiment surrounding any given subject using news articles written in the past 24 hours.

# How To Run

It's recommended that you do this in a virtual environment.

1. Clone the repository.
2. Ensure the requirements in `requirements.txt` are satisfied (run `pip install -r requirements.txt`).
3. Set the `NEWS_API_KEY` environment variable to an API key that can be generated [here](https://newsapi.org/).
4. Run the `index.py` file in the `api` directory using using a python 3.9 interpreter.
5. Call `npm run dev` within the `site/Site/market_sentiment` directory (ensure you have the `next` npm package installed, i.e. run `npm install next`).
6. Once both applications are running, the website should be accessible locally.
