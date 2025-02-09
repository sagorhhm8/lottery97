import time
import pandas as pd
import numpy as np
import joblib
import schedule
import threading
from selenium import webdriver
from selenium.webdriver.common.by import By
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext

# ✅ Setup Selenium WebDriver
options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(options=options)

# ✅ Function to Fetch Latest Wingo Result
def fetch_latest_wingo_results():
    try:
        driver.get("https://97lottery.com/#/")
        time.sleep(3)

        latest_number = driver.find_element(By.XPATH, '//*[@id="result-number"]').text
        latest_result = "Big" if int(latest_number) >= 5 else "Small"

        print(f"Fetched Result: {latest_number} ({latest_result})")

        data = pd.DataFrame([[latest_number, latest_result]], columns=["Number", "Big/Small"])
        data.to_csv("wingo_results.csv", mode='a', header=False, index=False)

        return latest_number, latest_result

    except Exception as e:
        print(f"Error fetching results: {e}")
        return None, None

# ✅ Function to Train Machine Learning Model (XGBoost)
def train_ml_model():
    data = pd.read_csv("wingo_results.csv", names=["Number", "Big/Small"])
    
    # Convert data to numeric values
    data["Big/Small"] = data["Big/Small"].apply(lambda x: 1 if x == "Big" else 0)
    data["Number"] = data["Number"].astype(int)
    
    # Use last 3 numbers as features
    data["Prev1"] = data["Number"].shift(1)
    data["Prev2"] = data["Number"].shift(2)
    data["Prev3"] = data["Number"].shift(3)
    data = data.dropna()
    
    # Define features and target
    X = data[["Prev1", "Prev2", "Prev3"]]
    y = data["Big/Small"]
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, "wingo_model.pkl")
    
    # Show accuracy
    accuracy = model.score(X_test, y_test)
    print(f"ML Model Trained - Accuracy: {accuracy * 100:.2f}%")

# ✅ Function to Predict Next Wingo Outcome
def predict_next_wingo():
    data = pd.read_csv("wingo_results.csv", names=["Number", "Big/Small"]).tail(3)
    if len(data) < 3:
        return "Not enough data"

    # Extract last 3 numbers
    last_numbers = data["Number"].astype(int).values
    model = joblib.load("wingo_model.pkl")
    
    # Make prediction
    prediction = model.predict([[last_numbers[2], last_numbers[1], last_numbers[0]]])[0]
    return "Big" if prediction == 1 else "Small"

# ✅ Telegram Bot Setup
TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text("🎰 Welcome to Wingo Predictor Bot! Use /predict to get the next prediction.")

def predict(update: Update, context: CallbackContext) -> None:
    predicted_result = predict_next_wingo()
    
    # Fetch past 5 results
    past_results = pd.read_csv("wingo_results.csv", names=["Number", "Big/Small"]).tail(5)
    history = ""
    correct_count = 0

    for _, row in past_results.iterrows():
        actual = row["Big/Small"]
        predicted = "✅" if actual == predicted_result else "❌"
        history += f"🔹 {row['Number']} → {actual} {predicted}\n"
        if actual == predicted_result:
            correct_count += 1

    accuracy = (correct_count / len(past_results)) * 100
    message = f"🔮 **Next Prediction:** {predicted_result}\n\n📊 **Past Results:**\n{history}\n🎯 **Accuracy:** {accuracy:.2f}%"
    
    update.message.reply_text(message)

# ✅ Schedule Data Fetching Every Minute
def schedule_scraper():
    while True:
        fetch_latest_wingo_results()
        train_ml_model()
        time.sleep(60)

# ✅ Start Web Scraper in Background
threading.Thread(target=schedule_scraper, daemon=True).start()

# ✅ Start Telegram Bot
updater = Updater(TOKEN, use_context=True)
dispatcher = updater.dispatcher

dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(CommandHandler("predict", predict))

print("🚀 Wingo Prediction Bot is Running...")
updater.start_polling()
