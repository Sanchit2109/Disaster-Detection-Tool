const express = require("express");
const axios = require("axios");
const mongoose = require("mongoose");
const cors = require("cors");
const dotenv = require("dotenv");

dotenv.config();
const app = express();
app.use(express.json());
app.use(cors());

// 🔹 MongoDB Connection
mongoose.connect(process.env.MONGO_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
});

// 🔹 Disaster Schema
const DisasterSchema = new mongoose.Schema({
  type: String,
  location: String,
  severity: Number,
  timestamp: { type: Date, default: Date.now },
});
const Disaster = mongoose.model("Disaster", DisasterSchema);

// 🔹 Fetch Real-time Disaster Data
app.get("/fetch-disaster-data", async (req, res) => {
  try {
    const weatherData = await axios.get(
      `https://api.openweathermap.org/data/2.5/weather?q=California&appid=${process.env.WEATHER_API_KEY}`
    );

    const earthquakeData = await axios.get(
      "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"
    );

    res.json({ weather: weatherData.data, earthquake: earthquakeData.data });
  } catch (error) {
    res.status(500).json({ error: "Error fetching disaster data" });
  }
});

// 🔹 Predict Disaster Using ML Model
app.post("/predict", async (req, res) => {
  try {
    // Ensure the request body contains valid features
    if (!req.body.features || !Array.isArray(req.body.features)) {
      return res.status(400).json({ error: "Invalid input format" });
    }

    // Call Flask ML API for prediction
    const response = await axios.post("http://127.0.0.1:5000/predict", req.body);

    // Extract prediction result
    const { risk_level } = response.data;

    // Save prediction to MongoDB
    const newDisaster = new Disaster({
      type: "Predicted Disaster",
      location: req.body.location || "Unknown",
      severity: parseInt(risk_level),
    });

    await newDisaster.save();

    res.json({ prediction: risk_level, message: "Prediction stored successfully!" });
  } catch (error) {
    console.error("ML API Error:", error.message);
    res.status(500).json({ error: "Error in ML prediction" });
  }
});

// Start Server on Port 3000
app.listen(3000, () => console.log("Backend running on port 3000"));
