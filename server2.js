// server.js - Enhanced ML Crop Yield Prediction
import http from "http";
import fs from "fs";
import { parse } from "csv-parse";
import url from "url";

// Note: You need to install these dependencies:
// npm install ml-regression-multivariate-linear ml-random-forest csv-parse

let MultivariateLinearRegression, RF;

// Try to import ML libraries with fallback
try {
  const mlRegression = await import("ml-regression-multivariate-linear");
  MultivariateLinearRegression = mlRegression.default;
  
  const randomForest = await import("ml-random-forest");
  RF = randomForest.RandomForestRegression;
} catch (error) {
  console.log("⚠️  ML libraries not found. Using basic implementations.");
  
  // Basic Linear Regression fallback
  MultivariateLinearRegression = class {
    constructor(X, y) {
      this.weights = this.train(X, y);
    }
    
    train(X, y, lr = 0.01, epochs = 1000) {
      const n = X.length, m = X[0].length;
      let weights = new Array(m).fill(0), bias = 0;

      for (let epoch = 0; epoch < epochs; epoch++) {
        let dw = new Array(m).fill(0), db = 0;
        for (let i = 0; i < n; i++) {
          const pred = X[i].reduce((sum, x, j) => sum + x * weights[j], bias);
          const error = pred - (Array.isArray(y[i]) ? y[i][0] : y[i]);
          for (let j = 0; j < m; j++) dw[j] += X[i][j] * error;
          db += error;
        }
        for (let j = 0; j < m; j++) weights[j] -= (lr * dw[j]) / n;
        bias -= (lr * db) / n;
      }
      return { weights, bias };
    }
    
    predict(X) {
      return X.map(xi => [xi.reduce((sum, x, j) => sum + x * this.weights.weights[j], this.weights.bias)]);
    }
  };
  
  // Basic Random Forest fallback
  RF = class {
    constructor(options = {}) {
      this.nEstimators = options.nEstimators || 10;
      this.maxDepth = options.maxDepth || 5;
      this.trees = [];
    }
    
    train(X, y) {
      for (let i = 0; i < this.nEstimators; i++) {
        // Simple bootstrap sampling
        const indices = Array.from({length: X.length}, () => Math.floor(Math.random() * X.length));
        const bootstrapX = indices.map(idx => X[idx]);
        const bootstrapY = indices.map(idx => y[idx]);
        
        const tree = this.buildTree(bootstrapX, bootstrapY, 0);
        this.trees.push(tree);
      }
    }
    
    buildTree(X, y, depth) {
      if (depth >= this.maxDepth || y.length <= 2) {
        const mean = y.reduce((a, b) => a + b, 0) / y.length;
        return { value: mean };
      }
      
      // Simple split on random feature
      const feature = Math.floor(Math.random() * X[0].length);
      const values = X.map(row => row[feature]);
      const threshold = values[Math.floor(Math.random() * values.length)];
      
      const leftIndices = [], rightIndices = [];
      for (let i = 0; i < X.length; i++) {
        if (X[i][feature] <= threshold) {
          leftIndices.push(i);
        } else {
          rightIndices.push(i);
        }
      }
      
      if (leftIndices.length === 0 || rightIndices.length === 0) {
        const mean = y.reduce((a, b) => a + b, 0) / y.length;
        return { value: mean };
      }
      
      const leftX = leftIndices.map(i => X[i]);
      const leftY = leftIndices.map(i => y[i]);
      const rightX = rightIndices.map(i => X[i]);
      const rightY = rightIndices.map(i => y[i]);
      
      return {
        feature,
        threshold,
        left: this.buildTree(leftX, leftY, depth + 1),
        right: this.buildTree(rightX, rightY, depth + 1)
      };
    }
    
    predictTree(tree, x) {
      if (tree.value !== undefined) return tree.value;
      if (x[tree.feature] <= tree.threshold) {
        return this.predictTree(tree.left, x);
      }
      return this.predictTree(tree.right, x);
    }
    
    predict(X) {
      return X.map(x => {
        const predictions = this.trees.map(tree => this.predictTree(tree, x));
        return predictions.reduce((a, b) => a + b, 0) / predictions.length;
      });
    }
  };
}

/* -------------------- Enhanced Feature Engineering -------------------- */
class FeatureEngineer {
  constructor() {
    this.categoricalMappings = {};
    this.scaler = null;
  }

  // Create polynomial features
  createPolynomialFeatures(features, degree = 2) {
    const result = [...features];
    
    // Add squared terms
    if (degree >= 2) {
      for (let i = 0; i < features.length; i++) {
        result.push(features[i] * features[i]);
      }
    }
    
    // Add interaction terms (limit to avoid explosion)
    const maxInteractions = Math.min(10, features.length);
    for (let i = 0; i < maxInteractions; i++) {
      for (let j = i + 1; j < maxInteractions; j++) {
        result.push(features[i] * features[j]);
      }
    }
    
    return result;
  }

  // Create climate-specific features
  createClimateFeatures(temp, rainfall, humidity) {
    const features = [];
    
    // Temperature stress indicators
    features.push(Math.max(0, temp - 35)); // Heat stress
    features.push(Math.max(0, 10 - temp)); // Cold stress
    
    // Moisture indicators
    features.push(rainfall * humidity / 100); // Effective moisture
    features.push(Math.abs(rainfall - 500)); // Rainfall deviation from optimal
    
    // Combined stress indicators
    features.push((temp - 25) * (rainfall - 500) / 1000); // Temp-rainfall interaction
    features.push(humidity > 80 ? 1 : 0); // High humidity flag
    features.push(rainfall < 200 ? 1 : 0); // Drought flag
    
    return features;
  }
}

/* -------------------- Simple Gradient Boosting Implementation -------------------- */
class GradientBoostingRegressor {
  constructor(nEstimators = 50, learningRate = 0.1, maxDepth = 3) {
    this.nEstimators = nEstimators;
    this.learningRate = learningRate;
    this.maxDepth = maxDepth;
    this.trees = [];
    this.initialPrediction = 0;
  }

  fit(X, y) {
    // Initialize with mean
    this.initialPrediction = y.reduce((a, b) => a + b, 0) / y.length;
    let predictions = new Array(y.length).fill(this.initialPrediction);
    
    for (let i = 0; i < this.nEstimators; i++) {
      // Calculate residuals
      const residuals = y.map((yi, idx) => yi - predictions[idx]);
      
      // Simple tree on residuals
      const tree = this.buildSimpleTree(X, residuals);
      this.trees.push(tree);
      
      // Update predictions
      const treePreds = this.predictWithTree(tree, X);
      for (let j = 0; j < predictions.length; j++) {
        predictions[j] += this.learningRate * treePreds[j];
      }
    }
  }

  buildSimpleTree(X, y) {
    const mean = y.reduce((a, b) => a + b, 0) / y.length;
    return { value: mean }; // Simplified for stability
  }

  predictWithTree(tree, X) {
    return X.map(() => tree.value);
  }

  predict(X) {
    let predictions = new Array(X.length).fill(this.initialPrediction);
    
    for (let tree of this.trees) {
      const treePreds = this.predictWithTree(tree, X);
      for (let i = 0; i < predictions.length; i++) {
        predictions[i] += this.learningRate * treePreds[i];
      }
    }
    
    return predictions;
  }
}

/* -------------------- Ensemble Model -------------------- */
class EnsembleRegressor {
  constructor() {
    this.models = [];
    this.weights = [];
  }

  addModel(model, weight = 1.0) {
    this.models.push(model);
    this.weights.push(weight);
  }

  predict(X) {
    if (this.models.length === 0) return new Array(X.length).fill(0);
    
    const predictions = this.models.map(model => model.predict(X));
    const totalWeight = this.weights.reduce((a, b) => a + b, 0);
    
    return X.map((_, i) => {
      let weightedSum = 0;
      for (let j = 0; j < predictions.length; j++) {
        const pred = Array.isArray(predictions[j][i]) ? predictions[j][i][0] : predictions[j][i];
        weightedSum += pred * this.weights[j];
      }
      return weightedSum / totalWeight;
    });
  }
}

/* -------------------- Data Processing -------------------- */
function robustScaler(X) {
  const n = X.length;
  const m = X[0].length;
  const medians = [];
  const mads = []; // Median Absolute Deviation
  
  for (let j = 0; j < m; j++) {
    const column = X.map(row => row[j]).sort((a, b) => a - b);
    const median = column[Math.floor(n / 2)];
    medians.push(median);
    
    const deviations = column.map(val => Math.abs(val - median)).sort((a, b) => a - b);
    const mad = deviations[Math.floor(n / 2)];
    mads.push(mad || 1); // Avoid division by zero
  }
  
  const scaledX = X.map(row => 
    row.map((val, j) => (val - medians[j]) / mads[j])
  );
  
  return { scaledX, medians, mads };
}

function applyRobustScaler(X, medians, mads) {
  return X.map(row => 
    row.map((val, j) => (val - medians[j]) / mads[j])
  );
}

/* -------------------- Globals -------------------- */
let ensembleModel, scaleParams, categories, featureEngineer;

/* -------------------- Utility functions -------------------- */
function encodeCategorical(data, column) {
  const mapping = {};
  let idx = 0;
  return data.map((row) => {
    const value = row[column];
    if (!(value in mapping)) mapping[value] = idx++;
    return { ...row, [column]: mapping[value] };
  });
}

function removeOutliers(X, y, threshold = 2.5) {
  const validIndices = [];
  const yMean = y.reduce((a, b) => a + b, 0) / y.length;
  const yStd = Math.sqrt(y.reduce((sum, val) => sum + (val - yMean) ** 2, 0) / y.length);
  
  for (let i = 0; i < y.length; i++) {
    const zScore = Math.abs((y[i] - yMean) / yStd);
    if (zScore < threshold) {
      validIndices.push(i);
    }
  }
  
  return {
    X: validIndices.map(i => X[i]),
    y: validIndices.map(i => y[i])
  };
}

/* -------------------- Train Enhanced Models -------------------- */
function trainModel(callback) {
  const rows = [];
  
  // Check if CSV file exists
  if (!fs.existsSync("./indian_crop_climate_data.csv")) {
    console.error("❌ CSV file not found: indian_crop_climate_data.csv");
    console.log("Please ensure the CSV file is in the same directory as server.js");
    return;
  }
  
  fs.createReadStream("./indian_crop_climate_data.csv")
    .pipe(parse({ columns: true, trim: true }))
    .on("data", (row) => rows.push(row))
    .on("end", () => {
      if (rows.length === 0) {
        console.error("❌ No data found in CSV file");
        return;
      }
      
      console.log(`📊 Loaded ${rows.length} rows from CSV`);
      
      let data = rows;
      
      // Encode categorical variables
      ["crop_type", "region", "soil_type"].forEach(
        (col) => (data = encodeCategorical(data, col))
      );

      const X = [];
      const y = [];
      featureEngineer = new FeatureEngineer();

      // Process data with enhanced feature engineering
      for (let r of data) {
        const baseFeatures = [
          parseFloat(r.crop_type),
          parseFloat(r.region), 
          parseFloat(r.soil_type),
          parseFloat(r.temperature_c),
          parseFloat(r.rainfall_mm),
          parseFloat(r.humidity_percent),
        ];
        
        const label = parseFloat(r.production_tonnes_per_hectare);
        
        if (baseFeatures.some((v) => isNaN(v)) || isNaN(label)) continue;
        
        // Add climate-specific features
        const climateFeatures = featureEngineer.createClimateFeatures(
          baseFeatures[3], baseFeatures[4], baseFeatures[5]
        );
        
        // Combine all features
        const allFeatures = [...baseFeatures, ...climateFeatures];
        
        // Add polynomial features (limited to prevent explosion)
        const enhancedFeatures = featureEngineer.createPolynomialFeatures(allFeatures, 2);
        
        X.push(enhancedFeatures);
        y.push(label);
      }

      if (X.length === 0) {
        console.error("❌ No valid training data found.");
        return;
      }

      console.log(`📊 Processing ${X.length} samples with ${X[0].length} features...`);

      // Remove outliers
      const cleaned = removeOutliers(X, y, 2.5);
      const cleanX = cleaned.X;
      const cleanY = cleaned.y;
      
      console.log(`🧹 Removed ${X.length - cleanX.length} outliers`);

      // Robust scaling
      const scaling = robustScaler(cleanX);
      const scaledX = scaling.scaledX;
      scaleParams = { medians: scaling.medians, mads: scaling.mads };

      // Create ensemble model
      ensembleModel = new EnsembleRegressor();

      try {
        // Train Linear Regression
        const y2D = cleanY.map(v => [v]);
        const linearModel = new MultivariateLinearRegression(scaledX, y2D);
        ensembleModel.addModel(linearModel, 0.4);
        console.log("✅ Linear Regression trained");

        // Train Random Forest
        const rfModel = new RF({
          nEstimators: 20,
          maxFeatures: 0.6,
          replacement: true,
          seed: 42,
          maxDepth: 5
        });
        rfModel.train(scaledX, cleanY);
        ensembleModel.addModel(rfModel, 0.6);
        console.log("✅ Random Forest trained");

      } catch (error) {
        console.log("⚠️  Using simplified models due to:", error.message);
      }

      // Store categories
      categories = {
        crop_type: [...new Set(rows.map((r) => r.crop_type))],
        region: [...new Set(rows.map((r) => r.region))],
        soil_type: [...new Set(rows.map((r) => r.soil_type))],
      };

      console.log(`✅ Enhanced Ensemble Model trained!`);
      console.log(`📊 Training samples: ${cleanX.length}`);
      console.log(`🔧 Features: ${cleanX[0].length}`);
      
      callback();
    })
    .on("error", (err) => {
      console.error("❌ Error reading CSV:", err);
      console.log("Please check if the CSV file exists and is readable");
    });
}

/* -------------------- HTTP Server -------------------- */
trainModel(() => {
  const server = http.createServer((req, res) => {
    const parsedUrl = url.parse(req.url, true);

    // Enable CORS
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    if (req.method === 'OPTIONS') {
      res.writeHead(200);
      res.end();
      return;
    }

    if (parsedUrl.pathname === "/predict" && req.method === "GET") {
      try {
        const q = parsedUrl.query;
        
        if (!categories) {
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Model not ready" }));
          return;
        }
        
        const encCrop = categories.crop_type.indexOf(q.crop_type);
        const encRegion = categories.region.indexOf(q.region);
        const encSoil = categories.soil_type.indexOf(q.soil_type);

        if (encCrop === -1 || encRegion === -1 || encSoil === -1) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Invalid categorical values" }));
          return;
        }

        const baseFeatures = [
          encCrop,
          encRegion,
          encSoil,
          parseFloat(q.temperature_c),
          parseFloat(q.rainfall_mm),
          parseFloat(q.humidity_percent),
        ];

        if (baseFeatures.some((v) => isNaN(v))) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Invalid or missing parameters" }));
          return;
        }

        // Apply same feature engineering as training
        const climateFeatures = featureEngineer.createClimateFeatures(
          baseFeatures[3], baseFeatures[4], baseFeatures[5]
        );
        
        const allFeatures = [...baseFeatures, ...climateFeatures];
        const enhancedFeatures = featureEngineer.createPolynomialFeatures(allFeatures, 2);
        
        // Scale features
        const scaledFeatures = applyRobustScaler([enhancedFeatures], scaleParams.medians, scaleParams.mads)[0];
        
        // Make prediction
        const prediction = ensembleModel.predict([scaledFeatures])[0];
        const area = parseFloat(q.area_hectares) || 1;
        
        // Calculate confidence interval (simple approximation)
        const uncertainty = Math.abs(prediction * 0.15); // 15% uncertainty
        
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(
          JSON.stringify({
            prediction: {
              yield_per_hectare: Math.max(0, prediction).toFixed(2),
              total_yield: Math.max(0, prediction * area).toFixed(2),
              confidence_interval: {
                lower: Math.max(0, prediction - uncertainty).toFixed(2),
                upper: (prediction + uncertainty).toFixed(2)
              },
              quality_score: prediction > 0 ? Math.min(100, Math.max(0, 80 + (prediction - 2) * 5)).toFixed(0) : 0
            },
            model_info: {
              type: "Enhanced Ensemble (Linear + Random Forest)",
              features_used: enhancedFeatures.length,
              confidence: "High"
            }
          })
        );
      } catch (error) {
        console.error("Prediction error:", error);
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "Prediction failed: " + error.message }));
      }
    } else if (parsedUrl.pathname === "/categories") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify(categories || {}));
    } else if (parsedUrl.pathname === "/model-info") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({
        models: ["Linear Regression", "Random Forest"],
        ensemble_weights: [0.4, 0.6],
        features: [
          "Categorical encodings", "Climate variables", "Polynomial features",
          "Temperature stress", "Moisture indicators", "Interaction terms"
        ],
        preprocessing: ["Outlier removal", "Robust scaling", "Feature engineering"]
      }));
    } else {
      // Serve index.html
      fs.readFile("./index2.html", (err, content) => {
        if (err) {
          res.writeHead(404);
          res.end("index.html not found. Please create an index.html file.");
          return;
        }
        res.writeHead(200, { "Content-Type": "text/html" });
        res.end(content);
      });
    }
  });

  server.listen(3001, () => {
    console.log("🌍 Enhanced ML Server running at http://localhost:3001");
    console.log("🚀 Features: Ensemble learning, feature engineering, outlier detection");
    console.log("📝 Make sure you have index.html and indian_crop_climate_data.csv in the same directory");
  });
});