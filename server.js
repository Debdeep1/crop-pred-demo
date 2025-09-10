// server.js
import http from "http";
import fs from "fs";
import { parse } from "csv-parse";
import url from "url";
import MultivariateLinearRegression from "ml-regression-multivariate-linear";
import { RandomForestRegression as RF } from "ml-random-forest";

/* -------------------- Globals -------------------- */
let linModel, rfModel, mins, maxs, categories;

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

function normalize(X) {
  return X.map((row) =>
    row.map((val, j) => (val - mins[j]) / (maxs[j] - mins[j] + 1e-9))
  );
}

/* -------------------- Train models -------------------- */
function trainModel(callback) {
  const rows = [];
  fs.createReadStream("./indian_crop_climate_data.csv")
    .pipe(parse({ columns: true, trim: true }))
    .on("data", (row) => rows.push(row))
    .on("end", () => {
      let data = rows;
      ["crop_type", "region", "soil_type"].forEach(
        (col) => (data = encodeCategorical(data, col))
      );

      const X = [];
      const y = [];

      for (let r of data) {
        const features = [
          parseFloat(r.crop_type),
          parseFloat(r.region),
          parseFloat(r.soil_type),
          parseFloat(r.temperature_c),
          parseFloat(r.rainfall_mm),
          parseFloat(r.humidity_percent),
        ];
        const label = parseFloat(r.production_tonnes_per_hectare);

        // Skip bad rows
        if (features.some((v) => isNaN(v)) || isNaN(label)) continue;

        X.push(features);
        y.push(label);
      }

      if (X.length === 0) {
        console.error("❌ No valid training data found.");
        return;
      }

      // normalize features
      const XT = X[0].map((_, j) => X.map((row) => row[j]));
      mins = XT.map((col) => Math.min(...col));
      maxs = XT.map((col) => Math.max(...col));
      const Xnorm = normalize(X);

      // Convert y to 2D for ml-regression
      const y2D = y.map((v) => [v]);

      // Train Linear Regression
      linModel = new MultivariateLinearRegression(Xnorm, y2D);

      // Train Random Forest
      rfModel = new RF({
        nEstimators: 50,
        maxFeatures: 0.8,
        replacement: true,
        seed: 42,
      });
      rfModel.train(Xnorm, y);

      categories = {
        crop_type: [...new Set(rows.map((r) => r.crop_type))],
        region: [...new Set(rows.map((r) => r.region))],
        soil_type: [...new Set(rows.map((r) => r.soil_type))],
      };

      console.log(
        `✅ Trained models: Linear Regression + Random Forest on ${X.length} samples.`
      );
      callback();
    });
}

/* -------------------- HTTP Server -------------------- */
trainModel(() => {
  const server = http.createServer((req, res) => {
    const parsedUrl = url.parse(req.url, true);

    if (parsedUrl.pathname === "/predict" && req.method === "GET") {
      const q = parsedUrl.query;
      const encCrop = categories.crop_type.indexOf(q.crop_type);
      const encRegion = categories.region.indexOf(q.region);
      const encSoil = categories.soil_type.indexOf(q.soil_type);

      const features = [
        encCrop,
        encRegion,
        encSoil,
        parseFloat(q.temperature_c),
        parseFloat(q.rainfall_mm),
        parseFloat(q.humidity_percent),
      ];

      if (features.some((v) => isNaN(v))) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "Invalid or missing parameters" }));
        return;
      }

      const normFeatures = features.map(
        (v, j) => (v - mins[j]) / (maxs[j] - mins[j] + 1e-9)
      );

      const linPred = linModel.predict([normFeatures])[0][0]; // unwrap 2D
      const rfPred = rfModel.predict([normFeatures])[0];
      const area = parseFloat(q.area_hectares);

      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(
        JSON.stringify({
          linear: {
            yield_per_hectare: linPred.toFixed(2),
            total_yield: (linPred * area).toFixed(2),
          },
          random_forest: {
            yield_per_hectare: rfPred.toFixed(2),
            total_yield: (rfPred * area).toFixed(2),
          },
        })
      );
    } else if (parsedUrl.pathname === "/categories") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify(categories));
    } else {
      fs.readFile("./index.html", (err, content) => {
        if (err) {
          res.writeHead(500);
          res.end("Error loading page");
          return;
        }
        res.writeHead(200, { "Content-Type": "text/html" });
        res.end(content);
      });
    }
  });

  server.listen(3000, () =>
    console.log("🌍 Server running at http://localhost:3000")
  );
});
