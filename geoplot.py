"""
geoplot.py
----------

This visualization renders a 3-D plot of the data given the state
trajectory of a simulation, and the path of the property to render.

It generates an HTML file that contains code to render the plot
using Cesium Ion, and the GeoJSON file of data provided to the plot.

An example of its usage is as follows:

```py
from agent_torch.visualize import GeoPlot

# create a simulation
# ...

# create a visualizer
engine = GeoPlot(config, {
  cesium_token: "...",
  step_time: 3600,
  coordinates = "agents/consumers/coordinates",
  feature = "agents/consumers/money_spent",
})

# visualize in the runner-loop
for i in range(0, num_episodes):
  runner.step(num_steps_per_episode)
  engine.render(runner.state_trajectory)
```
"""

import re
import json

import pandas as pd
import numpy as np

from string import Template
from agent_torch.core.helpers import get_by_path

# HTML Template with embedded JavaScript using CesiumJS to render animated time-series data
geoplot_template = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Cesium Time-Series Heatmap Visualization</title>
  <script src="https://cesium.com/downloads/cesiumjs/releases/1.95/Build/Cesium/Cesium.js"></script>
  <link href="https://cesium.com/downloads/cesiumjs/releases/1.95/Build/Cesium/Widgets/widgets.css" rel="stylesheet"/>
  <style>
    #cesiumContainer {
      width: 100%;
      height: 100%;
    }
  </style>
</head>
<body>
  <div id="cesiumContainer"></div>
  <script>
    Cesium.Ion.defaultAccessToken = '$accessToken'; // Access Token for Cesium

    const viewer = new Cesium.Viewer('cesiumContainer'); // Main Cesium viewer

    // Function to interpolate between two colors based on factor [0, 1]
    function interpolateColor(color1, color2, factor) {
      const result = new Cesium.Color();
      result.red = color1.red + factor * (color2.red - color1.red);
      result.green = color1.green + factor * (color2.green - color1.green);
      result.blue = color1.blue + factor * (color2.blue - color1.blue);
      result.alpha = '$visualType' == 'size' ? 0.2 : color1.alpha + factor * (color2.alpha - color1.alpha);
      return result;
    }

    // Maps numerical value to interpolated color between BLUE and RED
    function getColor(value, min, max) {
      const factor = (value - min) / (max - min);
      return interpolateColor(Cesium.Color.BLUE, Cesium.Color.RED, factor);
    }

    // Maps value to pixel size (used for 'size' visualization)
    function getPixelSize(value, min, max) {
      const factor = (value - min) / (max - min);
      return 100 * (1 + factor); // Scale between 100 to 200
    }

    // Process GeoJSON into time-based agent tracking dictionary
    function processTimeSeriesData(geoJsonData) {
      const timeSeriesMap = new Map();
      let minValue = Infinity;
      let maxValue = -Infinity;

      geoJsonData.features.forEach((feature) => {
        const id = feature.properties.id;
        const time = Cesium.JulianDate.fromIso8601(feature.properties.time);
        const value = feature.properties.value;
        const coordinates = feature.geometry.coordinates;

        if (!timeSeriesMap.has(id)) {
          timeSeriesMap.set(id, []);
        }
        timeSeriesMap.get(id).push({ time, value, coordinates });

        // Track global min/max for feature scaling
        minValue = Math.min(minValue, value);
        maxValue = Math.max(maxValue, value);
      });

      return { timeSeriesMap, minValue, maxValue };
    }

    // Create animated Cesium entities for each agent
    function createTimeSeriesEntities(timeSeriesData, startTime, stopTime) {
      const dataSource = new Cesium.CustomDataSource('AgentTorch Simulation');

      for (const [id, timeSeries] of timeSeriesData.timeSeriesMap) {
        const entity = new Cesium.Entity({
          id: id,
          availability: new Cesium.TimeIntervalCollection([
            new Cesium.TimeInterval({ start: startTime, stop: stopTime }),
          ]),
          position: new Cesium.SampledPositionProperty(),
          point: {
            pixelSize: '$visualType' == 'size' ? new Cesium.SampledProperty(Number) : 10,
            color: new Cesium.SampledProperty(Cesium.Color),
          },
          properties: {
            value: new Cesium.SampledProperty(Number),
          },
        });

        // Add time-based samples for each point
        timeSeries.forEach(({ time, value, coordinates }) => {
          const position = Cesium.Cartesian3.fromDegrees(coordinates[0], coordinates[1]);
          entity.position.addSample(time, position);
          entity.properties.value.addSample(time, value);
          entity.point.color.addSample(time, getColor(value, timeSeriesData.minValue, timeSeriesData.maxValue));
          if ('$visualType' == 'size') {
            entity.point.pixelSize.addSample(time, getPixelSize(value, timeSeriesData.minValue, timeSeriesData.maxValue));
          }
        });

        dataSource.entities.add(entity);
      }

      return dataSource;
    }

    // Initialize globe animation with GeoJSON data
    const geoJsons = $data;
    const start = Cesium.JulianDate.fromIso8601('$startTime');
    const stop = Cesium.JulianDate.fromIso8601('$stopTime');

    viewer.clock.startTime = start.clone();
    viewer.clock.stopTime = stop.clone();
    viewer.clock.currentTime = start.clone();
    viewer.clock.clockRange = Cesium.ClockRange.LOOP_STOP;
    viewer.clock.multiplier = 3600; // 1 hour per second
    viewer.timeline.zoomTo(start, stop);

    for (const geoJsonData of geoJsons) {
      const timeSeriesData = processTimeSeriesData(geoJsonData);
      const dataSource = createTimeSeriesEntities(timeSeriesData, start, stop);
      viewer.dataSources.add(dataSource);
      viewer.zoomTo(dataSource);
    }
  </script>
</body>
</html>
"""

# Helper to extract nested variable using path like 'agents/consumers/coordinates'
def read_var(state, var):
    return get_by_path(state, re.split("/", var))

class GeoPlot:
    def __init__(self, config, options):
        """
        Setup GeoPlot engine with simulation config and rendering options.
        Options:
            cesium_token: Cesium Ion access token
            step_time: time between simulation steps (in seconds)
            coordinates: path to agent coordinates
            feature: path to agent-specific property to visualize
            visualization_type: 'color' or 'size'
        """
        self.config = config
        (
            self.cesium_token,
            self.step_time,
            self.entity_position,
            self.entity_property,
            self.visualization_type,
        ) = (
            options["cesium_token"],
            options["step_time"],
            options["coordinates"],
            options["feature"],
            options["visualization_type"],
        )

    def render(self, state_trajectory):
        """
        Render the entire trajectory into .geojson and .html output files.
        Each state stores the position and selected feature for each agent.
        """
        coords, values = [], []
        name = self.config["simulation_metadata"]["name"]
        geodata_path, geoplot_path = f"{name}.geojson", f"{name}.html"

        # For each episode, get final step's coordinates and feature values
        for i in range(0, len(state_trajectory) - 1):
            final_state = state_trajectory[i][-1]
            coords = np.array(read_var(final_state, self.entity_position)).tolist()
            values.append(
                np.array(read_var(final_state, self.entity_property)).flatten().tolist()
            )

        # Generate timestamp list using configured step_time
        start_time = pd.Timestamp.utcnow()
        timestamps = [
            start_time + pd.Timedelta(seconds=i * self.step_time)
            for i in range(
                self.config["simulation_metadata"]["num_episodes"]
                * self.config["simulation_metadata"]["num_steps_per_episode"]
            )
        ]

        # Build GeoJSON format: list of FeatureCollections (one per agent)
        geojsons = []
        for i, coord in enumerate(coords):
            features = []
            for time, value_list in zip(timestamps, values):
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [coord[1], coord[0]],  # longitude, latitude
                        },
                        "properties": {
                            "id": str(i),
                            "value": value_list[i],
                            "time": time.isoformat(),
                        },
                    }
                )
            geojsons.append({"type": "FeatureCollection", "features": features})

        # Write GeoJSON data file
        with open(geodata_path, "w", encoding="utf-8") as f:
            json.dump(geojsons, f, ensure_ascii=False, indent=2)

        # Substitute template placeholders and write final HTML file
        tmpl = Template(geoplot_template)
        with open(geoplot_path, "w", encoding="utf-8") as f:
            f.write(
                tmpl.substitute(
                    {
                        "accessToken": self.cesium_token,
                        "startTime": timestamps[0].isoformat(),
                        "stopTime": timestamps[-1].isoformat(),
                        "data": json.dumps(geojsons),
                        "visualType": self.visualization_type,
                    }
                )
            )
