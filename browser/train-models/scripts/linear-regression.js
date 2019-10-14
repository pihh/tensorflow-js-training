import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import localForage from "localforage";
import { normalizeToPercentageData } from "../../helpers.js";

const URL = "https://storage.googleapis.com/tfjs-tutorials/carsData.json";
const STORAGE = "lesson#1:linear_regression";

export default class LinearRegression {
  constuctor() {
    [
      "START",
      `Our goal is to train a model that will take one number, Horsepower and learn to predict one number, Miles per Gallon. Remember that one-to-one mapping, as it will be important for the next section. We are going to feed these examples, the horsepower and the MPG, to a neural network that will learn from these examples a formula (or function) to predict MPG given horsepower. This learning from examples for which we have the correct answers is called Supervised Learning.`
    ].forEach(l => console.log(l));
    // Instaciate data
    this.data = [];
    this.inputs = [];
    this.labels = [];
    this.model = {};
    this.inputMax = 0;
    this.inputMin = 0;
    this.labelMax = 0;
    this.labelMin = 0;
  }

  async getTrainedModel() {
    return new Promise((res, rej) => {
      localForage.getItem(STORAGE, function(err, value) {
        if (err) rej(err);
        res(value);
      });
    });
  }

  async setTrainedModel(v) {
    return new Promise((res, rej) => {
      localForage.setItem(STORAGE, v, function(err, value) {
        if (err) rej(err);
        res(value);
      });
    });
  }

  // SYNC
  createModel() {
    [
      "",
      "CREATE MODEL",
      `Model architecture: Model architecture is just a fancy way of saying "which functions will the model run when it is executing", or alternatively "what algorithm will our model use to compute its answers".`,
      `This model is sequential because its inputs flow straight down to its output.`,
      `This adds a hidden layer to our network. A dense layer is a type of layer that multiplies its inputs by a matrix (called weights) and then adds a number (called the bias) to the result. As this is the first layer of the network, we need to define our inputShape. The inputShape is [1] because we have 1 number as our input (the horsepower of a given car).`,
      `units sets how big the weight matrix will be in the layer. By setting it to 1 here we are saying there will be 1 weight for each of the input features of the data.`
    ].forEach(l => console.log(l));

    // Add an output layer
    this.model = tf.sequential();

    // Add a single hidden layer
    this.model.add(
      tf.layers.dense({ inputShape: [1], units: 50, useBias: true })
    );

    // Add an output layer
    this.model.add(
      tf.layers.dense({ units: 1, useBias: true, activation: "sigmoid" })
    );

    return this.model;
  }

  testModel() {
    [
      "",
      "TEST MODEL",
      `Now that our model is trained, we want to make some predictions. Let's evaluate the model by seeing what it predicts for a uniform range of numbers of low to high horsepowers.`
    ].forEach(l => console.log(l));

    // Generate predictions for a uniform range of numbers between 0 and 1;
    // We un-normalize the data by doing the inverse of the min-max scaling
    // that we did earlier.
    const [xs, preds] = tf.tidy(() => {
      const xs = tf.linspace(0, 1, 100);
      const preds = this.model.predict(xs.reshape([100, 1]));

      const unNormXs = xs
        .mul(this.inputMax.sub(this.inputMin))
        .add(this.inputMin);

      const unNormPreds = preds
        .mul(this.labelMax.sub(this.labelMin))
        .add(this.labelMin);

      console.log(
        `.dataSync() is a method we can use to get a typedarray of the values stored in a tensor. This allows us to process those values in regular JavaScript. This is a synchronous version of the .data() method which is generally preferred.`
      );
      // Un-normalize the data
      return [unNormXs.dataSync(), unNormPreds.dataSync()];
    });

    const predictedPoints = Array.from(xs).map((val, i) => {
      return { x: val, y: preds[i] };
    });

    const originalPoints = this.data.map(d => ({
      x: d.horsepower,
      y: d.mpg
    }));

    tfvis.render.scatterplot(
      { name: "Model Predictions vs Original Data" },
      {
        values: [originalPoints, predictedPoints],
        series: ["original", "predicted"]
      },
      {
        xLabel: "Horsepower",
        yLabel: "MPG",
        height: 300
      }
    );
  }

  convertToTensor() {
    [
      "",
      "CONVERT TO TENSOR",
      `To get the performance benefits of TensorFlow.js that make training machine learning models practical, we need to convert our data to tensors. We will also perform a number of transformations on our data that are best practices, namely shuffling and normalization.`
    ].forEach(l => console.log(l));

    // Executes the provided function fn and after it is executed, cleans up all intermediate tensors allocated by fn except those returned by fn.
    // Prevents memory leaks
    return tf.tidy(() => {
      // Step 1. Shuffle the data
      tf.util.shuffle(this.data);

      // Step 2. Convert data to Tensor
      this.inputs = this.data.map(d => d.horsepower);
      this.labels = this.data.map(d => d.mpg);

      // I want to check if they are passing the right normalization
      // Just uncheck this
      // console.log(normalizeToPercentageData(this.inputs));
      // console.log(normalizeToPercentageData(this.labels));

      const inputTensor = tf.tensor2d(this.inputs, [this.inputs.length, 1]);
      const labelTensor = tf.tensor2d(this.labels, [this.labels.length, 1]);

      //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
      this.inputMax = inputTensor.max();
      this.inputMin = inputTensor.min();
      this.labelMax = labelTensor.max();
      this.labelMin = labelTensor.min();

      const normalizedInputs = inputTensor
        .sub(this.inputMin)
        .div(this.inputMax.sub(this.inputMin));
      const normalizedLabels = labelTensor
        .sub(this.labelMin)
        .div(this.labelMax.sub(this.labelMin));

      this.inputs = normalizedInputs;
      this.labels = normalizedLabels;
      return {
        inputs: normalizedInputs,
        labels: normalizedLabels,
        // Return the min/max bounds so we can use them later.
        inputMax: this.inputMax,
        inputMin: this.inputMin,
        labelMax: this.labelMax,
        labelMin: this.labelMin
      };
    });
  }

  // ASYNC
  async loadData() {
    const carsDataReq = await fetch(URL);
    const carsData = await carsDataReq.json();
    this.data = carsData
      .map(car => ({
        mpg: car.Miles_per_Gallon,
        horsepower: car.Horsepower
      }))
      .filter(car => car.mpg != null && car.horsepower != null);

    return this.data;
  }

  async trainModel() {
    [
      "",
      "TRAIN MODEL",
      `With our model instance created and our data represented as tensors we have everything in place to start the training process.`,
      `optimizer: This is the algorithm that is going to govern the updates to the model as it sees examples. There are many optimizers available in TensorFlow.js. Here we have picked the adam optimizer as it is quite effective in practice and requires no configuration.`,
      `this is a function that will tell the model how well it is doing on learning each of the batches (data subsets) that it is shown. Here we use meanSquaredError to compare the predictions made by the model with the true values.`,
      `batchSize: refers to the size of the data subsets that the model will see on each iteration of training. Common batch sizes tend to be in the range 32-512. There isn't really an ideal batch size for all problems and it is beyond the scope of this tutorial to describe the mathematical motivations for various batch sizes.`,
      `epochs: refers to the number of times the model is going to look at the entire dataset that you provide it. Here we will take 50 iterations through the dataset.`
    ].forEach(l => console.log(l));

    // Prepare the model for training.
    this.model.compile({
      optimizer: tf.train.adam(),
      loss: tf.losses.meanSquaredError,
      metrics: ["mse"]
    });

    const batchSize = 32;
    const epochs = 50;

    this.trainedModel = await this.model.fit(this.inputs, this.labels, {
      batchSize,
      epochs,
      shuffle: true,
      callbacks: tfvis.show.fitCallbacks(
        { name: "Training Performance" },
        ["loss", "mse"],
        { height: 200, callbacks: ["onEpochEnd"] }
      )
    });
  }

  async run() {
    [
      `LINEAR REGRESSION:
      * Load the JSON data.
      * Get the necessary values [x => horsePower, y => mpg].
      * Render the 2d graph HorsePower vs MPG.
      * Create Model:
           * Define Layers
      * Convert to tensor:
           * Shuffle data
           * Convert data to tensor
           * Normalize the data ( making it fit between 0 and 1)
      * Train the model:
           * Choose optimizer - Adam - simple and no config
           * Choose loss function - Mean Squared Error in this case
           * Choose batchSize: refers to the size of the data subsets that the model will see on each iteration of training.
           * Choose epochs: Number of times this will loop through the model.
      * Test the model.
           * Generate predictions for a uniform range of numbers between 0 and 1;
           * Un-normalize the data by doing the inverse of the min-max scaling that we did earlier.
      * Improve the Model:
          * Added activation: sigmoid to output and 50 units to hidden layer, it now almost fits perfectly
      `
    ].forEach(l =>
      console.log(`%c ${l}`, "background: black ; color: rgb( 0, 255, 0 )")
    );
    // Load and plot the original input data that we are going to train on.
    const data = await this.loadData();
    const values = data.map(d => ({
      x: d.horsepower,
      y: d.mpg
    }));

    tfvis.render.scatterplot(
      { name: "Horsepower v MPG" },
      { values },
      {
        xLabel: "Horsepower",
        yLabel: "MPG",
        height: 300
      }
    );

    // More code will be added below
    // Create the model - Defines the layers
    this.createModel();
    tfvis.show.modelSummary({ name: "Model Summary" }, this.model);

    // Convert the data to a form we can use for training.
    // Shuffle data
    // Convert data to tensor
    // Normalize the data ( making it fit between 0 and 1)
    const tensorData = this.convertToTensor();

    // Train the model
    // Choose optimizer - Adam - simple and no config
    // Choose loss function - Mean Squared Error in this case
    // Choose batchSize: refers to the size of the data subsets that the model will see on each iteration of training.
    // Choose epochs: Number of times this will loop through the model.
    await this.trainModel();

    // Test Model
    // Check how the predictions match with the model
    this.testModel();
  }
}
