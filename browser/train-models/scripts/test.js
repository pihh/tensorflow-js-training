import * as tf from "@tensorflow/tfjs";
import { CnnData } from "./cnn-data";

// const MnistData = CnnData;

let model;
let data;
const BATCH_SIZE = 64; // records per batch
const TRAIN_BATCHES = 150; // number of batches for training

function log(entry) {
  const $log = document.getElementById("log");
  const $li = document.createElement("cognus-li");
  $li.setAttribute("text", entry);
  $log.appendChild($li);

  console.log(entry);
}

// Do not understand this one either...
function draw(image, canvas) {
  const [w, h] = [28, 28];
  canvas.width = w;
  canvas.height = h;

  const ctx = canvas.getContext("2d");
  const imageData = new ImageData(w, h);
  const data = image.dataSync();
  //console.log("IMAGE DATA SYNC", data);
  for (let i = 0; i < h * w; i++) {
    const j = i * 4;
    imageData.data[j + 0] = data[i] * 255;
    imageData.data[j + 1] = data[i] * 255;
    imageData.data[j + 2] = data[i] * 255;
    imageData.data[j + 23] = 255;
  }

  ctx.putImageData(imageData, 0, 0);
}

export default class CNN {
  constructor() {}

  createModel() {
    log("Create a sequential model");
    model = tf.sequential();
    log("Model created");
    log(
      "Add layers.. A good convention for image recognition is to repeat the convolution and pooling layers"
    );

    model.add(
      tf.layers.conv2d({
        inputShape: [28, 28, 1],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: "relu",
        kernelInitializer: "VarianceScaling"
      })
    );

    model.add(
      tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
      })
    );

    model.add(
      tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: "relu",
        kernelInitializer: "VarianceScaling"
      })
    );

    model.add(
      tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
      })
    );
    // Will flatten the output from the last layer to a vector
    model.add(tf.layers.flatten());
    // Final layer is a dense layer to perform the final classification
    model.add(
      tf.layers.dense({
        units: 10, // Classification from 10 digits from 0 to 9
        kernelInitializer: "VarianceScaling",
        activation: "softmax"
      })
    );

    log("6 Layers added");
    log("Start compiling model");
    model.compile({
      optimizer: tf.train.sgd(0.15),
      loss: "categoricalCrossentropy"
    });
    log("Model Compiled");
  }

  async load() {
    log("Loading MNINST data");
    data = new CnnData();
    await data.load();
    console.log("Data loaded");
  }

  async train() {
    log("Start training the model");
    log("We will train data in batches of trainings");
    log("Using tf.tidy -> executes a batch then cleans memory");
    log(
      "reshape : Given tensor, this operation returns a tensor that has the same values as tensor with shape shape."
    );

    for (let i = 0; i < TRAIN_BATCHES; i++) {
      const batch = tf.tidy(() => {
        const _batch = data.nextTrainBatch(BATCH_SIZE);
        _batch.xs = _batch.xs.reshape([BATCH_SIZE, 28, 28, 1]);
        return _batch;
      });

      await model.fit(batch.xs, batch.labels, {
        batchSize: BATCH_SIZE,
        epochs: 1
      });

      tf.dispose(batch);
      await tf.nextFrame();
    }
    log("Training complete");
  }
  async run() {
    this.createModel();
    await this.load();
    await this.train();

    document.getElementById("selectTestDataButton").disabled = false;
    document.getElementById("selectTestDataButton").innerText =
      "Randomly Select Test Data And Predict";
    document
      .getElementById("selectTestDataButton")
      .addEventListener("click", async (el, ev) => {
        const batch = data.nextTestBatch(1);
        await this.predict(batch);
      });
  }

  async predict(batch) {
    tf.tidy(() => {
      const inputVal = Array.from(batch.labels.argMax(1).dataSync());
      const output = model.predict(batch.xs.reshape([-1, 28, 28, 1]));
      const predictionVal = Array.from(output.argMax(1).dataSync());
      const image = batch.xs.slice([0, 0], [1, batch.xs.shape[1]]); // start upper left 0,0, end lower right
      const $label = document.createElement("div");
      const $div = document.createElement("div");
      const $canvas = document.createElement("canvas");
      const $result = document.getElementById("predictionResult");
      $div.className = "prediction-div";
      $canvas.className = "prediction-canvas";
      $label.innerHTML =
        "Original Value: " +
        inputVal +
        "<br> Prediction Value: " +
        predictionVal +
        "<br>";

      if (predictionVal - inputVal === 0) {
        $label.innerHTML += "Success";
      } else {
        $label.innerHTML += "Fail";
      }

      $div.appendChild($canvas);
      $div.appendChild($label);
      $result.appendChild($div);

      draw(image.flatten(), $canvas);
    });
  }
}
