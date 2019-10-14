import * as tf from "@tensorflow/tfjs";
import * as mobilenet from "@tensorflow-models/mobilenet";
import localforage from "localforage";
import { random, Canvas } from "../../helpers";

const STORAGE = "TransferLearning:color-training";
const LABELS = ["red", "green", "blue"];

export default class ColorClassification {
  constructor() {}

  async loadDataSet() {
    this.dataSet = await localforage.getItem(STORAGE);
    if (!this.dataSet) {
      this.dataSet = {
        blue: [],
        green: [],
        red: []
      };
    }
  }

  createCanvas(w = 200, h = 200) {
    const body = document.getElementById("canvas-color");
    const canvas = new Canvas();
    this.canvas = canvas.canvas;
    this.ctx = canvas.ctx;

    body.appendChild(this.canvas);
  }

  backgroundColor(ctx, r, g, b, x0, y0, x1, y1) {
    const context = ctx || this.ctx;
    context.fillStyle = `rgb(${r || this.r}, ${g || this.g}, ${b || this.b})`;
    context.fillRect(
      x0 || 0,
      y0 || 0,
      x1 || this.canvas.width,
      y1 || this.canvas.height
    );
  }

  rgb() {
    this.r = Math.floor(random(256));
    this.g = Math.floor(random(256));
    this.b = Math.floor(random(256));
  }

  next() {
    this.rgb();
    this.backgroundColor();

    try {
      const xs = tf.tensor2d([[this.r / 255, this.g / 255, this.b / 255]]);
      const results = this.model.predict(xs);
      const index = results.argMax(1).dataSync(); // Vai buscar o indice do maior
      console.clear();
      console.log("Predict", { color: LABELS[index], results });
      results.print();
    } catch (ex) {
      console.warn(ex);
    }
  }

  paintCanvas(color) {
    const step = 10;
    const canvas = new Canvas();

    let x = 0;
    let y = 0;
    let appendTo = document.getElementById("canvas-color");

    appendTo.appendChild(canvas.canvas);

    for (let i = 0; i < this.dataSet[color].length; i++) {
      const d = this.dataSet[color][i];
      this.backgroundColor(canvas.ctx, d.r, d.g, d.b, x, y, step, step);
      x += step;
      if (x >= canvas.width) {
        x = 0;
        y += 10;
      }
    }
  }

  dataSetToArray() {
    let a = [];
    LABELS.forEach(key => {
      a = a.concat(
        this.dataSet[key].map(el => {
          el.label = key;
          return el;
        })
      );
    });

    //a = shuffle(a);

    this.dataSetArray = a;
  }

  dataSetToTensors() {
    this.dataSetToArray();

    // Tensor 2d - cada input tem 3 valores [[r,g,b]]
    // Vamos normalizar isto
    let colors = [];
    let labels = [];
    for (let color of this.dataSetArray) {
      // Normalizados vão de 1 a 5, max de rgb é 255
      colors.push([color.r / 255, color.g / 255, color.b / 255]);
      //labels.push(color.label);
      labels.push(LABELS.indexOf(color.label));
    }

    // One hot encoding = o vector de output como se fosse perfeito, ex -> label = blue [1,0,0]
    let labelsTensor = tf.tensor1d(labels, "int32");
    this.xs = tf.tensor2d(colors);
    this.ys = tf.oneHot(labelsTensor, LABELS.length);
  }

  async setModel() {
    // layers vão do inicio para o fim
    this.model = tf.sequential();
    let hidden = tf.layers.dense({
      units: 16, // nós da layer
      activation: "sigmoid", // Apanha todas as multiplicações pelos pesos e comprime em um valor
      inputDim: 3 // RGB
    });

    let output = tf.layers.dense({
      units: 3, // Labels possiveis
      activation: "softmax" // Apanha todas as multiplicações pelos pesos e comprime em um valor - geral distribuição de probabilidades
    });

    this.model.add(hidden);
    this.model.add(output);

    // Optimization -> meanSquaredError -> categoricalCrossEntropy
    // CategoricalCrossEntropy e boa pa comparar distribuições de probabilidade
    const lr = 0.2; // learning rate
    const loss = "categoricalCrossentropy"; // erro entre a distribuição de probabilidade do certo com a aproximação
    const optimizer = tf.train.sgd(lr);

    this.model.compile({
      optimizer,
      loss
    });

    const result = await this.model.fit(this.xs, this.ys, {
      epochs: 50,
      validationData: 0.1,
      shuffle: true,
      callbacks: {
        onTrainBegin: () => console.log("Training Start"),
        onTrainEnd: () => console.log("Training End"),
        onBatchEnd: tf.nextFrame,
        onEpochEnd: async (num, logs) => {
          console.log("Epoch: " + num);
          console.log("Loss: " + logs.loss);
          return true;
        }
      }
    });

    return result;
  }

  async setup() {
    await this.loadDataSet();

    this.createCanvas(400, 400);
    this.rgb();
    this.backgroundColor();

    document.getElementById("next").addEventListener("click", () => {
      this.next();
    });

    const colors = ["red", "blue", "green"];
    for (let i = 0; i < colors.length; i++) {
      const color = colors[i];
      document.getElementById(color).addEventListener("click", () => {
        this.classify(color);
      });
    }

    this.paintCanvas("blue");
    return true;
  }

  async loadMobilenet() {
    return mobilenet.load();
  }

  async loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = "";
      img.src = src;
      img.onload = () => resolve(tf.browser.fromPixels(img));
      img.onerror = err => reject(err);
    });
  }

  async classify(color) {
    const r = this.r;
    const g = this.g;
    const b = this.g;

    this.dataSet[color].push({ r, g, b });
    await localforage.setItem(STORAGE, this.dataSet);

    return this.next();
  }

  async run() {
    await this.setup();
    this.dataSetToTensors();
    const results = await this.setModel();

    console.log({ results });
  }
}
