import * as tf from "@tensorflow/tfjs";
import * as mobilenet from "@tensorflow-models/mobilenet";
import { loadImage, random } from "../../helpers";

const STORAGE = "TransferLearning:image-training";
const LABELS = ["blue", "red"];
const DATASET_SIZE = 1000;
const IMAGE_SIZE = 256;

export default class ImageClassification {
  constructor() {}

  async dataSetToTensors() {
    // Tensor 2d - cada input tem 3 valores [[r,g,b]]
    // Vamos normalizar isto
    let images = [];
    let labels = [];
    for (let image of this.dataSet) {
      // Normalizados vão de 1 a 5, max de rgb é 255
      const imgData = await loadImage(image.src);
      images.push(imgData);
      //labels.push(color.label);
      labels.push(LABELS.indexOf(image.label));
    }

    // // One hot encoding = o vector de output como se fosse perfeito, ex -> label = blue [1,0,0]
    let labelsTensor = tf.tensor1d(labels, "int32");
    this.xs = tf.tensor2d(inputs);
    this.ys = tf.oneHot(labelsTensor, LABELS.length);

    console.log({ images, labels, xs: this.xs, ys: this.ys });
  }

  async loadDataSet() {
    this.dataSet = [];
    const images = [
      { src: "data/colors/training/blue1.png", label: "blue" },
      { src: "data/colors/training/blue2.png", label: "blue" },
      { src: "data/colors/validation/blue3.png", label: "blue" },
      { src: "data/colors/training/red1.png", label: "red" },
      { src: "data/colors/training/red2.png", label: "red" },
      { src: "data/colors/validation/red3.png", label: "red" }
    ];
    for (let i = 0; i < DATASET_SIZE; i++) {
      this.dataSet.push(images[random(5)]);
    }

    return this.dataSet;
  }

  async setup() {
    await this.loadDataSet();
    await this.dataSetToTensors();
    console.log(this.dataSet);
  }

  async classify() {}

  async run() {
    await this.setup();
    return true;
  }
}
