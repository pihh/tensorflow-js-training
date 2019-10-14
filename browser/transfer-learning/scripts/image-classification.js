import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import * as mobilenet from "@tensorflow-models/mobilenet";
import { loadImage, random, stringToArrayBuffer } from "../../helpers";

const STORAGE = "TransferLearning:image-training";
const LABELS = ["like", "unlike"];

const IMAGE_CHANNELS = 1;
const NUM_DATASET_ELEMENTS = 500;
const NUM_TRAIN_ELEMENTS = (NUM_DATASET_ELEMENTS * 4) / 5;
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS / 5;
const NUM_CLASSES = LABELS.length;
const NUM_OUTPUT_CLASSES = 2;
const IMAGE_LENGTH = 256;
const IMAGE_WIDTH = IMAGE_LENGTH;
const IMAGE_HEIGHT = IMAGE_LENGTH;
const IMAGE_SIZE = IMAGE_LENGTH * IMAGE_LENGTH;
const BATCH_SIZE = 5;

function generateRandomImageDataSet() {
  const dataSet = [];
  const images = [
    { src: "data/colors/training/blue1.png", label: "like" },
    { src: "data/colors/training/blue2.png", label: "like" },
    { src: "data/colors/validation/blue3.png", label: "like" },
    { src: "data/colors/training/red1.png", label: "unlike" },
    { src: "data/colors/training/red2.png", label: "unlike" },
    { src: "data/colors/validation/red3.png", label: "unlike" }
  ];
  for (let i = 0; i < NUM_DATASET_ELEMENTS; i++) {
    dataSet.push(images[random(5)]);
  }

  return dataSet;
}

class Data {
  constructor() {
    this.shuffledTrainIndex = 0;
    this.shuffledTestIndex = 0;
  }
  async load() {
    // Gerar um dataset random enquanto n temos o nosso
    const dataset = generateRandomImageDataSet();
    const labels = [];
    // No MNIST temos uma unica imagem, que tornamos em várias pequenas. Aqui temos várias imagens
    // Talvez fosse interessante meter isto em chunks por causa de probs de memória
    const datasetBytesBuffer = new ArrayBuffer(
      NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4
    );

    for (let i = 0; i < NUM_DATASET_ELEMENTS; i++) {
      // Criar uma imagem para cada uma dos datasets
      const datasetBytesView = new Float32Array(
        datasetBytesBuffer,
        i * IMAGE_SIZE * 4,
        IMAGE_SIZE
      );
      labels.push(LABELS.indexOf(dataset[i].label));
      await new Promise((resolve, reject) => {
        const img = new Image();
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");
        img.crossOrigin = "";
        img.onload = () => {
          img.width = img.naturalWidth;
          img.height = img.naturalHeight;

          canvas.width = img.width;
          canvas.height = img.height;
          // We dont have a chunk size here because we will have a full array in the end
          ctx.drawImage(img, 0, 0, img.width, img.height);

          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
          for (let j = 0; j < imageData.data.length / 4; j++) {
            // All channels hold an equal value since the image is grayscale, so
            // just read the red channel.
            datasetBytesView[j] = imageData.data[j * 4] / 255;
          }
          resolve();
        };
        img.src = dataset[i].src;
      });
    }

    this.datasetImages = new Float32Array(datasetBytesBuffer);
    this.datasetLabels = new Uint8Array(stringToArrayBuffer(labels.join()));

    // Create shuffled indices into the train/test set for when we select a
    // random dataset element for training / validation.
    this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS);
    this.testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS);

    // Slice the the images and labels into train and test sets.
    this.trainImages = this.datasetImages.slice(
      0,
      IMAGE_SIZE * NUM_TRAIN_ELEMENTS
    );
    this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    this.trainLabels = this.datasetLabels.slice(
      0,
      NUM_CLASSES * NUM_TRAIN_ELEMENTS
    );
    this.testLabels = this.datasetLabels.slice(
      NUM_CLASSES * NUM_TRAIN_ELEMENTS
    );
  }

  nextTrainBatch(batchSize) {
    return this.nextBatch(
      batchSize,
      [this.trainImages, this.trainLabels],
      () => {
        this.shuffledTrainIndex =
          (this.shuffledTrainIndex + 1) % this.trainIndices.length;
        return this.trainIndices[this.shuffledTrainIndex];
      }
    );
  }

  nextTestBatch(batchSize) {
    return this.nextBatch(batchSize, [this.testImages, this.testLabels], () => {
      this.shuffledTestIndex =
        (this.shuffledTestIndex + 1) % this.testIndices.length;
      return this.testIndices[this.shuffledTestIndex];
    });
  }

  nextBatch(batchSize, data, index) {
    const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE);
    const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES);

    for (let i = 0; i < batchSize; i++) {
      const idx = index();

      const image = data[0].slice(
        idx * IMAGE_SIZE,
        idx * IMAGE_SIZE + IMAGE_SIZE
      );
      batchImagesArray.set(image, i * IMAGE_SIZE);

      const label = data[1].slice(
        idx * NUM_CLASSES,
        idx * NUM_CLASSES + NUM_CLASSES
      );
      batchLabelsArray.set(label, i * NUM_CLASSES);
    }

    const xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE]);
    const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES]);

    return { xs, labels };
  }
}

export default class ImageClassification {
  constructor() {}

  getModel() {
    const model = tf.sequential();
    console.log(model);

    // In the first layer of our convolutional neural network we have
    // to specify the input shape. Then we specify some parameters for
    // the convolution operation that takes place in this layer.
    model.add(
      tf.layers.conv2d({
        inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: "relu",
        kernelInitializer: "varianceScaling"
      })
    );

    // The MaxPooling layer acts as a sort of downsampling using max values
    // in a region instead of averaging.
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

    // Repeat another conv2d + maxPooling stack.
    // Note that we have more filters in the convolution.
    model.add(
      tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: "relu",
        kernelInitializer: "varianceScaling"
      })
    );
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

    // Now we flatten the output from the 2D filters into a 1D vector to prepare
    // it for input into our last layer. This is common practice when feeding
    // higher dimensional data to a final classification output layer.
    model.add(tf.layers.flatten());

    // Our last layer is a dense layer which has 10 output units, one for each
    // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).

    model.add(
      tf.layers.dense({
        units: NUM_OUTPUT_CLASSES,
        kernelInitializer: "varianceScaling",
        activation: "softmax"
      })
    );

    // Choose an optimizer, loss function and accuracy metric,
    // then compile and return the model
    const optimizer = tf.train.adam();
    model.compile({
      optimizer: optimizer,
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"]
    });

    this.model = model;
    return model;
  }

  doPrediction(model, data, testDataSize = NUM_TEST_ELEMENTS) {
    const testData = this.dataset.nextTestBatch(testDataSize);
    const testxs = testData.xs.reshape([
      testDataSize,
      IMAGE_WIDTH,
      IMAGE_HEIGHT,
      1
    ]);
    const labels = testData.labels.argMax([-1]);
    const preds = this.model.predict(testxs).argMax([-1]);

    testxs.dispose();
    return [preds, labels];
  }

  async showAccuracy() {
    const [preds, labels] = this.doPrediction(this.model, this.dataset);
    const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
    const container = { name: "Accuracy", tab: "Evaluation" };
    tfvis.show.perClassAccuracy(container, classAccuracy, LABELS);

    labels.dispose();
  }

  async showConfusion() {
    const [preds, labels] = this.doPrediction(this.model, this.dataset);
    const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
    const container = { name: "Confusion Matrix", tab: "Evaluation" };
    tfvis.render.confusionMatrix(
      container,
      { values: confusionMatrix },
      LABELS
    );

    labels.dispose();
  }

  async showExamples(data) {
    // Create a container in the visor
    const surface = tfvis
      .visor()
      .surface({ name: "Input Data Examples", tab: "Input Data" });

    // Get the examples
    const examples = data.nextTestBatch(20);
    const numExamples = examples.xs.shape[0];

    // Create a canvas element to render each example
    for (let i = 0; i < numExamples; i++) {
      const imageTensor = tf.tidy(() => {
        // Reshape the image to 28x28 px
        return examples.xs
          .slice([i, 0], [1, examples.xs.shape[1]])
          .reshape([IMAGE_LENGTH, IMAGE_LENGTH, 1]);
      });

      const canvas = document.createElement("canvas");
      canvas.width = IMAGE_LENGTH;
      canvas.height = IMAGE_LENGTH;
      canvas.style = "margin: 4px;";
      await tf.browser.toPixels(imageTensor, canvas);
      surface.drawArea.appendChild(canvas);

      imageTensor.dispose();
    }
  }

  async loadDataSet() {
    this.dataset = new Data();
    await this.dataset.load();
  }

  async train() {
    const metrics = ["loss", "val_loss", "acc", "val_acc"];
    const container = {
      name: "Model Training",
      styles: { height: "1000px" }
    };
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

    const TRAIN_DATA_SIZE = NUM_TRAIN_ELEMENTS;
    const TEST_DATA_SIZE = NUM_TEST_ELEMENTS;

    const [trainXs, trainYs] = tf.tidy(() => {
      const d = this.dataset.nextTrainBatch(TRAIN_DATA_SIZE);
      return [
        d.xs.reshape([TRAIN_DATA_SIZE, IMAGE_LENGTH, IMAGE_LENGTH, 1]),
        d.labels
      ];
    });

    const [testXs, testYs] = tf.tidy(() => {
      const d = this.dataset.nextTestBatch(TEST_DATA_SIZE);
      return [
        d.xs.reshape([TEST_DATA_SIZE, IMAGE_LENGTH, IMAGE_LENGTH, 1]),
        d.labels
      ];
    });

    return this.model.fit(trainXs, trainYs, {
      batchSize: BATCH_SIZE,
      validationData: [testXs, testYs],
      epochs: 5,
      shuffle: true,
      callbacks: fitCallbacks
    });
  }

  async setup() {
    await this.loadDataSet();
    await this.showExamples(this.dataset);
    console.log("Did Show Examples");
    console.log("Will get Model");
    this.getModel();
    console.log("Did get Model", this.model);
    tfvis.show.modelSummary({ name: "Model Architecture" }, this.model);
  }

  async predict() {
    const testData = this.dataset.nextTestBatch(1);
    const testxs = testData.xs.reshape([1, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
    const labels = testData.labels.argMax([-1]);
    const preds = this.model.predict(testxs).argMax([-1]);

    console.log({ testData, testxs, labels, preds });
    testxs.dispose();
    return [preds, labels];
  }

  async run() {
    try {
      console.log("Will setup");
      await this.setup();
      console.log("Did Setup");
      console.log("Will Train");
      await this.train();
      console.log("Did Train");
      await this.showAccuracy();
      await this.showConfusion();
      document.getElementById("predict").addEventListener("click", () => {
        this.predict();
      });
      return true;
    } catch (ex) {
      console.warn(ex);
    }
  }
}
