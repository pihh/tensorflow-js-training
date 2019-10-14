import * as tf from "@tensorflow/tfjs";
import * as smartcrop from "smartcrop";

function randomString(length = 5) {
  return Math.random()
    .toString(36)
    .replace(/[^a-z]+/g, "")
    .substr(0, length);
}

function modelPath(name) {
  return `indexeddb://${name}`;
}

export const arrayMax = function(arr) {
  return Math.max.apply(null, arr);
};

export const arrayMin = function(arr) {
  return Math.min.apply(null, arr);
};

// Turns data set into 0 -> 1 arrays
export const normalizeToPercentageData = function(array) {
  const min = arrayMin(array);
  const max = arrayMax(array);
  const normalized = array.map(el => {
    el = (el - min) / (max - min);
    return el;
  });

  return {
    normalized,
    min,
    max,
    normalizedMin: arrayMin(normalized),
    normalizedMax: arrayMin(normalized)
  };
};

export const saveModel = async function(model, name) {
  try {
    const save = await model.save(modelPath(name));
    return save;
  } catch (ex) {
    // ... Silence is golden
    return false;
  }
};

export const loadModel = async function(name) {
  try {
    const model = await tf.loadLayersModel(modelPath(name));
    return model;
  } catch (ex) {
    // ... silence is golden
    return false;
  }
};

export const random = function() {
  const args = [...arguments];
  let min = 0;
  let max = args[0];
  if (args.length === 2) {
    min = args[0];
    max = args[1];
  }

  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min)) + min;
};

export class Canvas {
  constructor(w = 200, h = 200, config = {}) {
    const id = config.id || randomString();
    const canvas = document.createElement("canvas");
    canvas.id = id;
    canvas.width = w;
    canvas.height = h;
    canvas.style.zIndex = 8;
    canvas.style.border = "1px solid";
    canvas.style.marginRight = "10px";
    canvas.style.marginBottom = "10px";
    canvas.addEventListener("click", event => {
      console.log("canvas clicked");
      console.log("coordinates", this.clickCoordinates(event));
    });
    this.id = id;
    this.width = w;
    this.height = h;
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
  }

  clickCoordinates(event) {
    let totalOffsetX = 0;
    let totalOffsetY = 0;
    let canvasX = 0;
    let canvasY = 0;
    let currentElement = this.canvas;

    do {
      totalOffsetX += currentElement.offsetLeft - currentElement.scrollLeft;
      totalOffsetY += currentElement.offsetTop - currentElement.scrollTop;
    } while ((currentElement = currentElement.offsetParent));

    canvasX = event.pageX - totalOffsetX;
    canvasY = event.pageY - totalOffsetY;
    console.log({ canvasX, canvasY, currentElement });

    return { x: canvasX, y: canvasY };
  }
}
export const loadImage = async function(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "";
    img.src = src;
    img.onload = () => resolve(img);
    img.onerror = err => reject(err);
  });
};
export const cropImage = async function(
  image,
  config = {
    width: 100,
    height: 100
  }
) {
  const img = await loadImage(image);
  const result = await smartcrop.crop(img, {
    width: config.width,
    height: config.height
  });
  console.log(result);
  return result;
};

cropImage(
  "https://cdn.vox-cdn.com/thumbor/9KWZcsQc2pnKa73CVfdG8lp-bu8=/0x0:2040x1360/1200x675/filters:focal(1228x281:1554x607)/cdn.vox-cdn.com/uploads/chorus_image/image/63920405/IMG_B990CF208719_1.0.jpeg"
);
