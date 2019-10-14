import * as tf from "@tensorflow/tfjs";

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
