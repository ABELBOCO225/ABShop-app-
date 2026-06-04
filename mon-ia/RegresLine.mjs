import * as tf from "@tensorflow/tfjs";
const xs = tf.tensor([1, 2, 3, 4]);
const ys = tf.tensor([2, 4, 6, 8]);
const model = tf.sequential();

model.add(tf.layers.dense({
  units: 1,
  inputShape: [1]
}));

model.compile({
  optimizer: "sgd",
  loss: "meanSquaredError"
});

await model.fit(xs, ys, {
  epochs: 1500
});

const output = model.predict(tf.tensor([6,14,25,47]));
output.print();

console.log("Poids et biais :");
const weights = model.getWeights();
const w = await weights[0].array();
const b = await weights[1].array();

console.log("w =", w);
console.log("b =", b);
