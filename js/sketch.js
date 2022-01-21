// ml5.js: Training a Convolutional Neural Network for Image Classification (Mask)
// The Coding Train / Daniel Shiffman
// https://thecodingtrain.com/learning/ml5/8.4-cnn-image-classification.html
// https://youtu.be/hWurN0XhzLY
// https://editor.p5js.org/codingtrain/sketches/ogxO8har_
// (mask) https://editor.p5js.org/codingtrain/sketches/tKLoeUD0u



//Div Console
function dvconsole(txt){
  var para = document.createElement("p");
  var node = document.createTextNode(txt);
  para.appendChild(node);
  var dvc = document.getElementById("console");
  dvc.appendChild(para);
  dvc.scrollTop = dvc.scrollHeight;

}


let video;
let videoSize = 64;
let ready = false;

let pixelBrain;
let label = '';

function setup() {
  createCanvas(600, 600);
  video = createCapture(VIDEO, videoReady);
  video.size(videoSize, videoSize);
  video.hide();

  let options = {
    inputs: [64, 64, 4],
    task: 'imageClassification',
    debug: true,
  };
  pixelBrain = ml5.neuralNetwork(options);
}

function loaded() {
  pixelBrain.train(
    {
      epochs: 50,
    },
    finishedTraining
  );
}

function finishedTraining() {
  console.log('training complete');
  dvconsole('Entrenamiento Completado');
  classifyVideo();
}

function classifyVideo() {
  let inputImage = {
    image: video,
  };
  pixelBrain.classify(inputImage, gotResults);
}

function gotResults(error, results) {
  if (error) {
    return;
  }
  label = results[0].label;
  classifyVideo();
}

function keyPressed() {
  if (key == 't') {
    pixelBrain.normalizeData();
    pixelBrain.train(
      {
        epochs: 50,
      },
      finishedTraining
    );
  } else if (key == 's') {
    pixelBrain.save();
  } else if (key == 'm') {
    //for(var i=1; i<=100; i++ ){
      addExample('cm');
      dvconsole('Añadiendo ejemplo: cm');
    //}
  } else if (key == 'n') {
    //for(var i=1; i<=100; i++ ){
      addExample('nm');
      dvconsole('Añadiendo ejemplo: nm');
    //}
  }else if (key == 'v') {
    //for(var i=1; i<=100; i++ ){
      addExample('vv');
      dvconsole('Añadiendo ejemplo: vv');
    //}
  }
}

function addExample(label) {
  let inputImage = {
    image: video,
  };
  let target = {
    label,
  };
  console.log('Adding example: ' + label);
  pixelBrain.addData(inputImage, target);
}

// Video is ready!
function videoReady() {
  ready = true;
}

function draw() {
  background(0);
  if (ready) {
    image(video, 0, 0, width, height);
  }

  textSize(64);
  textAlign(CENTER, CENTER);
  if(label=="nm"){
    fill(color(255,0,0));
    text("Usa Cubrebocas!", width / 2, height / 2);
  }else if(label=="cm"){
    fill(color(0,255,0));
    text("Lindo Cubrebocas!", width / 2, height / 2);
  }else if (label=="vv") {
    fill(255);
    text("Sin Persona", width / 2, height / 2);
  }
}
