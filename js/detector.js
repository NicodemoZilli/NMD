
let video;
let videoSize = 64;
let ready = false;

let classifier;
let pixelBrain;
let mobilenet;
let label = 'Cargando Modelo';


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
  modelReady();

}

// Video is ready!
function videoReady() {
  ready = true;
}
function  modelReady(){

  let modelDetails = {
   model: 'modeldata/model.json',
   metadata: 'modeldata/model_meta.json',
   weights: 'modeldata/model.weights.bin'
  };

  pixelBrain.load(modelDetails,customModelReady);
}

function customModelReady(){
  label = 'Modelo Cargado';
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
  }else if( label=="Cargando Modelo" || label=="Modelo Cargado"){
    fill(255);
    text(label, width / 2, height / 2);
  }
}
