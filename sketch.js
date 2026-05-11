let video;
let facemesh;
let handpose;

let predictions = [];
let handPredictions = [];
let maskImgs = [];

let faceModelLoaded = false;
let handModelLoaded = false;

let currentMaskIndex = 0;
let maskCycleCounter = 0;

const MAX_FACES = 2;

const landmarkSequence1 = [409, 270, 269, 267, 0, 37, 39, 40, 185, 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291];
const landmarkSequence2 = [76, 77, 90, 180, 85, 16, 315, 404, 320, 307, 306, 408, 304, 303, 302, 11, 72, 73, 74, 184];

const leftEyeSequence1 = [243, 190, 56, 28, 27, 29, 30, 247, 130, 25, 110, 24, 23, 22, 26, 112];
const leftEyeSequence2 = [133, 173, 157, 158, 159, 160, 161, 246, 33, 7, 163, 144, 145, 153, 154, 155];
const rightEyeSequence1 = [359, 467, 260, 259, 257, 258, 286, 414, 463, 341, 256, 252, 253, 254, 339, 255];
const rightEyeSequence2 = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249];

const RIGHT_EARLOBE_INDEX = 147;
const LEFT_EARLOBE_INDEX = 376;

const faceColors = [
  [255, 0, 0],
  [0, 255, 0],
  [0, 0, 255],
  [255, 255, 0],
  [255, 0, 255],
  [0, 255, 255],
  [255, 165, 0],
  [128, 0, 128],
];

function preload() {
  for (let i = 1; i <= 6; i++) {
    maskImgs.push(loadImage(`0${i}.png`));
  }
}

function setup() {
  createCanvas(windowWidth, windowHeight);
  background("#e7c6ff");

  video = createCapture(
    {
      video: {
        facingMode: "user",
      },
      audio: false,
    },
    () => {
      console.log("Camera stream started.");
      startModels();
    }
  );

  video.size(windowWidth * 0.8, windowHeight * 0.8);
  video.hide();
}

function startModels() {
  const faceOptions = {
    maxFaces: MAX_FACES,
    detectionConfidence: 0.5,
  };

  facemesh = ml5.facemesh(video, faceOptions, () => {
    console.log("FaceMesh model ready.");
    faceModelLoaded = true;
  });
  facemesh.on("predict", (results) => {
    predictions = results;
  });

  handpose = ml5.handpose(video, () => {
    console.log("Handpose model ready.");
    handModelLoaded = true;
  });
  handpose.on("predict", (results) => {
    handPredictions = results;
  });
}

function draw() {
  background("#e7c6ff");

  if (!video || video.width === 0 || video.height === 0) {
    drawStatus("Loading camera...");
    return;
  }

  const videoX = (width - video.width) / 2;
  const videoY = (height - video.height) / 2;

  drawMirroredVideo(videoX, videoY);

  if (!faceModelLoaded || !handModelLoaded) {
    drawStatus("Loading AI models...");
    return;
  }

  for (let i = 0; i < predictions.length; i += 1) {
    const facePrediction = predictions[i];
    if (!facePrediction) {
      continue;
    }
    drawKeypoints(facePrediction, i, videoX, videoY);
  }
}

function drawMirroredVideo(videoX, videoY) {
  push();
  translate(videoX + video.width, videoY);
  scale(-1, 1);
  image(video, 0, 0, video.width, video.height);
  pop();
}

function drawStatus(message) {
  push();
  fill(0);
  noStroke();
  textAlign(CENTER, CENTER);
  textSize(16);
  text(message, width / 2, height / 2);
  pop();
}

function drawKeypoints(facePrediction, faceIndex, videoCanvasX, videoCanvasY) {
  const keypoints = facePrediction.scaledMesh;
  const clr = faceColors[faceIndex % faceColors.length];

  const drawLandmarkLines = (sequence) => {
    stroke(clr[0], clr[1], clr[2]);
    strokeWeight(1.5);
    noFill();
    beginShape();

    for (let j = 0; j < sequence.length; j += 1) {
      const index = sequence[j];
      const point = keypoints[index];

      if (!point) {
        continue;
      }

      const [x, y] = scalePoint(point);
      vertex(videoCanvasX + video.width - x, videoCanvasY + y);
    }

    endShape(CLOSE);
  };

  // 如果偵測到手（揮手狀態）
  if (handPredictions.length > 0) {
    // 1. 更新臉譜切換邏輯 (每 10 幀切換一次圖片)
    maskCycleCounter++;
    if (maskCycleCounter > 10) {
      currentMaskIndex = (currentMaskIndex + 1) % 6;
      maskCycleCounter = 0;
    }

    // 2. 計算臉部邊界以決定臉譜位置與大小
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;

    for (let i = 0; i < keypoints.length; i++) {
      const [px, py] = scalePoint(keypoints[i]);
      if (px < minX) minX = px;
      if (px > maxX) maxX = px;
      if (py < minY) minY = py;
      if (py > maxY) maxY = py;
    }

    // 3. 繪製臉譜圖片
    const maskImg = maskImgs[currentMaskIndex];
    if (maskImg) {
      const padding = 20; // 稍微擴大臉譜覆蓋範圍
      const faceW = (maxX - minX) + padding * 2;
      const faceH = (maxY - minY) + padding * 2;
      
      push();
      // 因為鏡像關係，X 座標計算需要調整
      const drawX = videoCanvasX + video.width - maxX - padding;
      const drawY = videoCanvasY + minY - padding;
      
      image(maskImg, drawX, drawY, faceW, faceH);
      pop();
    }
  }
}

function scalePoint(point) {
  const intrinsicVideoWidth = video.elt.videoWidth || video.width;
  const intrinsicVideoHeight = video.elt.videoHeight || video.height;
  const scaleX = video.width / intrinsicVideoWidth;
  const scaleY = video.height / intrinsicVideoHeight;

  return [point[0] * scaleX, point[1] * scaleY];
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);

  if (!video) {
    return;
  }

  video.size(windowWidth * 0.8, windowHeight * 0.8);
}
