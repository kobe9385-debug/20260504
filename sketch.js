let video;
let facemesh;
let handpose;

let predictions = [];
let handPredictions = [];
let maskImgs = [];
let earringImgs = [];

let faceModelLoaded = false;
let handModelLoaded = false;

let currentMaskIndex = 0;
let currentFingerCount = 0;

let handWasOverFace = false; // 用於偵測揮手切換的狀態
let vScale = 1; // 全域縮放比例

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
  for (let i = 1; i <= 5; i++) {
    earringImgs.push(loadImage(`${i}.png`));
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
    }
  );
  video.size(640, 480);
  video.elt.setAttribute('playsinline', ''); // 行動裝置必備
  video.elt.play(); 
  video.hide();
  startModels();
}

function startModels() {
  const faceOptions = {
    maxFaces: MAX_FACES,
    detectionConfidence: 0.5,
  };

  facemesh = ml5.facemesh(video.elt, faceOptions, () => {
    console.log("FaceMesh model ready.");
    faceModelLoaded = true;
  });
  facemesh.on("predict", (results) => {
    predictions = results;
  });

  handpose = ml5.handpose(video.elt, () => {
    console.log("Handpose model ready.");
    handModelLoaded = true;
  });
  handpose.on("predict", (results) => {
    handPredictions = results;
    currentFingerCount = countFingers(handPredictions);
  });
}

function draw() {
  background("#e7c6ff");

  if (!video || video.width === 0 || video.height === 0) {
    drawStatus("Loading camera...");
    return;
  }

  // 計算縮放比例以適應視窗的 80% 並維持比例不壓扁
  let scaleW = (windowWidth * 0.8) / video.width;
  let scaleH = (windowHeight * 0.8) / video.height;
  vScale = min(scaleW, scaleH);
  const vW = video.width * vScale;
  const vH = video.height * vScale;
  const vX = (width - vW) / 2;
  const vY = (height - vH) / 2;

  drawMirroredVideo(vX, vY, vW, vH);

  if (!faceModelLoaded || !handModelLoaded) {
    let msg = "Loading AI models...";
    if (faceModelLoaded && !handModelLoaded) msg = "Loading Hand Model...";
    if (!faceModelLoaded && handModelLoaded) msg = "Loading Face Model...";
    drawStatus(msg);
    return;
  }

  for (let i = 0; i < predictions.length; i += 1) {
    const facePrediction = predictions[i];
    if (!facePrediction) {
      continue;
    }
    drawKeypoints(facePrediction, i, vX, vY, vW, vH);
  }
}

function drawMirroredVideo(x, y, w, h) {
  push();
  translate(x + w, y);
  scale(-1, 1);
  image(video, 0, 0, w, h);
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

function drawKeypoints(facePrediction, faceIndex, vX, vY, vW, vH) {
  if (!facePrediction || !facePrediction.scaledMesh) return;
  const keypoints = facePrediction.scaledMesh;
  const faceBox = facePrediction.boundingBox; // [x, y, w, h]

  // --- 繪製耳環邏輯 ---
  if (currentFingerCount >= 1 && currentFingerCount <= 5) {
    const earringImg = earringImgs[currentFingerCount - 1];
    const eSize = 50 * vScale; // 隨畫面縮放
    
    const drawEarring = (index) => {
      const p = keypoints && keypoints[index];
      if (p) {
        const cp = getCanvasPoint(p, vX, vY, vW);
        image(earringImg, cp.x - eSize/2, cp.y, eSize, eSize * 1.5);
      }
    };

    drawEarring(RIGHT_EARLOBE_INDEX);
    drawEarring(LEFT_EARLOBE_INDEX);
  }

  // --- 繪製臉譜邏輯 ---
  
  // 偵測手是否經過臉部
  let handIsOverFace = false;
  if (handPredictions.length > 0) {
    const handWrist = handPredictions[0].landmarks[0]; // 拿手腕點當中心
    const hX = handWrist[0];
    const hY = handWrist[1];
    
    // 檢查手腕座標是否在臉部 Bounding Box 內 (使用原始影像座標比對)
    if (hX > faceBox.topLeft[0][0] && hX < faceBox.bottomRight[0][0] &&
        hY > faceBox.topLeft[0][1] && hY < faceBox.bottomRight[0][1]) {
      handIsOverFace = true;
    }
  }

  // 只有在手「進入」臉部區域的那一幀切換
  if (handIsOverFace && !handWasOverFace) {
    currentMaskIndex = (currentMaskIndex + 1) % maskImgs.length;
  }
  handWasOverFace = handIsOverFace;

  // --- 繪製臉譜圖片 (始終顯示) ---
  const pTop = getCanvasPoint(keypoints[10], vX, vY, vW);
  const pBottom = getCanvasPoint(keypoints[152], vX, vY, vW);
  const pLeft = getCanvasPoint(keypoints[234], vX, vY, vW);
  const pRight = getCanvasPoint(keypoints[454], vX, vY, vW);

  const minX = Math.min(pLeft.x, pRight.x);
  const maxX = Math.max(pLeft.x, pRight.x);
  const minY = pTop.y;
  const maxY = pBottom.y;

  const maskImg = maskImgs[currentMaskIndex];
  if (maskImg) {
    const padding = 20 * vScale;
    const faceW = (maxX - minX) + padding * 2;
    const faceH = (maxY - minY) + padding * 2;
    
    push();
    const drawX = minX - padding;
    const drawY = minY - padding;
    image(maskImg, drawX, drawY, faceW, faceH);
    pop();
  }
}

function countFingers(hands) {
  if (!hands || hands.length === 0 || !hands[0].landmarks) return 0;
  
  let count = 0;
  const lm = hands[0].landmarks;
  if (lm.length < 21) return 0; // 確保點位資料完整

  // 手指尖端與關節索引
  const tips = [8, 12, 16, 20];
  const joints = [6, 10, 14, 18];

  for (let i = 0; i < 4; i++) {
    if (lm[tips[i]][1] < lm[joints[i]][1]) {
      count++;
    }
  }

  // 大拇指辨識 (根據 X 軸水平距離判斷是否張開)
  const thumbTip = lm[4];
  const thumbBase = lm[2];
  if (Math.abs(thumbTip[0] - lm[0][0]) > Math.abs(thumbBase[0] - lm[0][0])) {
    count++;
  }

  return count;
}

function getCanvasPoint(point, vX, vY, vW) {
  const intrinsicVideoWidth = video.elt.videoWidth || video.width;
  const intrinsicVideoHeight = video.elt.videoHeight || video.height;
  
  // 1. 正規化並鏡像 X 軸
  let normX = point[0] / intrinsicVideoWidth;
  let cpX = vX + (1 - normX) * vW;
  let cpY = vY + (point[1] / intrinsicVideoHeight) * (video.height * vScale); // 使用 vScale 統一 Y 座標

  return { x: cpX, y: cpY };
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
}
