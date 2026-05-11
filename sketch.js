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
    },
    (stream) => {
      console.log("Camera stream started.");
      video.size(640, 480);
      startModels();
    }
  );
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
    currentFingerCount = countFingers(handPredictions);
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
  if (!facePrediction || !facePrediction.scaledMesh) return;
  const keypoints = facePrediction.scaledMesh;

  // --- 繪製耳環邏輯 ---
  if (currentFingerCount >= 1 && currentFingerCount <= 5) {
    const earringImg = earringImgs[currentFingerCount - 1];
    const eSize = 50; // 耳環大小
    
    const drawEarring = (index) => {
      const p = keypoints && keypoints[index];
      if (p) {
        const [x, y] = scalePoint(p);
        // 考慮鏡像繪製在耳垂位置
        image(earringImg, videoCanvasX + video.width - x - eSize/2, videoCanvasY + y, eSize, eSize * 1.5);
      }
    };

    drawEarring(RIGHT_EARLOBE_INDEX);
    drawEarring(LEFT_EARLOBE_INDEX);
  }

  // --- 繪製臉譜邏輯 ---

  // 如果偵測到手（揮手狀態）
  if (handPredictions.length > 0) {
    maskCycleCounter++;
    if (maskCycleCounter > 5) { // 縮短切換幀數，讓變臉更靈敏
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

function scalePoint(point) {
  const intrinsicVideoWidth = video.elt.videoWidth || video.width;
  const intrinsicVideoHeight = video.elt.videoHeight || video.height;
  const scaleX = video.width / intrinsicVideoWidth;
  const scaleY = video.height / intrinsicVideoHeight;

  return [point[0] * scaleX, point[1] * scaleY];
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
}
