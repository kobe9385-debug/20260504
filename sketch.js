let video;
let facemesh;
let handpose;

let predictions = [];
let handPredictions = [];
let fingerImgs = [];

let faceModelLoaded = false;
let handModelLoaded = false;
let currentFingerCount = 0;

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
  for (let i = 1; i <= 5; i++) {
    fingerImgs.push(loadImage(`${i}.png`));
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

  drawLandmarkLines(landmarkSequence1);
  drawLandmarkLines(landmarkSequence2);
  drawLandmarkLines(leftEyeSequence1);
  drawLandmarkLines(leftEyeSequence2);
  drawLandmarkLines(rightEyeSequence1);
  drawLandmarkLines(rightEyeSequence2);

  // 檢查手指數量並顯示對應耳環圖片
  if (currentFingerCount >= 1 && currentFingerCount <= 5) {
    const img = fingerImgs[currentFingerCount - 1];
    const imgW = 40; // 圖片寬度
    const imgH = 80; // 圖片高度 (耳環通常較長)

    // 繪製右耳環 (考慮鏡像)
    const rightEarlobe = keypoints[RIGHT_EAR_INDEX]; 
    const rE = keypoints[RIGHT_EARLOBE_INDEX];
    if (rE) {
      const [x, y] = scalePoint(rE);
      image(img, videoCanvasX + video.width - x - imgW / 2, videoCanvasY + y, imgW, imgH);
    }

    // 繪製左耳環 (考慮鏡像)
    const lE = keypoints[LEFT_EARLOBE_INDEX];
    if (lE) {
      const [x, y] = scalePoint(lE);
      image(img, videoCanvasX + video.width - x - imgW / 2, videoCanvasY + y, imgW, imgH);
    }
  } else {
    // 若手指數量不在 1-5 之間，維持黃色圓圈標示
    fill(255, 255, 0);
    noStroke();
  const rightEarlobe = keypoints[RIGHT_EARLOBE_INDEX];
  if (rightEarlobe) {
    const [x, y] = scalePoint(rightEarlobe);
    ellipse(videoCanvasX + video.width - x, videoCanvasY + y, 15, 15);
  }
  const leftEarlobe = keypoints[LEFT_EARLOBE_INDEX];
  if (leftEarlobe) {
    const [x, y] = scalePoint(leftEarlobe);
    ellipse(videoCanvasX + video.width - x, videoCanvasY + y, 15, 15);
  }
  }
}

function countFingers(hands) {
  if (hands.length === 0) return 0;

  let count = 0;
  const landmarks = hands[0].landmarks;

  // 手指尖端索引：食指(8), 中指(12), 無名指(16), 小指(20)
  const fingerTips = [8, 12, 16, 20];
  for (let tip of fingerTips) {
    // 如果指尖的 Y 座標低於第二關節，代表手指伸直
    if (landmarks[tip][1] < landmarks[tip - 2][1]) {
      count++;
    }
  }

  // 大拇指(4) 判斷 X 軸與掌心(0) 的相對距離
  if (Math.abs(landmarks[4][0] - landmarks[0][0]) > Math.abs(landmarks[3][0] - landmarks[0][0])) {
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

  if (!video) {
    return;
  }

  video.size(windowWidth * 0.8, windowHeight * 0.8);
}
