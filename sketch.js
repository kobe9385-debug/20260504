let video;
let facemesh;
let handpose;

let predictions = [];
let handPredictions = [];

let faceModelLoaded = false;
let handModelLoaded = false;
let pixelationBuffer;

const MAX_FACES = 2;
const DISPERSE_DURATION = 1000;
const REAPPEAR_DURATION = 2000;
const DISPERSE_THRESHOLD = 90;

const landmarkSequence1 = [409, 270, 269, 267, 0, 37, 39, 40, 185, 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291];
const landmarkSequence2 = [76, 77, 90, 180, 85, 16, 315, 404, 320, 307, 306, 408, 304, 303, 302, 11, 72, 73, 74, 184];

const leftEyeSequence1 = [243, 190, 56, 28, 27, 29, 30, 247, 130, 25, 110, 24, 23, 22, 26, 112];
const leftEyeSequence2 = [133, 173, 157, 158, 159, 160, 161, 246, 33, 7, 163, 144, 145, 153, 154, 155];
const rightEyeSequence1 = [359, 467, 260, 259, 257, 258, 286, 414, 463, 341, 256, 252, 253, 254, 339, 255];
const rightEyeSequence2 = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249];

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

let facesData = [];

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

  pixelationBuffer = createGraphics(video.width, video.height);
  pixelationBuffer.noSmooth();
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

    if (predictions.length === 0) {
      handPredictions = [];
    }

    syncFaceData(predictions.length);
  });

  handpose = ml5.handpose(video, () => {
    console.log("Handpose model ready.");
    handModelLoaded = true;
  });

  handpose.on("predict", (results) => {
    handPredictions = predictions.length > 0 ? results : [];
  });
}

function syncFaceData(count) {
  while (facesData.length < count) {
    const graphics = createGraphics(video.width, video.height);
    graphics.noSmooth();

    facesData.push({
      graphics,
      isDispersed: false,
      isReappearing: false,
      effectStartTime: 0,
      originalImage: null,
    });
  }

  facesData = facesData.slice(0, count);
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
    const faceData = facesData[i];

    if (!facePrediction || !faceData) {
      continue;
    }

    updateFaceEffect(faceData, facePrediction);

    if (faceData.isDispersed || faceData.isReappearing) {
      image(faceData.graphics, videoX, videoY);
    } else {
      drawKeypoints(facePrediction, i, videoX, videoY);
    }
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

function updateFaceEffect(faceData, facePrediction) {
  const handDetected = checkHandInteraction(facePrediction, handPredictions);
  const elapsed = millis() - faceData.effectStartTime;

  if (handDetected && !faceData.isDispersed && !faceData.isReappearing) {
    faceData.isDispersed = true;
    faceData.isReappearing = false;
    faceData.effectStartTime = millis();
    captureFaceImage(faceData.graphics, facePrediction);
    faceData.originalImage = faceData.graphics.get();
    return;
  }

  if (faceData.isDispersed && handDetected) {
    const progress = constrain(elapsed / DISPERSE_DURATION, 0, 1);
    applyDisperseEffect(faceData.graphics, faceData.originalImage, progress);
    return;
  }

  if (faceData.isDispersed && !handDetected) {
    if (elapsed < DISPERSE_DURATION) {
      const progress = constrain(elapsed / DISPERSE_DURATION, 0, 1);
      applyDisperseEffect(faceData.graphics, faceData.originalImage, progress);
      return;
    }

    faceData.isDispersed = false;
    faceData.isReappearing = true;
    faceData.effectStartTime = millis();
  }

  if (faceData.isReappearing) {
    const progress = constrain((millis() - faceData.effectStartTime) / REAPPEAR_DURATION, 0, 1);

    if (progress >= 1) {
      resetFaceEffect(faceData);
      return;
    }

    applyReappearEffect(faceData.graphics, faceData.originalImage, progress);
  }
}

function resetFaceEffect(faceData) {
  faceData.isDispersed = false;
  faceData.isReappearing = false;
  faceData.effectStartTime = 0;
  faceData.originalImage = null;
  faceData.graphics.clear();
}

function captureFaceImage(graphics, facePrediction) {
  const bounds = getFaceBounds(facePrediction);

  if (!bounds) {
    graphics.clear();
    return;
  }

  graphics.clear();
  graphics.push();
  graphics.translate(bounds.mirrorX + bounds.width, bounds.y);
  graphics.scale(-1, 1);
  graphics.image(
    video,
    0,
    0,
    bounds.width,
    bounds.height,
    bounds.x,
    bounds.y,
    bounds.width,
    bounds.height
  );
  graphics.pop();
}

function applyDisperseEffect(graphics, originalImage, progress) {
  if (!originalImage) {
    return;
  }

  drawPixelatedImage(graphics, originalImage, {
    pixelScale: map(progress, 0, 1, 1, 0.06),
    expand: map(progress, 0, 1, 1, 1.28),
    alpha: map(progress, 0, 1, 255, 0),
  });
}

function applyReappearEffect(graphics, originalImage, progress) {
  if (!originalImage) {
    return;
  }

  drawPixelatedImage(graphics, originalImage, {
    pixelScale: map(progress, 0, 1, 0.06, 1),
    expand: map(progress, 0, 1, 1.28, 1),
    alpha: map(progress, 0, 1, 0, 255),
  });
}

function drawPixelatedImage(graphics, originalImage, settings) {
  const smallWidth = max(1, floor(originalImage.width * settings.pixelScale));
  const smallHeight = max(1, floor(originalImage.height * settings.pixelScale));

  pixelationBuffer.clear();
  pixelationBuffer.noSmooth();
  pixelationBuffer.image(originalImage, 0, 0, smallWidth, smallHeight);

  graphics.clear();
  graphics.push();
  graphics.noSmooth();
  graphics.tint(255, settings.alpha);
  graphics.translate(graphics.width / 2, graphics.height / 2);
  graphics.scale(settings.expand);
  graphics.image(pixelationBuffer, -graphics.width / 2, -graphics.height / 2, graphics.width, graphics.height);
  graphics.pop();
}

function checkHandInteraction(facePrediction, currentHandPredictions) {
  if (!currentHandPredictions || currentHandPredictions.length === 0) {
    return false;
  }

  const faceBounds = getFaceBounds(facePrediction);

  if (!faceBounds) {
    return false;
  }

  const faceCenterX = faceBounds.x + faceBounds.width / 2;
  const faceCenterY = faceBounds.y + faceBounds.height / 2;
  const threshold = max(DISPERSE_THRESHOLD, min(video.width, video.height) * 0.12);

  for (let i = 0; i < currentHandPredictions.length; i += 1) {
    const hand = currentHandPredictions[i];

    if (!hand.landmarks || hand.landmarks.length === 0) {
      continue;
    }

    for (let j = 0; j < hand.landmarks.length; j += 1) {
      const [handX, handY] = scalePoint(hand.landmarks[j]);

      if (dist(faceCenterX, faceCenterY, handX, handY) < threshold) {
        return true;
      }
    }
  }

  return false;
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
}

function getFaceBounds(facePrediction) {
  if (!facePrediction || !facePrediction.scaledMesh || facePrediction.scaledMesh.length === 0) {
    return null;
  }

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (let i = 0; i < facePrediction.scaledMesh.length; i += 1) {
    const [x, y] = scalePoint(facePrediction.scaledMesh[i]);
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }

  const padding = 24;
  minX = constrain(minX - padding, 0, video.width);
  minY = constrain(minY - padding, 0, video.height);
  maxX = constrain(maxX + padding, 0, video.width);
  maxY = constrain(maxY + padding, 0, video.height);

  const faceWidth = maxX - minX;
  const faceHeight = maxY - minY;

  if (faceWidth <= 0 || faceHeight <= 0) {
    return null;
  }

  return {
    x: minX,
    y: minY,
    width: faceWidth,
    height: faceHeight,
    mirrorX: video.width - maxX,
  };
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

  if (pixelationBuffer) {
    pixelationBuffer.resizeCanvas(video.width, video.height);
    pixelationBuffer.noSmooth();
  }

  for (let i = 0; i < facesData.length; i += 1) {
    facesData[i].graphics.resizeCanvas(video.width, video.height);
    facesData[i].graphics.noSmooth();
    resetFaceEffect(facesData[i]);
  }
}
