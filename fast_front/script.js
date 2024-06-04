const blob = window.URL || window.webkitURL;
const modalUpload = document.querySelector(".modal-upload");
const modalLoader = document.querySelector(".modal-loader");
const uploadArea = document.querySelector(".upload-area");
const fileInput = document.querySelector(".file-input");
const audioCutter = document.querySelector(".audio-cutter");
const audioInput = document.querySelector(".audio-input");
const startButton = document.querySelector(".start-button");
const cutButton = document.querySelector(".cut-button");
const resultAudio = document.querySelector(".result-audio");
const resultAudioDescription = document.querySelector(
  ".result-audio-description"
);
const uploadButton = document.querySelector(".upload-button");
const recognitionDescription = document.querySelector(
  ".recognition-description"
);
const warnDescription = document.querySelector(".warn-description");

let file = null;
let start = null;
let end = null;
let audio = null;

uploadArea.addEventListener("click", () => {
  fileInput.value = "";
  fileInput.click();
});
uploadArea.addEventListener("dragover", (event) => event.preventDefault());

fileInput.addEventListener("change", (event) => {
  if (event.target.files && event.target.files.length) {
    if (isWav(event.target.files[0])) {
      setAudioInputFile(event.target.files[0]);
      file = event.target.files[0];
    }
  }
});
uploadArea.addEventListener("drop", (event) => {
  event.preventDefault();

  if (event.dataTransfer.items) {
    for (const item of event.dataTransfer.items) {
      if (item.kind === "file") {
        if (isWav(item.getAsFile())) {
          setAudioInputFile(item.getAsFile());
          file = item.getAsFile();
          break;
        }
      }
    }
  } else {
    for (const item of event.dataTransfer.files) {
      if (isWav(item)) {
        setAudioInputFile(item);
        file = item;
        break;
      }
    }
  }
});

startButton.addEventListener("click", () => {
  resultAudio.classList.add("result-audio_hidden");
  uploadButton.classList.add("hidden");
  warnDescription.classList.add("hidden");
  start = audioInput.currentTime;
  end = start + 30;

  if (end > audioInput.duration) {
    end -= end - audioInput.duration;
  }

  if (end - start < 30) {
    warnDescription.classList.remove("hidden");
  }
});
cutButton.addEventListener("click", () => {
  if (end - start < 30) {
    return;
  }

  uploadButton.classList.add("hidden");
  resultAudioDescription.textContent = `Start (in seconds): ${(start).toFixed(
    2
  )}, End (in seconds): ${(end).toFixed(2)}`;
  resultAudio.classList.remove("result-audio_hidden");
  uploadButton.classList.remove("hidden");
});

function isWav(file) {
  let extension = null;

  if (file.type) {
    extension = file.type;
  } else {
    extension = file.name.split(".").at(-1) ?? null;
  }

  return extension && extension.includes("wav");
}

function setAudioInputFile(file) {
  fileUrl = blob.createObjectURL(file);
  audioInput.src = fileUrl;
  audioCutter.classList.remove("audio-cutter_hidden");
}

uploadButton.addEventListener("click", () => {
  processFile();
});

async function processFile() {
  modalUpload.classList.add("hidden");
  modalLoader.classList.remove("hidden");

  const formData = new FormData();
  formData.append("audio", file);
  formData.append("start", start);
  formData.append("end", end);

  let data = "Result test";

  try {
    const response = await fetch("http://localhost:8001", {
      method: "POST",
      body: formData,
    });
    data = await response.text();
  } catch (error) {
    console.error(error);
  }

  modalUpload.classList.remove("hidden");
  modalLoader.classList.add("hidden");

  recognitionDescription.classList.remove("hidden");
  recognitionDescription.textContent = data;
}
