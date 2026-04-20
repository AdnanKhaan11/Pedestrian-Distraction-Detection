/*
This file handles frontend interactions for upload, prediction, training,
and modal behavior. It keeps browser-side logic separate from backend logic.
The code is written simply so it is easy to follow and extend later.
It powers the responsive UI without needing a frontend framework.
*/

const imageInput = document.getElementById("imageInput");
const inputPreview = document.getElementById("inputPreview");
const outputPreview = document.getElementById("outputPreview");
const predictForm = document.getElementById("predictForm");
const predictStatus = document.getElementById("predictStatus");
const resultContent = document.getElementById("resultContent");
const trainStatus = document.getElementById("trainStatus");

const trainPostureBtn = document.getElementById("trainPostureBtn");
const trainPhoneBtn = document.getElementById("trainPhoneBtn");

const openLoginBtn = document.getElementById("openLoginBtn");
const closeLoginBtn = document.getElementById("closeLoginBtn");
const loginModal = document.getElementById("loginModal");

const openAboutBtn = document.getElementById("openAboutBtn");
const closeAboutBtn = document.getElementById("closeAboutBtn");
const aboutModal = document.getElementById("aboutModal");

function setStatusBox(element, message, mode = "neutral") {
  element.className = `status-box ${mode}`;
  element.textContent = message;
}

function showModal(modal) {
  modal.classList.remove("hidden");
}

function hideModal(modal) {
  modal.classList.add("hidden");
}

if (openLoginBtn)
  openLoginBtn.addEventListener("click", () => showModal(loginModal));
if (closeLoginBtn)
  closeLoginBtn.addEventListener("click", () => hideModal(loginModal));

if (openAboutBtn)
  openAboutBtn.addEventListener("click", () => showModal(aboutModal));
if (closeAboutBtn)
  closeAboutBtn.addEventListener("click", () => hideModal(aboutModal));

window.addEventListener("click", (event) => {
  if (event.target === loginModal) hideModal(loginModal);
  if (event.target === aboutModal) hideModal(aboutModal);
});

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (event) => {
    inputPreview.src = event.target.result;
    outputPreview.src = "";
    resultContent.innerHTML = "No prediction yet.";
    setStatusBox(
      predictStatus,
      "Image selected. Ready to run prediction.",
      "neutral",
    );
  };
  reader.readAsDataURL(file);
});

predictForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = imageInput.files[0];
  if (!file) {
    setStatusBox(predictStatus, "Please upload an image first.", "error");
    return;
  }

  setStatusBox(predictStatus, "Running prediction...", "neutral");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.detail || "Prediction failed.");
    }

    if (result.saved_result_path) {
      outputPreview.src = `${result.saved_result_path}?t=${Date.now()}`;
    }

    let html = `<p><strong>Persons detected:</strong> ${result.num_persons}</p>`;

    if (result.person_results && result.person_results.length > 0) {
      html += "<div>";
      result.person_results.forEach((person, index) => {
        html += `
                    <div style="margin-bottom:12px;padding:12px;border:1px solid #e5d7c5;border-radius:12px;background:#fff;">
                        <strong>Person ${index + 1}</strong><br>
                        Final Label: ${person.posture}<br>
                        Phone Detected: ${person.phone}<br>
                        State Code: ${person.state}<br>
                        Display Text: ${person.display_text}<br>
                        Score: ${person.score_text}
                    </div>
                `;
      });
      html += "</div>";
    }

    resultContent.innerHTML = html;
    setStatusBox(
      predictStatus,
      "Prediction completed successfully.",
      "success",
    );
  } catch (error) {
    setStatusBox(predictStatus, `Prediction failed: ${error.message}`, "error");
  }
});

trainPostureBtn.addEventListener("click", async () => {
  setStatusBox(trainStatus, "Starting posture training...", "neutral");

  try {
    const response = await fetch("/train/posture", { method: "POST" });
    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.detail || "Posture training failed.");
    }

    const msg = `Posture training ${result.status}. Return code: ${result.return_code}`;
    setStatusBox(
      trainStatus,
      msg,
      result.status === "completed" ? "success" : "error",
    );
  } catch (error) {
    setStatusBox(
      trainStatus,
      `Posture training failed: ${error.message}`,
      "error",
    );
  }
});

trainPhoneBtn.addEventListener("click", async () => {
  setStatusBox(trainStatus, "Starting phone detector training...", "neutral");

  try {
    const response = await fetch("/train/phone", { method: "POST" });
    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.detail || "Phone training failed.");
    }

    const msg = `Phone training ${result.status}. Return code: ${result.return_code}`;
    setStatusBox(
      trainStatus,
      msg,
      result.status === "completed" ? "success" : "error",
    );
  } catch (error) {
    setStatusBox(
      trainStatus,
      `Phone training failed: ${error.message}`,
      "error",
    );
  }
});
