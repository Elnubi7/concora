const sendBtn = document.getElementById("send_btn");
const rebuildBtn = document.getElementById("rebuild_btn");
const answerBox = document.getElementById("answer_box");
const recommendationsBox = document.getElementById("recommendations_box");
const retrievalBox = document.getElementById("retrieval_box");

sendBtn.addEventListener("click", async () => {
  const payload = {
    mbti_type: document.getElementById("mbti").value,
    user_message: document.getElementById("user_message").value,
    user_gender: "female",
    session_id: document.getElementById("session_id").value || null,
    include_recommendations: true,
    top_k: 6
  };

  answerBox.textContent = "جاري الإرسال...";
  recommendationsBox.innerHTML = "";
  retrievalBox.textContent = "";

  try {
    const response = await fetch("/api/v1/chat", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    const data = await response.json();
    answerBox.textContent = data.answer || "No answer";

    if (Array.isArray(data.recommendations) && data.recommendations.length) {
      recommendationsBox.innerHTML = data.recommendations.map(item => {
        const urlPart = item.url ? `<div><a href="${item.url}" target="_blank" rel="noreferrer">فتح الرابط</a></div>` : "";
        return `
          <div class="resource">
            <div class="resource-title">${item.title}</div>
            <div class="resource-meta">${item.category}${item.issue_title ? " • " + item.issue_title : ""}</div>
            ${urlPart}
          </div>
        `;
      }).join("");
    } else {
      recommendationsBox.textContent = "لا توجد توصيات مناسبة من الداتا الحالية.";
    }

    retrievalBox.textContent = JSON.stringify(data.retrieved_chunks || [], null, 2);
  } catch (error) {
    answerBox.textContent = "حصل خطأ أثناء الإرسال.";
    console.error(error);
  }
});

rebuildBtn.addEventListener("click", async () => {
  answerBox.textContent = "جاري إعادة بناء الـ index...";
  try {
    const response = await fetch("/api/v1/index/rebuild", { method: "POST" });
    const data = await response.json();
    answerBox.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    answerBox.textContent = "فشل إعادة بناء الـ index.";
    console.error(error);
  }
});