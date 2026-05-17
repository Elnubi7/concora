const sendBtn = document.getElementById("send_btn");
const rebuildBtn = document.getElementById("rebuild_btn");
const answerBox = document.getElementById("answer_box");
const recommendationsBox = document.getElementById("recommendations_box");
const retrievalBox = document.getElementById("retrieval_box");
const messageInput = document.getElementById("user_message");

if (messageInput) {
  messageInput.placeholder = navigator.language?.startsWith("ar") ? "اكتب رسالتك..." : "Type your message...";
}

function renderAnswer(data) {
  if (typeof data.answer === "string") {
    return data.answer;
  }
  const response = data.response || {};
  return [
    response.understanding,
    response.mbti_connection,
    response.grounded_answer,
    Array.isArray(response.practical_steps) && response.practical_steps.length
      ? response.practical_steps.map((step) => `• ${step}`).join("\n")
      : "",
    response.follow_up_question,
    response.choice_prompt,
    response.support_note
  ].filter(Boolean).join("\n\n") || "No answer";
}

sendBtn.addEventListener("click", async () => {
  if (sendBtn.disabled) {
    return;
  }

  const payload = {
    mbti_type: document.getElementById("mbti").value,
    user_message: messageInput.value,
    user_gender: "female",
    conversation_id: document.getElementById("session_id").value || null,
    include_recommendations: true,
    top_k: 6
  };

  if (!payload.user_message.trim()) {
    answerBox.textContent = "اكتب رسالتك الأول.";
    return;
  }

  sendBtn.disabled = true;
  answerBox.classList.add("typing");
  answerBox.textContent = "الروبوت يكتب...";
  recommendationsBox.innerHTML = "";
  retrievalBox.textContent = "";

  try {
    const response = await fetch("/api/v1/chat", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    if (!response.ok) {
      throw new Error(`Request failed: ${response.status}`);
    }
    const data = await response.json();
    answerBox.textContent = renderAnswer(data);

    const resources = [
      ...(data.recommended_videos || []),
      ...(data.recommended_books || []),
      ...(data.recommended_podcasts || [])
    ];
    if (resources.length) {
      recommendationsBox.innerHTML = resources.map(item => {
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
    answerBox.textContent = "حصل خطأ مؤقت في الرد. حاول مرة تانية.";
    console.error(error);
  } finally {
    answerBox.classList.remove("typing");
    sendBtn.disabled = false;
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
