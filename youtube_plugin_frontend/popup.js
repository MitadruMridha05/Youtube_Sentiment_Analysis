// popup.js

document.addEventListener("DOMContentLoaded", () => {
  const outputDiv = document.getElementById("output");

  // Keep your key out of source control; restrict it to the YouTube Data API in Google Cloud.
  const API_KEY = "AIzaSyDw7k-bODEq8-PeGwa_KR9nmzgFnqnXsGw";
  const API_URL = "http://localhost:5000"; // no trailing slash
  const MAX_COMMENTS = 500;

  const SENTIMENT_LABELS = {
    "1":  { text: "Positive", cls: "pos" },
    "0":  { text: "Neutral",  cls: "neu" },
    "-1": { text: "Negative", cls: "neg" }
  };

  const escapeHtml = (str) =>
    String(str).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    }[c]));

  // Appends without re-parsing existing DOM (so loaded images are kept)
  const append = (html) => outputDiv.insertAdjacentHTML("beforeend", html);
  const showError = (msg) => append(`<p class="error">${msg}</p>`);

  chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
    const url = tabs[0]?.url || "";
    const match = url.match(/^https:\/\/(?:www\.)?youtube\.com\/watch\?v=([\w-]{11})/);

    if (!match) {
      outputDiv.innerHTML = "<p>Open a YouTube video to see its comment insights.</p>";
      return;
    }

    const videoId = match[1];
    outputDiv.innerHTML = `
      <div class="section">
        <div class="section-title">YouTube video ID</div>
        <p class="video-id">${videoId}</p>
        <p>Fetching comments...</p>
      </div>`;

    const comments = await fetchComments(videoId);
    if (comments.length === 0) {
      append("<p>No comments found for this video.</p>");
      return;
    }

    append(`<p>Fetched ${comments.length} comments. Running sentiment analysis...</p>`);
    const predictions = await getSentimentPredictions(comments);
    if (!predictions) return;

    // Sentiment counts + trend data
    const sentimentCounts = { "1": 0, "0": 0, "-1": 0 };
    const sentimentData = [];
    let totalSentimentScore = 0;
    predictions.forEach((item) => {
      sentimentCounts[item.sentiment]++;
      totalSentimentScore += parseInt(item.sentiment);
      sentimentData.push({ timestamp: item.timestamp, sentiment: parseInt(item.sentiment) });
    });

    // Metrics
    const totalComments = comments.length;
    const uniqueCommenters = new Set(comments.map((c) => c.authorId)).size;
    const totalWords = comments.reduce(
      (sum, c) => sum + c.text.split(/\s+/).filter(Boolean).length, 0);
    const avgWordLength = (totalWords / totalComments).toFixed(2);
    const avgSentiment = totalSentimentScore / totalComments;
    const normalizedScore = (((avgSentiment + 1) / 2) * 10).toFixed(2); // 0–10

    append(`
      <div class="section">
        <div class="section-title">Comment analysis summary</div>
        <div class="metrics-container">
          <div class="metric">
            <div class="metric-title">Total comments</div>
            <div class="metric-value">${totalComments}</div>
          </div>
          <div class="metric">
            <div class="metric-title">Unique commenters</div>
            <div class="metric-value">${uniqueCommenters}</div>
          </div>
          <div class="metric">
            <div class="metric-title">Avg comment length</div>
            <div class="metric-value">${avgWordLength} <small>words</small></div>
          </div>
          <div class="metric">
            <div class="metric-title">Avg sentiment score</div>
            <div class="metric-value">${normalizedScore}<small>/10</small></div>
          </div>
        </div>
      </div>

      <div class="section">
        <div class="section-title">Sentiment analysis results</div>
        <p>Share of positive, neutral and negative comments.</p>
        <div id="chart-container" class="chart-container"></div>
      </div>

      <div class="section">
        <div class="section-title">Sentiment trend over time</div>
        <div id="trend-graph-container" class="chart-container"></div>
      </div>

      <div class="section">
        <div class="section-title">Comment word cloud</div>
        <div id="wordcloud-container" class="chart-container"></div>
      </div>

      <div class="section">
        <div class="section-title">Top 25 comments with sentiment</div>
        <ul class="comment-list">
          ${predictions.slice(0, 25).map((item) => {
            const s = SENTIMENT_LABELS[item.sentiment] || { text: item.sentiment, cls: "" };
            return `
              <li class="comment-item ${s.cls}">
                <span>${escapeHtml(item.comment)}</span>
                <span class="comment-sentiment">${s.text}</span>
              </li>`;
          }).join("")}
        </ul>
      </div>`);

    // Load the three images in parallel
    await Promise.all([
      fetchAndShowImage("generate_chart", { sentiment_counts: sentimentCounts }, "chart-container"),
      fetchAndShowImage("generate_trend_graph", { sentiment_data: sentimentData }, "trend-graph-container"),
      fetchAndShowImage("generate_wordcloud", { comments: comments.map((c) => c.text) }, "wordcloud-container")
    ]);
  });

  async function fetchComments(videoId) {
    const comments = [];
    let pageToken = "";
    try {
      while (comments.length < MAX_COMMENTS) {
        const params = new URLSearchParams({
          part: "snippet", videoId, maxResults: "100", key: API_KEY
        });
        if (pageToken) params.set("pageToken", pageToken);

        const response = await fetch(`https://www.googleapis.com/youtube/v3/commentThreads?${params}`);
        const data = await response.json();
        if (data.error) throw new Error(data.error.message);

        (data.items || []).forEach((item) => {
          const s = item.snippet.topLevelComment.snippet;
          comments.push({
            text: s.textOriginal,
            timestamp: s.publishedAt,
            authorId: s.authorChannelId?.value || "Unknown"
          });
        });

        pageToken = data.nextPageToken;
        if (!pageToken) break;
      }
    } catch (error) {
      console.error("Error fetching comments:", error);
      showError("Couldn't fetch comments. Check your API key and that comments are enabled.");
    }
    return comments;
  }

  async function getSentimentPredictions(comments) {
    const endpoint = `${API_URL}/predict_with_timestamps`;
    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comments })
      });
      const raw = await response.text();
      let result;
      try { result = JSON.parse(raw); } catch { result = null; }

      if (!response.ok || !result) {
        throw new Error(`Server replied ${response.status}: ${(result && result.error) || raw.slice(0, 120)}`);
      }
      return result;
    } catch (error) {
      console.error("Error fetching predictions:", error);
      showError(`Couldn't get sentiment predictions from ${escapeHtml(endpoint)}.<br>${escapeHtml(error.message)}`);
      return null;
    }
  }

  // One helper for chart, trend graph and word cloud
  async function fetchAndShowImage(endpoint, body, containerId) {
    const container = document.getElementById(containerId);
    try {
      const response = await fetch(`${API_URL}/${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });
      if (!response.ok) throw new Error(`Failed to fetch ${endpoint} image`);
      const blob = await response.blob();
      const img = document.createElement("img");
      img.src = URL.createObjectURL(blob);
      img.alt = endpoint.replace("generate_", "").replace("_", " ");
      container.appendChild(img);
    } catch (error) {
      console.error(`Error loading ${endpoint}:`, error);
      container.innerHTML = '<p class="error">Couldn\'t load this image.</p>';
    }
  }
});
