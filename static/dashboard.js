
const $ = (id) => document.getElementById(id);

const colors = {
    positive: "#34d399",
    negative: "#fb7185",
    neutral: "#60a5fa"
};

const chart = new Chart($("sentiment-chart"), {
    type: "doughnut",
    data: {
        labels: ["Positive", "Negative", "Neutral"],
        datasets: [{
            data: [0, 0, 0],
            backgroundColor: [
                colors.positive, colors.negative, colors.neutral
            ],
            borderColor: "#121a2d",
            borderWidth: 5,
            hoverOffset: 7
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: "72%",
        plugins: {
            legend: { display: false },
            tooltip: {
                callbacks: {
                    label: (context) => ` ${context.label}: ${context.raw}`
                }
            }
        }
    }
});

function showError(message) {
    $("error").textContent = message;
    $("error").hidden = false;
}

function hideError() {
    $("error").hidden = true;
}

function sentimentBadge(sentiment) {
    const safeSentiment = ["positive", "negative", "neutral"].includes(sentiment)
        ? sentiment : "neutral";

    return `<span class="badge ${safeSentiment}">${safeSentiment}</span>`;
}

async function refreshDashboard() {
    const [statsResponse, historyResponse] = await Promise.all([
        fetch("/api/stats"),
        fetch("/api/history")
    ]);

    if (!statsResponse.ok || !historyResponse.ok) {
        throw new Error("Unable to load dashboard data.");
    }

    const stats = await statsResponse.json();
    const data = await historyResponse.json();

    $("total").textContent = stats.total;
    $("positive").textContent = stats.positive;
    $("negative").textContent = stats.negative;
    $("neutral").textContent = stats.neutral;

    $("positive-pct").textContent = `${stats.positive_pct}% of reviews`;
    $("negative-pct").textContent = `${stats.negative_pct}% of reviews`;
    $("neutral-pct").textContent = `${stats.neutral_pct}% of reviews`;

    chart.data.datasets[0].data = [
        stats.positive, stats.negative, stats.neutral
    ];
    chart.update();

    const tbody = $("history-body");
    tbody.replaceChildren();

    if (!data.results.length) {
        const row = tbody.insertRow();
        const cell = row.insertCell();
        cell.colSpan = 3;
        cell.className = "empty";
        cell.textContent = "No reviews analyzed yet.";
    } else {
        data.results.slice(0, 10).forEach(item => {
            const row = tbody.insertRow();

            const textCell = row.insertCell();
            textCell.textContent = item.text.length > 110
                ? item.text.slice(0, 110) + "…" : item.text;

            const sentimentCell = row.insertCell();
            sentimentCell.innerHTML = sentimentBadge(item.sentiment);

            const confidenceCell = row.insertCell();
            confidenceCell.textContent = `${item.confidence.toFixed(2)}%`;
        });
    }

    if (!stats.total) {
        $("insight").textContent =
            "Analyze a review to start generating insights.";
    } else {
        const leaders = [
            ["Positive", stats.positive],
            ["Negative", stats.negative],
            ["Neutral", stats.neutral]
        ].sort((a, b) => b[1] - a[1]);

        $("insight").textContent =
            `${leaders[0][0]} sentiment leads with ` +
            `${Math.round(leaders[0][1] / stats.total * 100)}% ` +
            "of analyzed reviews in this session.";
    }
}

$("review").addEventListener("input", () => {
    $("char-count").textContent = `${$("review").value.length} / 5000`;
});

$("analyze-btn").addEventListener("click", async () => {
    const text = $("review").value.trim();
    hideError();

    if (text.length < 2) {
        showError("Please enter at least 2 characters to analyze.");
        return;
    }

    const button = $("analyze-btn");
    button.disabled = true;
    button.textContent = "Analyzing...";

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || "Prediction failed.");
        }

        $("result").hidden = false;
        $("result-badge").className = `badge ${result.sentiment}`;
        $("result-badge").textContent = result.sentiment;
        $("result-title").textContent =
            result.sentiment.charAt(0).toUpperCase() +
            result.sentiment.slice(1) + " sentiment";
        $("result-text").textContent = result.text;
        $("confidence").textContent = `${result.confidence.toFixed(2)}%`;
        $("confidence-bar").style.width =
            `${Math.min(100, Math.max(0, result.confidence))}%`;

        const probabilities = $("probabilities");
        probabilities.replaceChildren();

        ["positive", "negative", "neutral"].forEach(label => {
            const item = document.createElement("div");
            item.className = "prob-item";

            const name = document.createElement("span");
            name.textContent = label.charAt(0).toUpperCase() + label.slice(1);

            const value = document.createElement("strong");
            value.textContent = `${result.probabilities[label] ?? 0}%`;
            value.style.color = colors[label];

            item.append(name, value);
            probabilities.appendChild(item);
        });

        await refreshDashboard();
    } catch (error) {
        showError(error.message || "Could not connect to the API.");
    } finally {
        button.disabled = false;
        button.innerHTML = "✦ &nbsp; Analyze sentiment";
    }
});

refreshDashboard().catch(error => showError(error.message));
