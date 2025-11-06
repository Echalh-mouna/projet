document.getElementById("predictForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const resultDiv = document.getElementById("result");
    resultDiv.innerHTML = "⏳ Analyse en cours...";
  
    try {
      const res = await fetch("/predict", {
        method: "POST",
        body: formData
      });
      const data = await res.json();
  
      if (data.status === "ok") {
        resultDiv.innerHTML = `
          ✅ <strong>Analyse terminée.</strong><br>
          <a href="/results" class="btn btn-register mt-3">Voir les résultats</a>
        `;
      } else {
        resultDiv.innerHTML = `❌ Erreur: ${data.error}`;
      }
    } catch (err) {
      resultDiv.innerHTML = "⚠️ Erreur lors de la requête.";
    }
  });
  