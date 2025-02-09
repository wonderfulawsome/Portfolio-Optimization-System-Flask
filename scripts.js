document.getElementById('optimize-button').addEventListener('click', function() {
  const formData = {
    PER: document.getElementById('PER').value,
    DividendYield: document.getElementById('DividendYield').value,
    Beta: document.getElementById('Beta').value,
    RSI: document.getElementById('RSI').value,
    volume: document.getElementById('volume').value,  // ID 수정됨
    Volatility: document.getElementById('Volatility').value,
  };

  // 필수 입력값이 빠지면 경고
  for (const key in formData) {
    if (formData[key] === "") {
      alert(`${key} 값을 선택해주세요.`);
      return;
    }
  }

  // 상대 경로를 사용하여 백엔드 API 호출 (배포 환경에 따라 올바르게 작동)
  fetch("/optimize", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(formData)
  })
  .then(response => response.json())
  .then(data => {
    document.getElementById("result").innerText =
      "추천 기업: " + data.optimized_companies.join(", ");
  })
  .catch(error => console.error("Error:", error));
});
