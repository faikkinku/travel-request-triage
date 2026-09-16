def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_index_page_loads(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "New client request" in response.text


def test_guardrail_check_flags_a_legal_threat(client):
    response = client.post(
        "/api/guardrail-check",
        json={"message": "I am disputing this charge with my bank."},
    )
    assert response.status_code == 200
    body = response.json()
    assert "legal or chargeback threat" in body["overrides"]


def test_guardrail_check_is_clean_for_a_benign_message(client):
    response = client.post(
        "/api/guardrail-check",
        json={"message": "Can I add a checked bag to my upcoming trip?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["overrides"] == []
    assert body["injection_attempts"] == []


def test_review_queue_requires_login(client):
    response = client.get("/review", follow_redirects=False)
    assert response.status_code == 401


def test_review_queue_rejects_non_manager(client):
    client.post("/login", data={"username": "agent1", "password": "agent123"})
    response = client.get("/review", follow_redirects=False)
    assert response.status_code == 403


def test_review_queue_allows_manager(client):
    client.post("/login", data={"username": "manager", "password": "manager123"})
    response = client.get("/review", follow_redirects=False)
    assert response.status_code == 200


def test_login_rejects_wrong_password(client):
    response = client.post("/login", data={"username": "manager", "password": "wrong"})
    assert response.status_code == 401


def test_broken_intake_demonstrates_the_contract_rejects_nonsense(client):
    response = client.post("/api/intake/broken")
    assert response.status_code == 422
    errors = response.json()
    assert isinstance(errors, list)
    assert len(errors) >= 3  # request_type, urgency, confidence, escalate all invalid


def test_api_intake_without_key_returns_service_unavailable(client):
    # ANTHROPIC_API_KEY is unset in the test environment, so the model call itself
    # should fail cleanly rather than crash the request.
    response = client.post("/api/intake", json={"message": "Can I add a checked bag?"})
    assert response.status_code == 503
