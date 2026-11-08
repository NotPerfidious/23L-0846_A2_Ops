import pytest
from app import app, allowed_file

@pytest.fixture
def client():
    # Configure Flask for testing
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_check(client):
    """Ensure the API health endpoint responds correctly."""
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json == {"status": "healthy", "service": "VisionGuard API"}

def test_allowed_file_extensions():
    """Verify that only authorized image formats pass the security check."""
    assert allowed_file("valid_artwork.png") == True
    assert allowed_file("photo.JPG") == True
    assert allowed_file("photo.jpeg") == True
    assert allowed_file("malicious_script.sh") == False
    assert allowed_file("document.pdf") == False
    assert allowed_file("no_extension_file") == False